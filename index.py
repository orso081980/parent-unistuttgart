"""
Gloria Patent Analysis Pipeline — Production Grade
====================================================

ARCHITECTURE
  Task 1  PAT + LINK + PMED  →  focal_terms_full.parquet   (streamed, never in RAM)
  Task 2  focal_terms         →  overlap statistics + plots
  Task 3  focal_terms         →  cosine similarity + plots   (single parquet read,
                                                              chunked CPU encoding)

PERFORMANCE DESIGN DECISIONS
  • Every large join uses LazyFrame + sink_parquet()  (Polars streaming engine)
  • link_clean and pmed_terms are materialised once because they are used in
    multiple joins; re-scanning them lazily would cause redundant I/O
  • context parquet is read ONCE into RAM (it is aggregated → much smaller than
    inputs), then split into in-memory chunks for sentence-transformer encoding
  • CHUNK_SIZE only controls how many sentences go to model.encode() per call;
    it no longer triggers repeated parquet reads (the old bottleneck)
  • gc.collect() is called after each encoding chunk to release numpy arrays
  • All CSVs are written from a single lazy scan after the parquet exists

TUNING FOR CONSTRAINED SERVERS
  POLARS_MAX_THREADS   2–4 on shared HPC nodes
  ENCODE_BATCH         32  on single-core / low-RAM servers (default 64)
  CHUNK_SIZE           256–512 rows per model.encode() call (default 512)
  If 502 timeout:      halve CHUNK_SIZE and ENCODE_BATCH

REQUIREMENTS
  polars >= 1.0.0   (streaming / sink_parquet support)
  sentence-transformers
  scipy, matplotlib, numpy
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def _sep(title: str = "") -> None:
    bar = "═" * 70
    log.info(bar)
    if title:
        log.info(f"  {title}")
        log.info(bar)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.1f} s"


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

ENCODE_BATCH: int = 64    # sentences per model.encode() call
CHUNK_SIZE:   int = 512   # rows per encoding iteration (in-memory split only)
MODEL_NAME:   str = "sentence-transformers/all-MiniLM-L6-v2"

os.environ.setdefault("POLARS_MAX_THREADS", "4")
log.info(f"Polars threads : {os.environ['POLARS_MAX_THREADS']}")
log.info(f"Polars version : {pl.__version__}")
log.info(f"ENCODE_BATCH   : {ENCODE_BATCH}")
log.info(f"CHUNK_SIZE     : {CHUNK_SIZE}")


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE     = Path(__file__).parent
OUT_DIR  = BASE / "output"
VIZ_DIR  = BASE / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

PAT_PATH  = BASE / "FullSampleGloria_Pat_GlinerLabels_16042026.parquet"
LINK_PATH = BASE / "FullSampleGloria_Link_PmidOa_16042026.parquet"
PMED_PATH = BASE / "FullSampleGloria_Pmed_GlinerLabels_16042026.parquet"

# Intermediate / output artefacts
FOCAL_PATH   = OUT_DIR / "focal_terms_full.parquet"
CONTEXT_PATH = OUT_DIR / "focal_term_context_full.parquet"
COSINE_PATH  = OUT_DIR / "cosine_similarity_results_full.parquet"

for p in (PAT_PATH, LINK_PATH, PMED_PATH):
    if not p.exists():
        log.error(f"Missing input file: {p}")
        sys.exit(1)

log.info("All input files present — starting pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: in-memory DataFrame chunker (no repeated disk reads)
# ─────────────────────────────────────────────────────────────────────────────

def iter_chunks(df: pl.DataFrame, size: int) -> Iterator[pl.DataFrame]:
    """Yield non-overlapping slices of *df* without copying data."""
    for start in range(0, len(df), size):
        yield df.slice(start, size)


# ─────────────────────────────────────────────────────────────────────────────
# PRE-MATERIALISE SMALL TABLES
# ─────────────────────────────────────────────────────────────────────────────
# link_clean and pmed_terms are reference tables reused in several joins.
# Collecting them once avoids redundant parquet scans and lets Polars use
# hash-join with an in-memory build side (fastest path).
# If either is unexpectedly huge, convert back to scan_parquet + lazy joins.

log.info("Materialising link and pmed reference tables …")
t_ref = time.time()

link_clean: pl.DataFrame = (
    pl.scan_parquet(LINK_PATH)
    .filter(pl.col("pmid").is_not_null())
    .with_columns(
        pl.col("pmid").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("pmid_num")
    )
    .filter(pl.col("pmid_num").is_not_null())
    .select(
        pl.col("patent_id"),
        pl.col("pmid_num").alias("pmid"),
    )
    .unique()
    .collect()
)

pmed_terms: pl.DataFrame = (
    pl.scan_parquet(PMED_PATH)
    .select(pl.col("pmid").cast(pl.Int64), "term")
    .collect()
)

log.info(
    f"link_clean : {len(link_clean):,} rows   "
    f"pmed_terms : {len(pmed_terms):,} rows   "
    f"[{_elapsed(t_ref)}]"
)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — FOCAL TERMS
# focal term = a term that appears in BOTH the patent AND at least one
#              cited PubMed paper linked to that patent.
# ═════════════════════════════════════════════════════════════════════════════

_sep("TASK 1 — Focal Terms  (patent ∩ cited-paper terms)")
t1 = time.time()

# ── Patent-side: term frequencies per patent (streamed)
pat_terms_lf = (
    pl.scan_parquet(PAT_PATH)
    .select("patent_id", "term")
    .group_by("patent_id", "term")
    .agg(pl.len().alias("freq_in_patent"))
)

# ── Cited-paper side: use materialised tables so the join build side fits RAM
#    We do a lazy join against the collected DataFrames (Polars accepts both)
cited_term_counts_lf = (
    pmed_terms
    .lazy()
    .join(link_clean.lazy(), on="pmid", how="inner")
    .group_by("patent_id", "term")
    .agg(pl.len().alias("freq_in_cited_papers"))
)

# ── Inner join → keep only overlapping terms → stream to disk
log.info("Streaming focal-term join → parquet …")
(
    pat_terms_lf
    .join(cited_term_counts_lf, on=["patent_id", "term"], how="inner")
    .rename({"term": "focal_term"})
    .sink_parquet(FOCAL_PATH)
)

# ── Summarise (small aggregation on the output, safe to collect)
stats = (
    pl.scan_parquet(FOCAL_PATH)
    .select(
        pl.len().alias("n_rows"),
        pl.col("patent_id").n_unique().alias("n_patents"),
        pl.col("focal_term").n_unique().alias("n_terms"),
    )
    .collect()
)
log.info(f"  Focal-term rows   : {stats['n_rows'][0]:,}")
log.info(f"  Unique patents    : {stats['n_patents'][0]:,}")
log.info(f"  Unique terms      : {stats['n_terms'][0]:,}")
del stats
gc.collect()

# ── CSV export (single lazy re-scan)
pl.scan_parquet(FOCAL_PATH).sink_csv(OUT_DIR / "focal_terms_full.csv")
log.info(f"✓ Task 1 done [{_elapsed(t1)}]  →  focal_terms_full.parquet / .csv")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — OVERLAP INTENSITY
# ═════════════════════════════════════════════════════════════════════════════

_sep("TASK 2 — Overlap Intensity (focal terms per patent)")
t2 = time.time()

counts: pl.DataFrame = (
    pl.scan_parquet(FOCAL_PATH)
    .group_by("patent_id")
    .agg(pl.col("focal_term").n_unique().alias("num_focal_terms"))
    .sort("patent_id")
    .collect()
)

total = len(counts)
if total == 0:
    log.warning("No patents with focal terms — skipping Task 2.")
else:
    values = counts["num_focal_terms"].to_numpy()
    mean_val   = float(values.mean())
    median_val = float(np.median(values))
    std_val    = float(values.std())
    min_val    = int(values.min())
    max_val    = int(values.max())
    # FIX: column name was "num_focal_term" (typo) in original code
    n_one      = int((counts["num_focal_terms"] == 1).sum())

    log.info(f"  Patents with focal terms     : {total:,}")
    log.info(f"  Mean focal terms / patent    : {mean_val:.2f}")
    log.info(f"  Median                       : {median_val:.1f}")
    log.info(f"  Std Dev                      : {std_val:.2f}")
    log.info(f"  Min / Max                    : {min_val} / {max_val}")
    log.info(f"  Patents with exactly 1 term  : {n_one:,}  ({n_one/total*100:.1f}%)")

    counts.write_csv(OUT_DIR / "focal_term_counts_per_patent_full.csv")

    pl.DataFrame({
        "statistic": ["mean", "median", "std", "min", "max", "n_patents", "pct_exactly_1"],
        "value": [mean_val, median_val, std_val, float(min_val), float(max_val),
                  float(total), float(n_one / total * 100)],
    }).write_csv(OUT_DIR / "task2_summary_stats_full.csv")

    # ── Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = list(range(1, min(max_val + 2, 202)))
    ax.hist(values, bins=bins, color="#3a7ebf", edgecolor="white", linewidth=0.4, align="left")
    ax.axvline(mean_val,   color="#e05c1a", linestyle="--", linewidth=1.5,
               label=f"Mean   {mean_val:.2f}")
    ax.axvline(median_val, color="#f5b800", linestyle="--", linewidth=1.5,
               label=f"Median {median_val:.0f}")
    ax.set_title("Focal Terms per Patent — Full Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Focal Terms")
    ax.set_ylabel("Number of Patents")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "histogram_focal_terms_full.png", dpi=150)
    plt.close()

    # ── Density plot
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(values) > 1:
        kde = gaussian_kde(values)
        x   = np.linspace(min_val - 0.5, max_val + 0.5, 500)
        ax.plot(x, kde(x), color="#3a7ebf", linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.15, color="#3a7ebf")
    ax.axvline(mean_val,   color="#e05c1a", linestyle="--", linewidth=1.5,
               label=f"Mean   {mean_val:.2f}")
    ax.axvline(median_val, color="#f5b800", linestyle="--", linewidth=1.5,
               label=f"Median {median_val:.0f}")
    ax.set_title("Density — Focal Terms per Patent — Full Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Focal Terms")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "density_focal_terms_full.png", dpi=150)
    plt.close()

    log.info(f"✓ Task 2 done [{_elapsed(t2)}]  →  stats + 2 plots saved")

del counts
try:
    del values
except NameError:
    pass
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — SEMANTIC CONTEXT COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
# For each (patent_id, focal_term) pair, we build two context strings:
#   patent_context  = all other terms in the patent that contain this focal term
#   paper_context   = all other terms in the cited PubMed papers that contain it
# Then we encode both with all-MiniLM-L6-v2 and compute cosine similarity.
#
# CRITICAL OPTIMISATION vs. original code
#   Original: pl.scan_parquet().slice(i, CHUNK_SIZE).collect() inside a loop
#             → re-reads the parquet file on every iteration (O(N²) I/O)
#   Fixed:    read the context parquet ONCE with pl.read_parquet(),
#             then iterate over in-memory slices  (O(N) I/O)
# ═════════════════════════════════════════════════════════════════════════════

_sep("TASK 3 — Semantic Context Comparison")
t3 = time.time()

focal_pairs_lf = (
    pl.scan_parquet(FOCAL_PATH)
    .select("patent_id", "focal_term")
    .unique()
)

n_pairs: int = focal_pairs_lf.select(pl.len()).collect().item()
if n_pairs == 0:
    log.warning("No focal terms — skipping Task 3.")
    sys.exit(0)

log.info(f"Focal-term pairs to process : {n_pairs:,}")

# ── Build patent-side context (lazy, streamed to disk)
log.info("Step 1/4  Building patent context …")
patent_ctx_lf = (
    pl.scan_parquet(PAT_PATH)
    .select("patent_id", "term")
    .join(focal_pairs_lf, on="patent_id", how="inner")
    .filter(pl.col("term") != pl.col("focal_term"))
    .group_by("patent_id", "focal_term")
    .agg(pl.col("term").alias("patent_context"))
)

# ── Build paper-side context
#    Use the materialised DataFrames as the build side for the hash-joins.
#    The focal-pairs lazy frame is the probe side (potentially large).
log.info("Step 2/4  Building paper context …")
paper_ctx_lf = (
    focal_pairs_lf
    .join(link_clean.lazy(), on="patent_id", how="inner")  # patent → pmids
    # keep only pmids that mention the focal term in pmed
    .join(
        pmed_terms.lazy().rename({"term": "focal_term"}),
        on=["pmid", "focal_term"],
        how="inner",
    )
    .select("patent_id", "focal_term", "pmid")
    .unique()
    # bring in all other terms from those pmids
    .join(pmed_terms.lazy(), on="pmid", how="inner")
    .filter(pl.col("term") != pl.col("focal_term"))
    .group_by("patent_id", "focal_term")
    .agg(pl.col("term").unique().alias("paper_context"))
)

# ── Sink combined context once (streaming → no RAM spike here)
log.info("Step 3/4  Sinking combined context → parquet …")
(
    focal_pairs_lf
    .join(patent_ctx_lf, on=["patent_id", "focal_term"], how="left")
    .join(paper_ctx_lf,  on=["patent_id", "focal_term"], how="left")
    .sink_parquet(CONTEXT_PATH)
)
log.info(f"  Context parquet written [{_elapsed(t3)}]")

# ── CSV export (lists → space-joined strings, single lazy scan)
(
    pl.scan_parquet(CONTEXT_PATH)
    .with_columns(
        pl.col("patent_context").map_elements(
            lambda x: " ".join(x) if x is not None else "",
            return_dtype=pl.String,
        ),
        pl.col("paper_context").map_elements(
            lambda x: " ".join(x) if x is not None else "",
            return_dtype=pl.String,
        ),
    )
    .sink_csv(OUT_DIR / "focal_term_context_full.csv")
)
log.info("  CSV version written")

# ── KEY FIX: read context parquet ONCE into RAM
#    The context table is aggregated (one row per focal-term pair) so it is
#    far smaller than the raw 500 MB inputs.  Reading it once avoids the
#    O(N²) parquet re-scan that made the original code take 2+ hours.
log.info("Step 4/4  Encoding with sentence-transformers …")
log.info(f"  Model       : {MODEL_NAME}")
log.info(f"  CHUNK_SIZE  : {CHUNK_SIZE}  (in-memory split, no disk re-reads)")
log.info(f"  ENCODE_BATCH: {ENCODE_BATCH}")

context_df: pl.DataFrame = pl.read_parquet(CONTEXT_PATH)
total_rows: int = len(context_df)
log.info(f"  Context rows loaded into RAM : {total_rows:,}")

model = SentenceTransformer(MODEL_NAME, device="cpu")
log.info("  Model loaded ✓")

t_enc = time.time()
all_patent_ids: list[str] = []
all_focal_terms: list[str] = []
all_sims: list[np.ndarray] = []

for chunk_i, chunk in enumerate(iter_chunks(context_df, CHUNK_SIZE)):
    rows = chunk.iter_rows(named=True)

    pat_sents: list[str] = []
    pap_sents: list[str] = []
    for r in rows:
        ft  = r["focal_term"]
        pat = r["patent_context"] or []
        pap = r["paper_context"]  or []
        pat_sents.append(f"{ft} {' '.join(pat)}")
        pap_sents.append(f"{ft} {' '.join(pap)}")

    pat_emb = model.encode(
        pat_sents,
        batch_size=ENCODE_BATCH,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    pap_emb = model.encode(
        pap_sents,
        batch_size=ENCODE_BATCH,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Cosine similarity: dot product of L2-normalised vectors
    sims = (pat_emb * pap_emb).sum(axis=1).astype(np.float32)

    all_patent_ids.extend(chunk["patent_id"].to_list())
    all_focal_terms.extend(chunk["focal_term"].to_list())
    all_sims.append(sims)

    del chunk, pat_sents, pap_sents, pat_emb, pap_emb, sims
    gc.collect()

    done = min((chunk_i + 1) * CHUNK_SIZE, total_rows)
    pct  = 100 * done / total_rows
    rate = done / max(time.time() - t_enc, 1)
    eta  = (total_rows - done) / max(rate, 1)
    log.info(
        f"  {done:>8,} / {total_rows:,}  ({pct:5.1f}%)  "
        f"[{_elapsed(t_enc)}]  ETA {eta/60:.1f} min"
    )

del context_df
gc.collect()

log.info(f"✓ Encoding complete [{_elapsed(t_enc)}]")

# ── Combine and save
sims_all: np.ndarray = np.concatenate(all_sims)
cosine_df = pl.DataFrame({
    "patent_id":        all_patent_ids,
    "focal_term":       all_focal_terms,
    "cosine_similarity": sims_all.tolist(),
})
del all_patent_ids, all_focal_terms, all_sims, sims_all
gc.collect()

cosine_df.write_parquet(COSINE_PATH)
cosine_df.write_csv(OUT_DIR / "cosine_similarity_results_full.csv")
log.info("✓ Saved cosine_similarity_results_full.parquet / .csv")

# ── Summary statistics
sim: np.ndarray = cosine_df["cosine_similarity"].drop_nulls().to_numpy()
del cosine_df
gc.collect()

if len(sim) == 0:
    log.warning("No cosine values — no summary or plot.")
    pl.DataFrame({
        "statistic": ["mean", "median", "std", "min", "max"],
        "value":     [None, None, None, None, None],
    }).write_csv(OUT_DIR / "task3_cosine_summary_stats_full.csv")
else:
    mean_sim   = float(sim.mean())
    median_sim = float(np.median(sim))
    std_sim    = float(sim.std())
    min_sim    = float(sim.min())
    max_sim    = float(sim.max())

    log.info("=== Cosine Similarity Summary ===")
    log.info(f"  Mean   : {mean_sim:.4f}")
    log.info(f"  Median : {median_sim:.4f}")
    log.info(f"  Std    : {std_sim:.4f}")
    log.info(f"  Min    : {min_sim:.4f}")
    log.info(f"  Max    : {max_sim:.4f}")

    pl.DataFrame({
        "statistic": ["mean", "median", "std", "min", "max"],
        "value":     [mean_sim, median_sim, std_sim, min_sim, max_sim],
    }).write_csv(OUT_DIR / "task3_cosine_summary_stats_full.csv")

    # ── Distribution plot (histogram + KDE overlay)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sim, bins=60, color="#3a7ebf", edgecolor="white",
            linewidth=0.3, density=True, alpha=0.7, label="Histogram (density)")
    if len(sim) > 1:
        kde = gaussian_kde(sim)
        x   = np.linspace(min_sim - 0.02, max_sim + 0.02, 500)
        ax.plot(x, kde(x), color="#1a4a7a", linewidth=2, label="KDE")
    ax.axvline(mean_sim,   color="#e05c1a", linestyle="--", linewidth=1.5,
               label=f"Mean   {mean_sim:.3f}")
    ax.axvline(median_sim, color="#f5b800", linestyle="--", linewidth=1.5,
               label=f"Median {median_sim:.3f}")
    ax.set_title(
        "Semantic Similarity Distribution — Full Dataset\n"
        "(patent context vs. cited-paper context per focal term)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "cosine_similarity_distribution_full.png", dpi=150)
    plt.close()
    log.info("✓ Saved cosine_similarity_distribution_full.png")

log.info(f"✓ Task 3 done [{_elapsed(t3)}]")


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────

_sep()
log.info("  ✓✓✓  PIPELINE COMPLETE  ✓✓✓")
_sep()

log.info("\nOutput files:")
for f in sorted(OUT_DIR.iterdir()):
    mb = f.stat().st_size / 1_048_576
    log.info(f"  {f.name:<55} {mb:>8.1f} MB")

log.info("\nVisualizations:")
for f in sorted(VIZ_DIR.iterdir()):
    kb = f.stat().st_size / 1024
    log.info(f"  {f.name:<55} {kb:>8.1f} KB")