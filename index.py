"""
Tasks 1–3: Full Dataset Pipeline  (streaming rewrite, Cloudflare R2 edition)
-----------------------------------------------------------------------------
Polars >= 1.0 required (tested on 1.40.1)
polars[cloud] required for S3/R2 streaming support.

Key changes:
  • Input files are read directly from Cloudflare R2 (S3-compatible)
  • .env file is loaded automatically for R2 credentials
  • Tasks are skipped gracefully when required files are not yet in R2
  • A diagnostic run is performed first to confirm R2 connectivity
  • Task 1: .collect() replaced with .sink_parquet() — never loads full join into RAM
  • Task 2: reads from the Task 1 output file, uses streaming collect
  • Task 3: sink_parquet for context, chunked encoding loop unchanged
  • POLARS_MAX_THREADS env var set to avoid using all cores at once

Required .env variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

Run:
    python3 index.py
"""

import gc
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.exceptions import ClientError
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer

# ── tunables ───────────────────────────────────────────────────────────────
ENCODE_BATCH = 64     # sentences per model.encode() call — lower = less RAM
CHUNK_SIZE   = 512    # rows per iteration in Task 3 encoding loop
# ──────────────────────────────────────────────────────────────────────────

# Limit Polars thread pool so it doesn't try to allocate RAM for all cores
os.environ.setdefault("POLARS_MAX_THREADS", "4")

BASE = Path(__file__).parent

# ── Cloudflare R2 credentials (loaded from .env) ───────────────────────────
R2_ACCOUNT_ID   = os.environ["NUXT_R2_ACCOUNT_ID"]
R2_ACCESS_KEY   = os.environ["NUXT_R2_ACCESS_KEY_ID"]
R2_SECRET_KEY   = os.environ["NUXT_R2_SECRET_ACCESS_KEY"]
R2_BUCKET       = os.environ["NUXT_R2_BUCKET_NAME"]
R2_ENDPOINT     = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Storage options forwarded to every pl.scan_parquet() call
STORAGE_OPTIONS = {
    "aws_access_key_id":     R2_ACCESS_KEY,
    "aws_secret_access_key": R2_SECRET_KEY,
    "endpoint_url":          R2_ENDPOINT,
    "region":                "auto",
}

# boto3 client — used only for head_object existence checks
_s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto",
)

# S3 keys (object names inside the bucket)
PAT_KEY  = "FullSampleGloria_Pat_GlinerLabels_16042026.parquet"
LINK_KEY = "FullSampleGloria_Link_PmidOa_16042026.parquet"
PMED_KEY = "FullSampleGloria_Pmed_GlinerLabels_16042026.parquet"

# S3 URIs used by Polars
PAT_PATH  = f"s3://{R2_BUCKET}/{PAT_KEY}"
LINK_PATH = f"s3://{R2_BUCKET}/{LINK_KEY}"
PMED_PATH = f"s3://{R2_BUCKET}/{PMED_KEY}"

OUT_DIR = BASE / "output"
VIZ_DIR = BASE / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"R2 endpoint   : {R2_ENDPOINT}")
print(f"R2 bucket     : {R2_BUCKET}")
print(f"Output dir    : {OUT_DIR}")
print(f"Visualization : {VIZ_DIR}")
print(f"Polars version: {pl.__version__}")


# ── Existence check ─────────────────────────────────────────────────────────
def _r2_exists(key: str) -> bool:
    """Return True if *key* exists in the R2 bucket."""
    try:
        _s3.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except ClientError:
        return False


print("\n=== Checking R2 file availability ===")
avail: dict[str, bool] = {}
for label, key in [("PAT", PAT_KEY), ("LINK", LINK_KEY), ("PMED", PMED_KEY)]:
    found = _r2_exists(key)
    avail[label] = found
    status = "found" if found else "NOT FOUND — upload to R2 before running"
    print(f"  {'✓' if found else '✗'} {label}: {key}  →  {status}")

if avail["PMED"] and not (avail["PAT"] and avail["LINK"]):
    print("\n[INFO] Only PMED is available. Tasks 1-3 require all three files.")
    print("       Printing PMED schema/row-count as a connectivity smoke-test.")
    pmed_info = pl.scan_parquet(PMED_PATH, storage_options=STORAGE_OPTIONS).fetch(5)
    print(f"       PMED schema   : {pmed_info.schema}")
    print(f"       PMED sample   :\n{pmed_info}")
    sys.exit(0)

if not (avail["PAT"] and avail["LINK"] and avail["PMED"]):
    print("\n[ERROR] One or more required files are missing from R2. Exiting.")
    sys.exit(1)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.1f}s"


# ===========================================================================
# Shared lazy frames
# ===========================================================================
link_clean_lf = (
    pl.scan_parquet(LINK_PATH, storage_options=STORAGE_OPTIONS)
    .filter(pl.col("pmid").is_not_null())
    .with_columns(
        pl.col("pmid").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("pmid_num")
    )
    .filter(pl.col("pmid_num").is_not_null())
    .select([pl.col("patent_id"), pl.col("pmid_num").alias("pmid")])
    .unique()
)

pmed_terms_lf = (
    pl.scan_parquet(PMED_PATH, storage_options=STORAGE_OPTIONS)
    .select([pl.col("pmid").cast(pl.Int64), "term"])
)


# ===========================================================================
# TASK 1 — Identify Focal Terms
# Uses sink_parquet() — the join result is NEVER loaded into RAM
# ===========================================================================
print("\n=== TASK 1: Focal Terms ===")
t0 = time.time()

FOCAL_PATH = OUT_DIR / "focal_terms_full.parquet"

pat_terms_lf = (
    pl.scan_parquet(PAT_PATH, storage_options=STORAGE_OPTIONS)
    .select(["patent_id", "term"])
    .group_by(["patent_id", "term"])
    .agg(pl.len().alias("freq_in_patent"))
)

cited_term_counts_lf = (
    pmed_terms_lf
    .join(link_clean_lf, on="pmid", how="inner")
    .group_by(["patent_id", "term"])
    .agg(pl.len().alias("freq_in_cited_papers"))
)

print("Sinking focal terms to parquet (streaming — no RAM spike) ...")
(
    pat_terms_lf
    .join(cited_term_counts_lf, on=["patent_id", "term"], how="inner")
    .rename({"term": "focal_term"})
    .sink_parquet(FOCAL_PATH)
)
print(f"Done in {_elapsed(t0)}")

# Read back just the stats we need (small aggregation — safe to collect)
stats = (
    pl.scan_parquet(FOCAL_PATH)
    .select([
        pl.len().alias("n_rows"),
        pl.col("patent_id").n_unique().alias("n_patents"),
        pl.col("focal_term").n_unique().alias("n_terms"),
    ])
    .collect()
)
print(f"Focal term rows           : {stats['n_rows'][0]:,}")
print(f"Unique patents w/ focal   : {stats['n_patents'][0]:,}")
print(f"Unique focal terms        : {stats['n_terms'][0]:,}")
del stats
gc.collect()

# Also write CSV (stream from the parquet, never load fully)
print("Writing CSV ...")
pl.scan_parquet(FOCAL_PATH).sink_csv(OUT_DIR / "focal_terms_full.csv")
print("Saved: focal_terms_full.parquet / .csv")


# ===========================================================================
# TASK 2 — Measure Overlap Intensity
# ===========================================================================
print("\n=== TASK 2: Overlap Intensity ===")
t0 = time.time()

counts = (
    pl.scan_parquet(FOCAL_PATH)
    .group_by("patent_id")
    .agg(pl.col("focal_term").n_unique().alias("num_focal_terms"))
    .sort("patent_id")
    .collect()
)
total = len(counts)

if total == 0:
    print("No patents with focal terms — skipping Task 2.")
else:
    values     = counts["num_focal_terms"].to_numpy()
    mean_val   = float(values.mean())
    median_val = float(np.median(values))
    std_val    = float(values.std())
    min_val    = int(values.min())
    max_val    = int(values.max())
    n_one      = int((counts["num_focal_terms"] == 1).sum())

    print(f"Unique patents with focal terms   : {total:,}")
    print(f"Mean focal terms per patent       : {mean_val:.2f}")
    print(f"Median                            : {median_val:.1f}")
    print(f"Std Dev                           : {std_val:.2f}")
    print(f"Min / Max                         : {min_val} / {max_val}")
    print(f"Patents with exactly 1 focal term : {n_one} ({n_one/total*100:.1f}%)")
    print(f"Done in {_elapsed(t0)}")

    counts.write_csv(OUT_DIR / "focal_term_counts_per_patent_full.csv")

    pl.DataFrame({
        "statistic": ["mean", "median", "std", "min", "max",
                      "n_patents", "pct_exactly_1"],
        "value": [mean_val, median_val, std_val, float(min_val), float(max_val),
                  float(total), float(n_one / total * 100)],
    }).write_csv(OUT_DIR / "task2_summary_stats_full.csv")
    print("Saved: focal_term_counts_per_patent_full.csv, task2_summary_stats_full.csv")

    # Histogram — cap bins to avoid memory issues with huge max_val
    bin_range = range(1, min(max_val + 2, 202))
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(values, bins=list(bin_range), color="steelblue",
             edgecolor="white", align="left")
    ax1.axvline(mean_val,   color="red",    linestyle="--",
                label=f"Mean ({mean_val:.2f})")
    ax1.axvline(median_val, color="orange", linestyle="--",
                label=f"Median ({median_val:.0f})")
    ax1.set_title("Histogram of Focal Terms per Patent (Full Dataset)")
    ax1.set_xlabel("Number of Focal Terms")
    ax1.set_ylabel("Number of Patents")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "histogram_focal_terms_full.png", dpi=150)
    plt.close()

    # Density plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if len(values) > 1:
        kde = gaussian_kde(values)
        x = np.linspace(values.min() - 0.5, values.max() + 0.5, 300)
        ax2.plot(x, kde(x), color="steelblue")
    ax2.axvline(mean_val,   color="red",    linestyle="--",
                label=f"Mean ({mean_val:.2f})")
    ax2.axvline(median_val, color="orange", linestyle="--",
                label=f"Median ({median_val:.0f})")
    ax2.set_title("Density Plot of Focal Terms per Patent (Full Dataset)")
    ax2.set_xlabel("Number of Focal Terms")
    ax2.set_ylabel("Density")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "density_focal_terms_full.png", dpi=150)
    plt.close()
    print("Saved: histogram_focal_terms_full.png, density_focal_terms_full.png")

del counts, values
gc.collect()


# ===========================================================================
# TASK 3 — Semantic Context Comparison
# ===========================================================================
print("\n=== TASK 3: Semantic Context Comparison ===")

focal_lf = (
    pl.scan_parquet(FOCAL_PATH)
    .select(["patent_id", "focal_term"])
    .unique()
)

n_pairs = focal_lf.select(pl.len()).collect().item()
if n_pairs == 0:
    print("No focal terms — skipping Task 3.")
else:
    t0 = time.time()

    patent_contexts_lf = (
        pl.scan_parquet(PAT_PATH, storage_options=STORAGE_OPTIONS)
        .select(["patent_id", "term"])
        .join(focal_lf, on="patent_id", how="inner")
        .filter(pl.col("term") != pl.col("focal_term"))
        .group_by(["patent_id", "focal_term"])
        .agg(pl.col("term").alias("patent_context"))
    )

    paper_contexts_lf = (
        focal_lf
        .join(link_clean_lf, on="patent_id", how="inner")
        .join(
            pmed_terms_lf.rename({"term": "focal_term"}),
            on=["pmid", "focal_term"],
            how="inner",
        )
        .select(["patent_id", "focal_term", "pmid"])
        .unique()
        .join(pmed_terms_lf, on="pmid", how="inner")
        .filter(pl.col("term") != pl.col("focal_term"))
        .group_by(["patent_id", "focal_term"])
        .agg(pl.col("term").unique().alias("paper_context"))
    )

    context_sink_path = OUT_DIR / "focal_term_context_full.parquet"
    print("Sinking context join to parquet ...")
    (
        focal_lf
        .join(patent_contexts_lf, on=["patent_id", "focal_term"], how="left")
        .join(paper_contexts_lf,  on=["patent_id", "focal_term"], how="left")
        .sink_parquet(context_sink_path)
    )
    print(f"Context parquet written [{_elapsed(t0)}]")

    # CSV version
    (
        pl.scan_parquet(context_sink_path)
        .with_columns([
            pl.col("patent_context").map_elements(
                lambda x: " ".join(x) if x is not None else "",
                return_dtype=pl.String
            ),
            pl.col("paper_context").map_elements(
                lambda x: " ".join(x) if x is not None else "",
                return_dtype=pl.String
            ),
        ])
        .sink_csv(OUT_DIR / "focal_term_context_full.csv")
    )
    print("Saved: focal_term_context_full.parquet / .csv")

    # Load model
    print("\nLoading sentence-transformers model (CPU) ...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    total_rows = (
        pl.scan_parquet(context_sink_path).select(pl.len()).collect().item()
    )
    print(f"Encoding {total_rows:,} rows in chunks of {CHUNK_SIZE} "
          f"(encode_batch={ENCODE_BATCH}) ...")
    t0 = time.time()

    all_patent_ids  = []
    all_focal_terms = []
    all_sims        = []

    for start in range(0, total_rows, CHUNK_SIZE):
        chunk = (
            pl.scan_parquet(context_sink_path)
            .slice(start, CHUNK_SIZE)
            .collect()
        )

        pat_sents = [
            f"{r['focal_term']} {' '.join(r['patent_context'] or [])}"
            for r in chunk.iter_rows(named=True)
        ]
        pap_sents = [
            f"{r['focal_term']} {' '.join(r['paper_context'] or [])}"
            for r in chunk.iter_rows(named=True)
        ]

        pat_emb = model.encode(
            pat_sents, show_progress_bar=False,
            batch_size=ENCODE_BATCH, normalize_embeddings=True
        )
        pap_emb = model.encode(
            pap_sents, show_progress_bar=False,
            batch_size=ENCODE_BATCH, normalize_embeddings=True
        )

        sims_chunk = (pat_emb * pap_emb).sum(axis=1).astype(np.float32)

        all_patent_ids.extend(chunk["patent_id"].to_list())
        all_focal_terms.extend(chunk["focal_term"].to_list())
        all_sims.append(sims_chunk)

        del chunk, pat_sents, pap_sents, pat_emb, pap_emb, sims_chunk
        gc.collect()

        done = min(start + CHUNK_SIZE, total_rows)
        print(f"  {done:,} / {total_rows:,}  [{_elapsed(t0)}]", end="\r")

    print()

    sims_all = np.concatenate(all_sims)
    cosine_df = pl.DataFrame({
        "patent_id":         all_patent_ids,
        "focal_term":        all_focal_terms,
        "cosine_similarity": sims_all.tolist(),
    })
    del all_patent_ids, all_focal_terms, all_sims, sims_all
    gc.collect()

    cosine_df.write_parquet(OUT_DIR / "cosine_similarity_results_full.parquet")
    cosine_df.write_csv(OUT_DIR / "cosine_similarity_results_full.csv")
    print("Saved: cosine_similarity_results_full.parquet / .csv")

    sim = cosine_df["cosine_similarity"].drop_nulls().to_numpy()
    del cosine_df
    gc.collect()

    if len(sim) == 0:
        print("No cosine similarity values — skipping summary and plot.")
        pl.DataFrame({
            "statistic": ["mean", "median", "std", "min", "max"],
            "value": [None, None, None, None, None],
        }).write_csv(OUT_DIR / "task3_cosine_summary_stats_full.csv")
    else:
        mean_sim   = float(sim.mean())
        median_sim = float(np.median(sim))
        std_sim    = float(sim.std())
        min_sim    = float(sim.min())
        max_sim    = float(sim.max())

        print("\n=== Cosine Similarity Summary Statistics ===")
        print(f"  Mean   : {mean_sim:.3f}")
        print(f"  Median : {median_sim:.3f}")
        print(f"  Std    : {std_sim:.3f}")
        print(f"  Min    : {min_sim:.3f}")
        print(f"  Max    : {max_sim:.3f}")

        pl.DataFrame({
            "statistic": ["mean", "median", "std", "min", "max"],
            "value": [mean_sim, median_sim, std_sim, min_sim, max_sim],
        }).write_csv(OUT_DIR / "task3_cosine_summary_stats_full.csv")
        print("Saved: task3_cosine_summary_stats_full.csv")

        plt.figure(figsize=(8, 4))
        plt.hist(sim, bins=50, color="steelblue", edgecolor="black")
        plt.axvline(mean_sim,   color="red",    linestyle="--",
                    label=f"Mean ({mean_sim:.3f})")
        plt.axvline(median_sim, color="orange", linestyle="--",
                    label=f"Median ({median_sim:.3f})")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.title("Distribution of Semantic Similarity\n"
                  "(Full Dataset — patent vs. paper context per focal term)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "cosine_similarity_distribution_full.png", dpi=150)
        plt.close()
        print("Saved: cosine_similarity_distribution_full.png")

print("\nDone.")