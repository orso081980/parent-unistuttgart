"""
Gloria Patent Analysis: Full Dataset Pipeline (Streaming, Memory-Optimized)
===========================================================================

ARCHITECTURE:
  - Task 1: Stream PAT + LINK + PMED joins → focal_terms.parquet (never collects join)
  - Task 2: Aggregate focal term counts per patent → statistics + visualizations
  - Task 3: Compute semantic similarity (chunked encoding to manage model memory)

MEMORY STRATEGY FOR LARGE FILES (500MB–2GB):
  1. All intermediate joins use lazy frames until sink_parquet()
  2. .sink_parquet() streams results directly to disk (no RAM buffering)
  3. Model encoding is chunked (default 128 rows) with aggressive garbage collection
  4. Sentence-transformers model batching (ENCODE_BATCH) controls GPU/CPU memory

REQUIREMENTS:
  - Polars >= 1.0.0 (must support streaming)
  - For server deployment: increase swap space and reduce CHUNK_SIZE if OOM

DEPLOYMENT NOTES (University of Stuttgart / HPC servers):
  - Set POLARS_MAX_THREADS=2 if constrained
  - ENCODE_BATCH=32 on single-core CPUs (vs. 64 on multicore)
  - If 502 Gateway Timeout: reduce CHUNK_SIZE to 64 and ENCODE_BATCH to 16
  - Monitor /tmp disk space for Polars temp files
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────
# LOGGING SETUP (for server debugging)
# ─────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS (adjust for server memory constraints)
# ─────────────────────────────────────────────────────────────────────────
# Reduce these if server memory is < 8GB or timeouts occur (502 errors)
ENCODE_BATCH = 64    # Batch size for sentence-transformers.encode()
CHUNK_SIZE = 128     # Rows per loop in Task 3 (reduced from 512 for stability)

# Thread limiting prevents Polars from allocating memory for all cores at once
# Recommended: 2-4 threads on shared HPC systems
os.environ.setdefault("POLARS_MAX_THREADS", "4")
logger.info(f"Polars threads: {os.environ['POLARS_MAX_THREADS']}")


BASE = Path(__file__).parent

# Input parquet files (500MB–2GB each)
# These must exist before running. For server deployment:
#   - Use NFS mount if files are remote
#   - Or download to /scratch or /tmp first (check disk space!)
PAT_PATH = BASE / "FullSampleGloria_Pat_GlinerLabels_16042026.parquet"
LINK_PATH = BASE / "FullSampleGloria_Link_PmidOa_16042026.parquet"
PMED_PATH = BASE / "FullSampleGloria_Pmed_GlinerLabels_16042026.parquet"

# Output directories
OUT_DIR = BASE / "output"
VIZ_DIR = BASE / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Input files location: {BASE}")
logger.info(f"Output directory: {OUT_DIR}")
logger.info(f"Visualization directory: {VIZ_DIR}")
logger.info(f"Polars version: {pl.__version__}")

# ─────────────────────────────────────────────────────────────────────────
# VALIDATION: Check file existence before processing
# ─────────────────────────────────────────────────────────────────────────
required_files = [PAT_PATH, LINK_PATH, PMED_PATH]
missing = [f for f in required_files if not f.exists()]

if missing:
    logger.error(f"Missing input files: {missing}")
    logger.error("Please ensure all three parquet files are in the working directory.")
    sys.exit(1)

logger.info("All input files found. Starting pipeline...")


def _elapsed(t0: float) -> str:
    """Format elapsed time for logging."""
    s = time.time() - t0
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.1f}s"




# ─────────────────────────────────────────────────────────────────────────
# LAZY FRAMES (never fully loaded into memory)
# These are scanned lazily and will only materialize when sink_parquet() runs
# ─────────────────────────────────────────────────────────────────────────

logger.info("Loading lazy frames (parquet metadata only)...")

# LINK_PATH: Patent → PubMed ID mappings
# Filtering for valid pmid before downstream joins prevents large cartesian joins
link_clean_lf = (
    pl.scan_parquet(LINK_PATH)
    .filter(pl.col("pmid").is_not_null())
    # Extract numeric PMID (handles "PMID:12345" format)
    .with_columns(
        pl.col("pmid").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("pmid_num")
    )
    .filter(pl.col("pmid_num").is_not_null())
    .select([pl.col("patent_id"), pl.col("pmid_num").alias("pmid")])
    .unique()  # Remove duplicate mappings
)

# PMED_PATH: PubMed article terms (extracted by NER model)
# Kept as lazy frame; will be joined with LINK in Task 1
pmed_terms_lf = (
    pl.scan_parquet(PMED_PATH)
    .select([pl.col("pmid").cast(pl.Int64), "term"])
)




# ═════════════════════════════════════════════════════════════════════════
# TASK 1: IDENTIFY FOCAL TERMS
# ═════════════════════════════════════════════════════════════════════════
# Focal terms = terms that appear in BOTH:
#   (a) patent documents (PAT_PATH)
#   (b) cited PubMed papers (LINK + PMED)
#
# Strategy: Use sink_parquet() to stream the join result directly to disk
# without ever loading the full join result into RAM.
# ═════════════════════════════════════════════════════════════════════════

logger.info("\n" + "="*70)
logger.info("TASK 1: Identifying Focal Terms (patent ∩ pubmed terms)")
logger.info("="*70)
t0 = time.time()

FOCAL_PATH = OUT_DIR / "focal_terms_full.parquet"

try:
    # PAT_PATH terms grouped by patent
    logger.info("Processing patent terms...")
    pat_terms_lf = (
        pl.scan_parquet(PAT_PATH)
        .select(["patent_id", "term"])
        .group_by(["patent_id", "term"])
        .agg(pl.len().alias("freq_in_patent"))
    )

    # Cited paper terms (via LINK mapping)
    logger.info("Processing cited paper terms...")
    cited_term_counts_lf = (
        pmed_terms_lf
        .join(link_clean_lf, on="pmid", how="inner")
        .group_by(["patent_id", "term"])
        .agg(pl.len().alias("freq_in_cited_papers"))
    )

    # Focal terms: inner join keeps only overlapping terms
    logger.info("Computing focal terms join (streaming to parquet)...")
    (
        pat_terms_lf
        .join(cited_term_counts_lf, on=["patent_id", "term"], how="inner")
        .rename({"term": "focal_term"})
        .sink_parquet(FOCAL_PATH)  # ← Critical: streams to disk, never loads in RAM
    )
    logger.info(f"✓ Focal terms written to {FOCAL_PATH.name} [{_elapsed(t0)}]")

    # Read back minimal aggregations for reporting (small result, safe to collect)
    logger.info("Computing summary statistics...")
    stats = (
        pl.scan_parquet(FOCAL_PATH)
        .select([
            pl.len().alias("n_rows"),
            pl.col("patent_id").n_unique().alias("n_patents"),
            pl.col("focal_term").n_unique().alias("n_terms"),
        ])
        .collect()
    )
    
    logger.info(f"  Focal term rows:          {stats['n_rows'][0]:,}")
    logger.info(f"  Unique patents w/ focal:  {stats['n_patents'][0]:,}")
    logger.info(f"  Unique focal terms:       {stats['n_terms'][0]:,}")
    del stats
    gc.collect()

    # Write CSV version (stream from parquet)
    logger.info("Writing CSV version...")
    pl.scan_parquet(FOCAL_PATH).sink_csv(OUT_DIR / "focal_terms_full.csv")
    logger.info("✓ Saved: focal_terms_full.parquet, focal_terms_full.csv")

except Exception as e:
    logger.error(f"Task 1 failed: {e}")
    logger.error("Possible causes:")
    logger.error("  - Insufficient disk space for output files")
    logger.error("  - LINK or PMED file format mismatch")
    logger.error("  - OOM during join (try reducing POLARS_MAX_THREADS)")
    sys.exit(1)




# ═════════════════════════════════════════════════════════════════════════
# TASK 2: MEASURE OVERLAP INTENSITY
# ═════════════════════════════════════════════════════════════════════════
# Compute statistics on how many focal terms each patent has.
# Safe to collect because output is ~1 row per patent (much smaller than input).
# ═════════════════════════════════════════════════════════════════════════

logger.info("\n" + "="*70)
logger.info("TASK 2: Measuring Overlap Intensity")
logger.info("="*70)
t0 = time.time()

try:
    counts = (
        pl.scan_parquet(FOCAL_PATH)
        .group_by("patent_id")
        .agg(pl.col("focal_term").n_unique().alias("num_focal_terms"))
        .sort("patent_id")
        .collect()  # Safe: output size ≈ num_patents (usually < 1M rows)
    )
    total = len(counts)

    if total == 0:
        logger.warning("No patents with focal terms — skipping Task 2.")
    else:
        values = counts["num_focal_terms"].to_numpy()
        mean_val = float(values.mean())
        median_val = float(np.median(values))
        std_val = float(values.std())
        min_val = int(values.min())
        max_val = int(values.max())
        n_one = int((counts["num_focal_term"] == 1).sum())

        logger.info(f"  Patents with focal terms:        {total:,}")
        logger.info(f"  Mean focal terms per patent:     {mean_val:.2f}")
        logger.info(f"  Median:                          {median_val:.1f}")
        logger.info(f"  Std Dev:                         {std_val:.2f}")
        logger.info(f"  Min / Max:                       {min_val} / {max_val}")
        logger.info(f"  Patents with exactly 1 term:     {n_one:,} ({n_one/total*100:.1f}%)")
        logger.info(f"✓ Done in {_elapsed(t0)}")

        # Write count distribution
        counts.write_csv(OUT_DIR / "focal_term_counts_per_patent_full.csv")

        # Write summary statistics
        pl.DataFrame({
            "statistic": ["mean", "median", "std", "min", "max", "n_patents", "pct_exactly_1"],
            "value": [mean_val, median_val, std_val, float(min_val), float(max_val),
                      float(total), float(n_one / total * 100)],
        }).write_csv(OUT_DIR / "task2_summary_stats_full.csv")
        logger.info("✓ Saved: focal_term_counts_per_patent_full.csv, task2_summary_stats_full.csv")

        # Histogram
        logger.info("Generating histogram...")
        bin_range = range(1, min(max_val + 2, 202))
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(values, bins=list(bin_range), color="steelblue", edgecolor="white", align="left")
        ax1.axvline(mean_val, color="red", linestyle="--", label=f"Mean ({mean_val:.2f})")
        ax1.axvline(median_val, color="orange", linestyle="--", label=f"Median ({median_val:.0f})")
        ax1.set_title("Histogram of Focal Terms per Patent (Full Dataset)")
        ax1.set_xlabel("Number of Focal Terms")
        ax1.set_ylabel("Number of Patents")
        ax1.legend()
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "histogram_focal_terms_full.png", dpi=150)
        plt.close()

        # Density plot
        logger.info("Generating density plot...")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        if len(values) > 1:
            kde = gaussian_kde(values)
            x = np.linspace(values.min() - 0.5, values.max() + 0.5, 300)
            ax2.plot(x, kde(x), color="steelblue")
        ax2.axvline(mean_val, color="red", linestyle="--", label=f"Mean ({mean_val:.2f})")
        ax2.axvline(median_val, color="orange", linestyle="--", label=f"Median ({median_val:.0f})")
        ax2.set_title("Density Plot of Focal Terms per Patent (Full Dataset)")
        ax2.set_xlabel("Number of Focal Terms")
        ax2.set_ylabel("Density")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "density_focal_terms_full.png", dpi=150)
        plt.close()
        logger.info("✓ Saved: histogram_focal_terms_full.png, density_focal_terms_full.png")

    del counts, values
    gc.collect()

except Exception as e:
    logger.error(f"Task 2 failed: {e}")
    logger.error("Possible causes:")
    logger.error("  - focal_terms_full.parquet is corrupted or missing")
    logger.error("  - Matplotlib rendering error (check /tmp disk space)")
    sys.exit(1)




# ═════════════════════════════════════════════════════════════════════════
# TASK 3: SEMANTIC CONTEXT COMPARISON
# ═════════════════════════════════════════════════════════════════════════
# Compute cosine similarity between patent context and cited paper context
# for each focal term, using sentence-transformers embeddings.
#
# CRITICAL FOR LARGE DATASETS:
#   1. Context data is streamed to disk (focal_term_context_full.parquet)
#   2. Model encoding uses CHUNKED PROCESSING: only CHUNK_SIZE rows in RAM at once
#   3. ENCODE_BATCH controls how many sentences go to model.encode() per call
#   4. Aggressive garbage collection after each chunk
#
# MEMORY OPTIMIZATION CHECKLIST:
#   ✓ Lazy evaluation for all context joins
#   ✓ Chunked model encoding (default 128 rows)
#   ✓ gc.collect() after each chunk
#   ✓ Model on CPU (no GPU VRAM)
#   ✓ Explicit del + gc for intermediate arrays
#
# If 502 timeout occurs:
#   - Reduce CHUNK_SIZE from 128 → 64 → 32
#   - Reduce ENCODE_BATCH from 64 → 32 → 16
#   - Set POLARS_MAX_THREADS=2
#   - Check /tmp for disk space (model writes ~5GB temp files)
# ═════════════════════════════════════════════════════════════════════════

logger.info("\n" + "="*70)
logger.info("TASK 3: Semantic Context Comparison")
logger.info("="*70)

focal_lf = (
    pl.scan_parquet(FOCAL_PATH)
    .select(["patent_id", "focal_term"])
    .unique()
)

n_pairs = focal_lf.select(pl.len()).collect().item()
if n_pairs == 0:
    logger.warning("No focal terms — skipping Task 3.")
else:
    t0 = time.time()

    try:
        logger.info(f"Building context tables for {n_pairs:,} focal term instances...")

        # ─ Patent-side context: all terms in patent EXCEPT the focal term
        logger.info("  Step 1: Patent context (lazy join)...")
        patent_contexts_lf = (
            pl.scan_parquet(PAT_PATH)
            .select(["patent_id", "term"])
            .join(focal_lf, on="patent_id", how="inner")
            .filter(pl.col("term") != pl.col("focal_term"))
            .group_by(["patent_id", "focal_term"])
            .agg(pl.col("term").alias("patent_context"))
        )

        # ─ Paper-side context: all terms in cited papers EXCEPT the focal term
        logger.info("  Step 2: Paper context (lazy join)...")
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

        # ─ Sink combined context to disk (streaming, no RAM spike)
        context_sink_path = OUT_DIR / "focal_term_context_full.parquet"
        logger.info("  Step 3: Sinking combined context to parquet...")
        (
            focal_lf
            .join(patent_contexts_lf, on=["patent_id", "focal_term"], how="left")
            .join(paper_contexts_lf, on=["patent_id", "focal_term"], how="left")
            .sink_parquet(context_sink_path)
        )
        logger.info(f"✓ Context parquet written [{_elapsed(t0)}]")

        # Write CSV version
        logger.info("  Step 4: Writing CSV version...")
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
        logger.info("✓ Saved: focal_term_context_full.parquet, focal_term_context_full.csv")

        # ─ Load model (once, before chunking)
        logger.info("\nLoading sentence-transformers model (CPU)...")
        logger.info(f"  Using device: CPU")
        logger.info(f"  Model: all-MiniLM-L6-v2 (22M params)")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        logger.info("✓ Model loaded")

        # ─ Chunked encoding: process in small batches to keep memory low
        total_rows = pl.scan_parquet(context_sink_path).select(pl.len()).collect().item()
        logger.info(f"\nEncoding {total_rows:,} rows...")
        logger.info(f"  CHUNK_SIZE:   {CHUNK_SIZE} rows/iteration")
        logger.info(f"  ENCODE_BATCH: {ENCODE_BATCH} sentences/model.encode() call")
        t0 = time.time()

        all_patent_ids = []
        all_focal_terms = []
        all_sims = []

        for chunk_idx, start in enumerate(range(0, total_rows, CHUNK_SIZE)):
            # Load chunk from parquet
            chunk = (
                pl.scan_parquet(context_sink_path)
                .slice(start, CHUNK_SIZE)
                .collect()
            )

            # Build sentences: focal_term + context words
            pat_sents = [
                f"{r['focal_term']} {' '.join(r['patent_context'] or [])}"
                for r in chunk.iter_rows(named=True)
            ]
            pap_sents = [
                f"{r['focal_term']} {' '.join(r['paper_context'] or [])}"
                for r in chunk.iter_rows(named=True)
            ]

            # Encode both sides
            pat_emb = model.encode(
                pat_sents, show_progress_bar=False,
                batch_size=ENCODE_BATCH, normalize_embeddings=True
            )
            pap_emb = model.encode(
                pap_sents, show_progress_bar=False,
                batch_size=ENCODE_BATCH, normalize_embeddings=True
            )

            # Compute cosine similarity (normalized embeddings → dot product = cosine)
            sims_chunk = (pat_emb * pap_emb).sum(axis=1).astype(np.float32)

            # Accumulate results
            all_patent_ids.extend(chunk["patent_id"].to_list())
            all_focal_terms.extend(chunk["focal_term"].to_list())
            all_sims.append(sims_chunk)

            # Aggressive cleanup before next iteration
            del chunk, pat_sents, pap_sents, pat_emb, pap_emb, sims_chunk
            gc.collect()

            done = min(start + CHUNK_SIZE, total_rows)
            elapsed = _elapsed(t0)
            pct = 100 * done / total_rows
            logger.info(f"  {done:7,} / {total_rows:,}  ({pct:5.1f}%)  [{elapsed}]")

        logger.info("✓ Encoding complete")

        # Combine all similarity scores
        sims_all = np.concatenate(all_sims)
        cosine_df = pl.DataFrame({
            "patent_id": all_patent_ids,
            "focal_term": all_focal_terms,
            "cosine_similarity": sims_all.tolist(),
        })
        del all_patent_ids, all_focal_terms, all_sims, sims_all
        gc.collect()

        # Write results
        cosine_df.write_parquet(OUT_DIR / "cosine_similarity_results_full.parquet")
        cosine_df.write_csv(OUT_DIR / "cosine_similarity_results_full.csv")
        logger.info("✓ Saved: cosine_similarity_results_full.parquet, .csv")

        # Summary statistics
        sim = cosine_df["cosine_similarity"].drop_nulls().to_numpy()
        del cosine_df
        gc.collect()

        if len(sim) == 0:
            logger.warning("No cosine similarity values — skipping summary and plot.")
            pl.DataFrame({
                "statistic": ["mean", "median", "std", "min", "max"],
                "value": [None, None, None, None, None],
            }).write_csv(OUT_DIR / "task3_cosine_summary_stats_full.csv")
        else:
            mean_sim = float(sim.mean())
            median_sim = float(np.median(sim))
            std_sim = float(sim.std())
            min_sim = float(sim.min())
            max_sim = float(sim.max())

            logger.info("\n=== Cosine Similarity Summary ===")
            logger.info(f"  Mean   : {mean_sim:.3f}")
            logger.info(f"  Median : {median_sim:.3f}")
            logger.info(f"  Std    : {std_sim:.3f}")
            logger.info(f"  Min    : {min_sim:.3f}")
            logger.info(f"  Max    : {max_sim:.3f}")

            pl.DataFrame({
                "statistic": ["mean", "median", "std", "min", "max"],
                "value": [mean_sim, median_sim, std_sim, min_sim, max_sim],
            }).write_csv(OUT_DIR / "task3_cosine_summary_stats_full.csv")
            logger.info("✓ Saved: task3_cosine_summary_stats_full.csv")

            # Histogram
            logger.info("Generating histogram...")
            plt.figure(figsize=(8, 4))
            plt.hist(sim, bins=50, color="steelblue", edgecolor="black")
            plt.axvline(mean_sim, color="red", linestyle="--", label=f"Mean ({mean_sim:.3f})")
            plt.axvline(median_sim, color="orange", linestyle="--", label=f"Median ({median_sim:.3f})")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Count")
            plt.title("Distribution of Semantic Similarity\n"
                      "(Full Dataset — patent vs. paper context per focal term)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(VIZ_DIR / "cosine_similarity_distribution_full.png", dpi=150)
            plt.close()
            logger.info("✓ Saved: cosine_similarity_distribution_full.png")

    except Exception as e:
        logger.error(f"Task 3 failed: {e}")
        logger.error("Possible causes:")
        logger.error("  - Out of memory during model encoding (reduce CHUNK_SIZE or ENCODE_BATCH)")
        logger.error("  - Timeout during context join (reduce POLARS_MAX_THREADS)")
        logger.error("  - Model download failed (check internet connection)")
        logger.error("  - /tmp full (check disk space for temp model files)")
        sys.exit(1)

logger.info("\n" + "="*70)
logger.info("✓✓✓ PIPELINE COMPLETE ✓✓✓")
logger.info("="*70)