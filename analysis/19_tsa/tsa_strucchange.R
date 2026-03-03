#!/usr/bin/env Rscript
# TSA R Enrichment: CROPS penalty selection + Bai-Perron confidence intervals
#
# Usage:
#   Rscript tsa_strucchange.R <input_csv> <output_dir> <party> <min_pen> <max_pen> <max_breaks>
#
# Input CSV: single column `mean_rice` (weekly Rice values, chronological)
# Outputs:
#   - crops_{party}.json  — CROPS penalty/changepoint pairs
#   - bai_perron_{party}.json — breakpoints + 95% CIs
#
# Required packages: changepoint, strucchange, jsonlite

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  cat("Usage: Rscript tsa_strucchange.R <input_csv> <output_dir> <party> <min_pen> <max_pen> <max_breaks>\n")
  quit(status = 1)
}

input_csv  <- args[1]
output_dir <- args[2]
party      <- tolower(args[3])
min_pen    <- as.numeric(args[4])
max_pen    <- as.numeric(args[5])
max_breaks <- as.integer(args[6])

# Read signal
signal <- read.csv(input_csv)$mean_rice
n <- length(signal)
cat(sprintf("  [R] Signal length: %d observations\n", n))

if (n < 10) {
  cat("  [R] Signal too short for analysis\n")
  # Write error JSON for both
  writeLines(
    jsonlite::toJSON(list(error = "Signal too short"), auto_unbox = TRUE),
    file.path(output_dir, paste0("crops_", party, ".json"))
  )
  writeLines(
    jsonlite::toJSON(list(error = "Signal too short"), auto_unbox = TRUE),
    file.path(output_dir, paste0("bai_perron_", party, ".json"))
  )
  quit(status = 0)
}

# ── CROPS ────────────────────────────────────────────────────────────────────

tryCatch({
  library(changepoint)

  cat(sprintf("  [R] Running CROPS with penalty range [%.1f, %.1f]\n", min_pen, max_pen))

  crops_result <- cpt.mean(
    signal,
    penalty = "CROPS",
    pen.value = c(min_pen, max_pen),
    method = "PELT",
    minseglen = 5
  )

  # Extract the solution path
  # pen.value.full gives the penalty/n_changepoints pairs
  pen_vals <- pen.value.full(crops_result)

  # pen_vals is a matrix with columns: penalty, n_changepoints
  if (is.matrix(pen_vals) && nrow(pen_vals) > 0) {
    crops_out <- list(
      penalties = pen_vals[, 1],
      n_changepoints = as.integer(pen_vals[, 2])
    )
    cat(sprintf("  [R] CROPS found %d distinct segmentations\n", nrow(pen_vals)))
  } else {
    crops_out <- list(error = "CROPS returned empty result")
  }

  writeLines(
    jsonlite::toJSON(crops_out, auto_unbox = TRUE),
    file.path(output_dir, paste0("crops_", party, ".json"))
  )

}, error = function(e) {
  cat(sprintf("  [R] CROPS error: %s\n", e$message))
  writeLines(
    jsonlite::toJSON(list(error = e$message), auto_unbox = TRUE),
    file.path(output_dir, paste0("crops_", party, ".json"))
  )
})

# ── Bai-Perron ───────────────────────────────────────────────────────────────

tryCatch({
  library(strucchange)

  # Compute safe max_breaks: strucchange requires (max_breaks + 1) * h <= n
  # where h is the minimum segment length (default: h = 0.15 * n)
  safe_max <- floor(n / ceiling(0.15 * n)) - 1
  if (safe_max < 1) safe_max <- 1
  bp_max <- min(max_breaks, safe_max)

  cat(sprintf("  [R] Running Bai-Perron with max_breaks=%d (safe limit=%d)\n", bp_max, safe_max))

  bp <- breakpoints(signal ~ 1, breaks = bp_max)

  if (is.na(bp$breakpoints[1])) {
    cat("  [R] Bai-Perron found no breakpoints\n")
    bp_out <- list(breakpoints = list(), ci_lower = list(), ci_upper = list())
  } else {
    n_breaks <- length(bp$breakpoints)
    cat(sprintf("  [R] Bai-Perron found %d breakpoints at: %s\n",
                n_breaks, paste(bp$breakpoints, collapse = ", ")))

    # Confidence intervals
    ci <- confint(bp, level = 0.95)
    ci_mat <- ci$confint  # matrix: n_breaks x 3 (lower, breakpoint, upper)

    bp_out <- list(
      breakpoints = as.integer(ci_mat[, 2]),
      ci_lower = as.integer(ci_mat[, 1]),
      ci_upper = as.integer(ci_mat[, 3])
    )
  }

  writeLines(
    jsonlite::toJSON(bp_out, auto_unbox = TRUE),
    file.path(output_dir, paste0("bai_perron_", party, ".json"))
  )

}, error = function(e) {
  cat(sprintf("  [R] Bai-Perron error: %s\n", e$message))
  writeLines(
    jsonlite::toJSON(list(error = e$message), auto_unbox = TRUE),
    file.path(output_dir, paste0("bai_perron_", party, ".json"))
  )
})

cat("  [R] TSA enrichment complete\n")
