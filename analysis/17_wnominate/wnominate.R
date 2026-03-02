#!/usr/bin/env Rscript
# W-NOMINATE and Optimal Classification for Kansas Legislature
#
# Interface: Rscript wnominate.R <input_csv> <output_dir> <chamber> <polarity_idx> <dims>
#
# Input CSV: pscl rollcall format (1=Yea, 6=Nay, 9=Missing).
#   First column = legislator_slug, remaining = vote columns.
#
# Outputs:
#   wnominate_coords_{chamber}.csv   — legislator coordinates + SEs
#   oc_coords_{chamber}.csv          — OC coordinates + correct classification
#   fit_statistics_{chamber}.json    — CC, APRE, GMP per method
#   eigenvalues_{chamber}.csv        — W-NOMINATE eigenvalues for scree plot

# ── Check packages ────────────────────────────────────────────────────────────

required <- c("wnominate", "oc", "pscl", "jsonlite")
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {
  cat("Missing R packages:", paste(missing, collapse = ", "), "\n")
  cat("Install with:\n")
  cat(sprintf("  install.packages(c(%s))\n",
              paste(sprintf('"%s"', missing), collapse = ", ")))
  quit(status = 1)
}

library(pscl)
library(wnominate)
library(jsonlite)

# ── Parse arguments ───────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  cat("Usage: Rscript wnominate.R <input_csv> <output_dir> <chamber> <polarity_idx> <dims>\n")
  quit(status = 1)
}

input_csv    <- args[1]
output_dir   <- args[2]
chamber      <- args[3]
polarity_idx <- as.integer(args[4])
dims         <- as.integer(args[5])

cat(sprintf("W-NOMINATE + OC: chamber=%s, polarity=%d, dims=%d\n", chamber, polarity_idx, dims))

# ── Read vote matrix ──────────────────────────────────────────────────────────

raw <- read.csv(input_csv, row.names = 1, check.names = FALSE)
cat(sprintf("  Vote matrix: %d legislators x %d votes\n", nrow(raw), ncol(raw)))

vote_data <- as.matrix(raw)

# Build pscl rollcall object
rc <- rollcall(vote_data, yea = 1, nay = 6, missing = 9, notInLegis = NA,
               legis.names = rownames(raw),
               vote.names = colnames(raw))

# ── W-NOMINATE ────────────────────────────────────────────────────────────────

cat("  Running W-NOMINATE...\n")
wn <- tryCatch({
  wnominate(rc, dims = dims, polarity = rep(polarity_idx, dims),
            minvotes = 20, lop = 0.025, trials = 3, verbose = FALSE)
}, error = function(e) {
  cat(sprintf("  W-NOMINATE failed: %s\n", e$message))
  NULL
})

if (is.null(wn)) {
  cat("  W-NOMINATE failed — exiting\n")
  quit(status = 2)
}

cat("  W-NOMINATE completed successfully\n")

# Extract coordinates
wn_coords <- data.frame(
  coord1D = wn$legislators$coord1D,
  coord2D = if (dims >= 2) wn$legislators$coord2D else NA,
  se1     = wn$legislators$se1,
  se2     = if (dims >= 2) wn$legislators$se2 else NA
)
rownames(wn_coords) <- rownames(raw)

write.csv(wn_coords,
          file.path(output_dir, sprintf("wnominate_coords_%s.csv", chamber)),
          row.names = TRUE)

# Extract eigenvalues
if (!is.null(wn$eigenvalues)) {
  eigen_df <- data.frame(
    dimension  = seq_along(wn$eigenvalues),
    eigenvalue = wn$eigenvalues
  )
  write.csv(eigen_df,
            file.path(output_dir, sprintf("eigenvalues_%s.csv", chamber)),
            row.names = FALSE)
}

# Fit statistics
fit <- list()
fit$wnominate <- list(
  correctClassification = wn$fits[1],
  APRE = wn$fits[2],
  GMP  = wn$fits[3]
)

# ── Optimal Classification ────────────────────────────────────────────────────

cat("  Running Optimal Classification...\n")
oc_result <- tryCatch({
  library(oc)
  oc(rc, dims = dims, polarity = rep(polarity_idx, dims),
     minvotes = 20, lop = 0.025, verbose = FALSE)
}, error = function(e) {
  cat(sprintf("  OC failed (non-fatal): %s\n", e$message))
  NULL
})

if (!is.null(oc_result)) {
  cat("  OC completed successfully\n")

  oc_leg <- oc_result$legislators
  oc_coords <- data.frame(
    coord1D = oc_leg[, "coord1D"],
    coord2D = if (dims >= 2 && "coord2D" %in% colnames(oc_leg)) oc_leg[, "coord2D"] else NA,
    correctClassification = if ("correctYea.1D" %in% colnames(oc_leg)) oc_leg[, "correctYea.1D"] else NA
  )

  # Use 2D classification if available
  if (dims >= 2 && "correctYea.2D" %in% colnames(oc_leg)) {
    oc_coords$correctClassification <- oc_leg[, "correctYea.2D"]
  }

  rownames(oc_coords) <- rownames(raw)

  write.csv(oc_coords,
            file.path(output_dir, sprintf("oc_coords_%s.csv", chamber)),
            row.names = TRUE)

  fit$oc <- list(
    correctClassification = oc_result$fits[1],
    APRE = oc_result$fits[2],
    GMP  = oc_result$fits[3]
  )
} else {
  cat("  OC failed — continuing with W-NOMINATE only\n")
}

# ── Save fit statistics ───────────────────────────────────────────────────────

write_json(fit,
           file.path(output_dir, sprintf("fit_statistics_%s.json", chamber)),
           pretty = TRUE, auto_unbox = TRUE)

cat("  All outputs written to:", output_dir, "\n")
