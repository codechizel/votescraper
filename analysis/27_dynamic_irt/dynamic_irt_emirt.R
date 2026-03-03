#!/usr/bin/env Rscript
# Dynamic IRT exploration via emIRT::dynIRT
#
# Usage: Rscript dynamic_irt_emirt.R <input_csv> <output_csv> <chamber>
#
# Input CSV columns: legislator_id, bill_id, vote, time_period
# Output CSV columns: legislator_id, time_period, xi_mean
#
# Requires: emIRT package (install.packages("emIRT"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript dynamic_irt_emirt.R <input_csv> <output_csv> <chamber>")
}

input_csv <- args[1]
output_csv <- args[2]
chamber <- args[3]

cat(sprintf("emIRT::dynIRT — %s\n", chamber))
cat(sprintf("  Input:  %s\n", input_csv))
cat(sprintf("  Output: %s\n", output_csv))

# Check emIRT availability
if (!requireNamespace("emIRT", quietly = TRUE)) {
    stop("emIRT package not installed. Run: install.packages('emIRT')")
}
library(emIRT)

# Read data
cat("  Loading data...\n")
data <- read.csv(input_csv)
cat(sprintf("  %d observations\n", nrow(data)))

# Build vote matrix (legislators x bills, +1/-1 for yea/nay, 0 for missing)
legislators <- sort(unique(data$legislator_id))
bills <- sort(unique(data$bill_id))
n_leg <- length(legislators)
n_bills <- length(bills)
n_time <- max(data$time_period) + 1  # 0-indexed to 1-indexed

cat(sprintf("  %d legislators, %d bills, %d periods\n", n_leg, n_bills, n_time))

# Build sparse roll call data
leg_map <- setNames(seq_along(legislators), legislators)
bill_map <- setNames(seq_along(bills), bills)

vote_matrix <- matrix(0, nrow = n_leg, ncol = n_bills)
for (i in seq_len(nrow(data))) {
    row <- leg_map[as.character(data$legislator_id[i])]
    col <- bill_map[as.character(data$bill_id[i])]
    # emIRT uses +1 for yea, -1 for nay, 0 for missing
    vote_matrix[row, col] <- ifelse(data$vote[i] == 1, 1, -1)
}

# Build time period assignments for legislators
# Each legislator's periods as a vector
leg_time <- rep(1, n_leg)  # placeholder — dynIRT assigns legislator-to-time mapping

# Build startlegis and endlegis (first and last bill index per time period)
bill_periods <- data$time_period[!duplicated(data$bill_id)]
startlegis <- integer(n_time)
endlegis <- integer(n_time)

for (t in 0:(n_time - 1)) {
    period_bills <- which(bill_periods == t)
    if (length(period_bills) > 0) {
        startlegis[t + 1] <- min(period_bills)
        endlegis[t + 1] <- max(period_bills)
    }
}

# Run dynIRT
cat("  Running dynIRT...\n")
cat("  omega2 = 0.1 (Martin-Quinn default)\n")

tryCatch({
    result <- dynIRT(
        .data = list(
            rc = vote_matrix,
            startlegis = startlegis,
            endlegis = endlegis
        ),
        .starts = list(
            alpha = rep(0, n_bills),
            beta = rep(1, n_bills),
            x = matrix(0, nrow = n_leg, ncol = n_time)
        ),
        .priors = list(
            omega2 = 0.1
        ),
        .control = list(
            threads = 6,
            verbose = TRUE,
            thresh = 1e-6
        )
    )

    # Extract ideal points
    cat("  Extracting results...\n")
    xi_estimates <- result$means$x  # n_leg x n_time matrix

    # Build output DataFrame
    output_rows <- data.frame(
        legislator_id = integer(),
        time_period = integer(),
        xi_mean = numeric()
    )

    for (i in seq_len(n_leg)) {
        for (t in seq_len(n_time)) {
            output_rows <- rbind(output_rows, data.frame(
                legislator_id = legislators[i],
                time_period = t - 1,  # back to 0-indexed
                xi_mean = xi_estimates[i, t]
            ))
        }
    }

    write.csv(output_rows, output_csv, row.names = FALSE)
    cat(sprintf("  Results written to %s\n", output_csv))
    cat(sprintf("  %d rows\n", nrow(output_rows)))

}, error = function(e) {
    cat(sprintf("  ERROR: %s\n", e$message))
    quit(status = 1)
})
