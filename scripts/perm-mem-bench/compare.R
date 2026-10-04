#!/usr/bin/env -S Rscript --vanilla
# compare.R ref_prefix run_prefix [k]
# Accuracy of a PCAone run against a reference run:
#  - max relative eigenvalue error over the top k
#  - |cor| of each PC with the matching reference PC
#  - subspace overlap ||Uref' U||_F^2 / k  (1 = same subspace; robust to PC order swaps)
args <- commandArgs(TRUE)
k <- if (length(args) >= 3) as.integer(args[3]) else 10
rd <- function(p) as.matrix(read.table(paste0(p, ".eigvecs")))[, 1:k, drop = FALSE]
ev <- function(p) scan(paste0(p, ".eigvals"), quiet = TRUE)[1:k]
U0 <- rd(args[1]); U1 <- rd(args[2])
U0 <- qr.Q(qr(scale(U0, scale = FALSE))); U1 <- qr.Q(qr(scale(U1, scale = FALSE)))
e0 <- ev(args[1]); e1 <- ev(args[2])
cors <- abs(sapply(1:k, function(i) cor(U0[, i], U1[, i])))
ov <- sum(crossprod(U0, U1)^2) / k
cat(sprintf("%s\tmaxrelerr_eig=%.2e\tminPCcor=%.5f\tsubspace=%.6f\tPCcor=%s\n", basename(args[2]),
            max(abs(e1 - e0) / e0), min(cors), ov, paste(sprintf("%.4f", cors), collapse = ",")))
