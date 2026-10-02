#!/usr/bin/env Rscript
# evalAdmix (van Waaij projection estimator) from PCAone output.
#
#   PCAone -b <plink> -k <K-1> -d 0 -D --maf 0.05 -o <out>
#   Rscript evalAdmix_pcaone.R <out> <plink> [nPC]
#
# Writes <out>.corres.txt (correlation of residuals) and <out>.kinship.txt (= corres/2).
# Validated against popgenDK/evalPopStructure evalPCA(): r = 1.00000, max diff 5.9e-08.

args <- commandArgs(TRUE)
if (length(args) < 2) stop("usage: evalAdmix_pcaone.R <pcaone_out_prefix> <plink_prefix> [nPC]")
pre <- args[1]; plink <- args[2]

## ---- PCAone residuals: uint32 M, uint32 N, then M*N float32, SNP-major ----
con <- file(paste0(pre, ".residuals"), "rb")
h <- readBin(con, "integer", n = 2, size = 4); M <- h[1]; N <- h[2]
R <- matrix(readBin(con, "numeric", n = as.numeric(M) * N, size = 4), nrow = N)
close(con); R <- t(R)                                   # M x N
cat("residuals:", M, "SNPs x", N, "individuals\n")

## ---- PC scores ----
V <- as.matrix(read.table(paste0(pre, ".eigvecs")))
k <- if (length(args) >= 3) as.integer(args[3]) else 1  # default K-1 = 1
S <- cbind(V[, seq_len(k), drop = FALSE], 1)            # k PCs + intercept  ->  dimension K
P <- S %*% MASS::ginv(crossprod(S)) %*% t(S)
I <- diag(N)
cat("projection: ", k, "PC(s) + intercept =", k + 1, "dimensions\n")

## ---- D_hat: per-individual mean heterozygosity, from the .bed ----
fam <- read.table(paste0(plink, ".fam")); bim <- read.table(paste0(plink, ".bim"))
n <- nrow(fam); m <- nrow(bim)
bed <- readBin(paste0(plink, ".bed"), "raw", n = 3 + ceiling(n/4) * m)
raw <- matrix(bed[-(1:3)], nrow = ceiling(n/4)); lut <- c(2L, NA_integer_, 1L, 0L)
G <- matrix(NA_integer_, n, m)
bs <- function(v) { x <- as.integer(v); cbind(x%%4, (x%/%4)%%4, (x%/%16)%%4, (x%/%64)%%4) }
for (j in seq_len(m)) { d <- bs(raw[, j]); G[, j] <- lut[as.vector(t(d)) + 1L][1:n] }
p <- colMeans(G, na.rm = TRUE)/2
kp <- p > 0.05 & p < 0.95                               # match PCAone's --maf
dhat <- rowMeans(G[, kp] * (2 - G[, kp]), na.rm = TRUE)

## ---- the estimator ----
bhat   <- cor(R)                                        # empirical
chat   <- cov2cor((I - P) %*% diag(dhat) %*% (I - P))   # model-implied
corres <- bhat - chat
corres <- pmin(pmax(corres, -1), 1)                     # a correlation cannot leave [-1,1]
kin    <- corres/2                                      # kinship scale
dimnames(corres) <- dimnames(kin) <- list(fam$V2, fam$V2)

write.table(round(corres, 6), paste0(pre, ".corres.txt"), quote = FALSE, sep = "\t")
write.table(round(kin, 6),    paste0(pre, ".kinship.txt"), quote = FALSE, sep = "\t")
cat("wrote", paste0(pre, ".corres.txt"), "and", paste0(pre, ".kinship.txt"), "\n")
rel <- which(kin > 0.04 & upper.tri(kin), arr.ind = TRUE)
if (nrow(rel)) { cat("\npairs with kinship > 0.04:\n")
  o <- order(-kin[rel]); for (i in head(o, 20))
    cat(sprintf("  %-10s %-10s  phi = %.4f\n", fam$V2[rel[i,1]], fam$V2[rel[i,2]], kin[rel[i,1], rel[i,2]])) }
