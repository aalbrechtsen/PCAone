#ifndef PCAONE_ROBUST_
#define PCAONE_ROBUST_

#include "Data.hpp"

/**
 * --robust: PCA that is not driven by close relatives, for small to moderate N
 * (everything after one streaming pass works on N x N matrices).
 *
 * The pass accumulates, on the 0/1/2 genotype scale g,
 *   A   = G G'              raw Gram
 *   D_i = mean_s g(2 - g)   per-individual heterozygosity
 *   KING-robust counts      het-het, IBS0, number of hets
 *   C   = X X' / M          standardised GRM (only for aarobust-kin)
 * The Chen & Storey matrix is H = A/M - diag(D) (low rank including the
 * diagonal under HWE; its first eigenvector is the mean).
 *
 * Modes (K-1 = -k structure PCs; tau = --kin-min; screen = --king-screen):
 *   detect-white    fixed rank + kinship threshold on H with a KING screen and
 *                   the diagonal free (imputed from the fit: robust to
 *                   inbreeding / genotype errors) detects the related pairs and
 *                   their structure-adjusted kinship phi_ij = R_ij / (2 sqrt(D_i
 *                   D_j)), R = H - L; then whitening of the raw Gram with
 *                   Sigma = v^1/2 (I + 2 Psi) v^1/2, v = diag(A/M) - diag(L)
 *   cswhite         CS whitening with KING kinship (or --kinship <file>) and
 *                   v = D; assumes HWE within individuals
 *   frkin           the free-diagonal fit alone (eigenvectors 2..K of L)
 *   aarobust-kin    PCP (inexact ALM) of the GRM with the diagonal unobserved
 *                   and a kinship threshold in place of lambda, KING screen
 */
void run_robust(Data* data, const Param& params);

#endif  // PCAONE_ROBUST_
