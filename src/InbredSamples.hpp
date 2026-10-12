#ifndef PCAONE_INBREDSAMPLES_
#define PCAONE_INBREDSAMPLES_

#include "Cmd.hpp"
#include "Data.hpp"

// --inbreed 2: the inbreeding coefficient of each sample accounting for
// population structure, i.e. with the individual allele frequencies pi of a
// reference PCA (-P/--USV) in place of one frequency per site. See
// InbredSamples.cpp for the estimator.
//
// Pi is the reference (a FileUSV). Writes <out>.inbred, one row per sample.
void run_inbred_samples(Data* Pi, const Param& params);

// --inbreed 2 within --robust aarobust-kin: write <out>.inbred from the sums of
// observed and expected heterozygotes (complete genotypes, nsites per sample)
void write_inbred_sums(const Param& params, const Mat1D& obs, const Mat1D& exp, uint64 nsites);

#endif  // PCAONE_INBREDSAMPLES_
