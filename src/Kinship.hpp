#ifndef PCAONE_KINSHIP_
#define PCAONE_KINSHIP_

#include <tuple>

#include "Cmd.hpp"
#include "Common.hpp"

/**
 * Kinship-whitened PCA (--kinship): PCA that is not driven by close relatives.
 *
 * With standardised genotypes X (N x M), the covariance between two samples at
 * a site is 2*phi_ij, so recent relatedness adds Sigma = I + 2*Psi to the sample
 * covariance, where Psi holds the kinship of close relatives and is zero
 * elsewhere. Each family adds an eigenvalue of about 1 + 2*phi*(n-1) that does
 * not shrink with N, so families surface as their own PCs.
 *
 * Under the model x_j = W z_j + e_j, e_j ~ N(0, s^2 Sigma), the ML fit of W is
 * the PCA of the whitened data L X with L = Sigma^{-1/2}. The ancestry scores
 * are then Sigma^{1/2} U~, which equals projecting the *unwhitened* genotypes
 * onto the loadings learnt from the whitened data, X V~ S~^{-1}. Unrelated
 * samples have Sigma = I and are untouched.
 *
 * Psi is sparse: only pairs with kinship >= --kin-min enter, so Sigma is block
 * diagonal over connected components ("families") and each block is
 * eigendecomposed on its own. Applying L to an N x b block costs
 * O(sum_f n_f^2 * b), negligible next to the SVD.
 */
/// IID of every sample, from the .fam / .psam (empty for other inputs)
String1D read_sample_ids(const Param& params);

/// pairs (i, j, phi), i < j, with kinship >= kmin from --kinship (pair table or N x N matrix)
std::vector<std::tuple<int, int, double>> read_kinship_pairs(const Param& params, uint N, double kmin);

class KinshipWhitener {
 public:
  KinshipWhitener(const Param& params, uint nsamples);

  /// rows of X (N x b) <- Sigma^{-1/2} rows of X
  void whiten(Mat2D& X) const { apply(X, Linv_half); }
  /// rows of X (N x b) <- Sigma^{1/2} rows of X; maps whitened scores back
  void unwhiten(Mat2D& X) const { apply(X, L_half); }

  bool empty() const { return families.empty(); }

  /// the pairs with kinship >= --kin-min, as (i, j, phi), i < j
  const std::vector<std::tuple<int, int, double>>& close_pairs() const { return pairs_; }

 private:
  std::vector<std::tuple<int, int, double>> pairs_;
  std::vector<Int1D> families;    // sample indices of each family, ascending
  std::vector<Mat2D> L_half;     // Sigma_f^{1/2}
  std::vector<Mat2D> Linv_half;  // Sigma_f^{-1/2}

  void apply(Mat2D& X, const std::vector<Mat2D>& B) const;
};

#endif  // PCAONE_KINSHIP_
