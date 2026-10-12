/* ML IBD sharing (k0, k1, k2) of a pair from its genotypes and individual
 * allele frequencies: the likelihood of PCAone's final relatedness
 * (src/Robust.cpp final_relatedness), maximised on the simplex. The
 * log-likelihood is concave: an interior stationary point (damped Newton in
 * k0, k2) is the maximum; otherwise the maximum is on an edge (1-D concave,
 * safeguarded Newton). Called from pcp_pi.py through ctypes.
 * build: gcc -O3 -shared -fPIC -o ibd_fit.so ibd_fit.c -lm */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static double hwe(int g, double p) {
  double q = 1 - p;
  return g == 0 ? q * q : (g == 1 ? 2 * p * q : p * p);
}
static double other(int g, int shared, double p) {
  int d = g - shared;
  return d == 0 ? 1 - p : (d == 1 ? p : 0.0);
}

static double loglik(const double* L, long M, const double* k) {
  double ll = 0;
  for (long s = 0; s < M; s++) {
    const double* l = L + 3 * s;
    double t = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
    if (!(t > 0)) {
      if (l[0] + l[1] + l[2] > 0) return -INFINITY;
      continue;
    }
    ll += log(t);
  }
  return ll;
}

/* max over k = t e_a + (1 - t) e_b */
static double edge(const double* L, long M, int a, int b, double* k) {
  double lo = 0, hi = 1, t;
  double g0 = 0, g1 = 0;
  for (long s = 0; s < M; s++) {
    double d = L[3 * s + a] - L[3 * s + b];
    double t0 = L[3 * s + b], t1 = L[3 * s + a];
    if (t0 > 0) g0 += d / t0; else if (d > 0) g0 = INFINITY;
    if (t1 > 0) g1 += d / t1; else if (d < 0) g1 = -INFINITY;
  }
  if (g0 <= 0) t = 0;
  else if (g1 >= 0) t = 1;
  else {
    t = 0.5;
    for (int it = 0; it < 100; it++) {
      double g = 0, h = 0;
      for (long s = 0; s < M; s++) {
        double d = L[3 * s + a] - L[3 * s + b];
        double tt = L[3 * s + b] + t * d;
        if (tt > 0) {
          double r = d / tt;
          g += r;
          h -= r * r;
        }
      }
      if (g > 0) lo = t; else hi = t;
      double tn = h < 0 ? t - g / h : 0.5 * (lo + hi);
      if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);
      if (fabs(tn - t) < 1e-12 || hi - lo < 1e-12) { t = tn; break; }
      t = tn;
    }
  }
  k[0] = k[1] = k[2] = 0;
  k[a] = t;
  k[b] = 1 - t;
  return loglik(L, M, k);
}

/* ga, gb: genotypes 0..2; pa, pb: allele frequencies; k: output */
void ibd_fit(const int8_t* ga, const int8_t* gb, const double* pa_, const double* pb_, long M, double* kout) {
  double* L = malloc(sizeof(double) * 3 * M);
  for (long s = 0; s < M; s++) {
    double pa = fmin(fmax(pa_[s], 1e-3), 1 - 1e-3), pb = fmin(fmax(pb_[s], 1e-3), 1 - 1e-3);
    int A = ga[s], B = gb[s];
    double ps = 0.5 * (pa + pb), qs = 1 - ps;
    L[3 * s] = hwe(A, pa) * hwe(B, pb);
    L[3 * s + 1] = qs * other(A, 0, pa) * other(B, 0, pb) + ps * other(A, 1, pa) * other(B, 1, pb);
    L[3 * s + 2] = A == B ? hwe(A, ps) : 0.0;
  }
  /* interior: x = (k0, k2), k1 = 1 - k0 - k2 */
  double x[2] = {1.0 / 3, 1.0 / 3}, k[3] = {x[0], 1 - x[0] - x[1], x[1]};
  double ll = loglik(L, M, k);
  int inside = 1;
  for (int it = 0; it < 200; it++) {
    double g[2] = {0, 0}, H[3] = {0, 0, 0}; /* H00 H01 H11 (of -loglik) */
    for (long s = 0; s < M; s++) {
      const double* l = L + 3 * s;
      double t = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
      if (!(t > 0)) continue;
      double w0 = (l[0] - l[1]) / t, w1 = (l[2] - l[1]) / t;
      g[0] += w0;
      g[1] += w1;
      H[0] += w0 * w0;
      H[1] += w0 * w1;
      H[2] += w1 * w1;
    }
    double det = H[0] * H[2] - H[1] * H[1];
    if (!(det > 1e-300)) { inside = 0; break; }
    double st[2] = {(H[2] * g[0] - H[1] * g[1]) / det, (H[0] * g[1] - H[1] * g[0]) / det};
    double s = 1, xn[2], kn[3], lln = -INFINITY;
    int ok = 0;
    for (; s > 1e-12; s *= 0.5) {
      xn[0] = x[0] + s * st[0];
      xn[1] = x[1] + s * st[1];
      kn[0] = xn[0]; kn[1] = 1 - xn[0] - xn[1]; kn[2] = xn[1];
      if (kn[0] < 0 || kn[1] < 0 || kn[2] < 0) continue;
      lln = loglik(L, M, kn);
      if (lln >= ll) { ok = 1; break; }
    }
    if (!ok) { inside = 0; break; }
    double dx = fabs(xn[0] - x[0]) + fabs(xn[1] - x[1]);
    x[0] = xn[0]; x[1] = xn[1];
    k[0] = kn[0]; k[1] = kn[1]; k[2] = kn[2];
    ll = lln;
    if (fmin(k[0], fmin(k[1], k[2])) < 1e-9) { inside = 0; break; }
    if (dx < 1e-12) break;
  }
  if (inside) { /* check stationarity */
    double g[2] = {0, 0};
    for (long s = 0; s < M; s++) {
      const double* l = L + 3 * s;
      double t = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
      if (t > 0) { g[0] += (l[0] - l[1]) / t; g[1] += (l[2] - l[1]) / t; }
    }
    if (fabs(g[0]) > 1e-6 * M || fabs(g[1]) > 1e-6 * M) inside = 0;
  }
  if (!inside) {
    double best = -INFINITY, kk[3];
    const int E[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int e = 0; e < 3; e++) {
      double l2 = edge(L, M, E[e][0], E[e][1], kk);
      if (l2 > best) { best = l2; k[0] = kk[0]; k[1] = kk[1]; k[2] = kk[2]; }
    }
  }
  kout[0] = k[0]; kout[1] = k[1]; kout[2] = k[2];
  free(L);
}
