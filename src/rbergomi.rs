// rough Bergomi: hybrid scheme (Bennedsen, Lunde, Pakkanen 2017) kernel.
//
// builds the covariance matrix for the (kappa+1)-dim correlated gaussian vector
// (W^n_i, W^n_{i,1}, ..., W^n_{i,kappa}) needed to *exactly* simulate the near-
// kernel Wiener integrals per step (their eq 3.5), plus its Cholesky factor, and
// the optimal tail evaluation points b*_k (their Prop 2.8) used later by the
// far Riemann-sum / FFT convolution.
//
// commit 1: kernel + Sigma + Cholesky only. path simulation (mc.rs) and the FFT
// convolution for the tail aren't wired in yet, this is deliberately just the
// piece that has to be right before touching mc.rs.
//
// for rBergomi specifically g(x) = x^alpha exactly, no slowly-varying L_g
// correction on top, so the near-term approximation in the paper's eq 2.4 is
// exact equality here, not an approximation stacked on an approximation.
//
// formulas checked against arxiv 1507.03004v4 directly.
//
// nothing here is called outside this file's own tests yet, mc.rs wires it in
// next commit. allow(dead_code) instead of leaving clippy -D warnings red for
// a commit that's deliberately just the kernel.
#![allow(dead_code)]

// Gauss hypergeometric 2F1(a, b, c, z), series definition, |z| < 1 only.
// only ever called here with b=1, which collapses (b)_n/n! to 1, but kept
// general since that's what the paper states (eq 3.5) and it costs nothing extra.
//
// every z we pass in is (j-1)/(k-1) or (j-2)/(k-2) for j<k, kappa <= 3ish, so z
// stays well under 1 and this converges in well under 100 terms. don't call this
// near z=1, it'd still converge (c-a-b = 2*alpha+1 > 0 for alpha > -0.5, Gauss's
// theorem) but slowly, not a case we hit here.
fn hyp2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    debug_assert!(z.abs() < 1.0, "series doesn't converge for |z| >= 1");

    let mut term = 1.0;
    let mut sum  = 1.0;
    let mut n    = 0.0;
    loop {
        term *= (a + n) * (b + n) / (c + n) * z / (n + 1.0);
        sum  += term;
        n    += 1.0;
        if term.abs() < 1e-16 * sum.abs() || n > 500.0 {
            break;
        }
    }
    sum
}

// Prop 2.8. optimal evaluation point for tail cell k, k >= kappa+1, minimizes
// the asymptotic MSE among all valid b_k in [k-1, k]. not used by anything in
// this commit yet, the tail sum lives in the future mc.rs/FFT piece, but it's
// the same "kernel" section of the paper and just as easy to verify now.
pub(crate) fn optimal_eval_point(alpha: f64, k: usize) -> f64 {
    debug_assert!(k >= 1);
    let kf  = k as f64;
    let raw = (kf.powf(alpha + 1.0) - (kf - 1.0).powf(alpha + 1.0)) / (alpha + 1.0);
    raw.powf(1.0 / alpha)
}

// covariance matrix Sigma for (W^n_i, W^n_{i,1}, ..., W^n_{i,kappa}), eq 3.5.
// flat (kappa+1)x(kappa+1), row-major, same convention as LocalVolSurface,
// don't switch this to Vec<Vec<f64>>.
//
// TODO: this rebuilds Sigma from scratch every call, fine for a one-time setup
// cost but the caller (mc.rs, not written yet) should factor it once per
// (alpha, kappa, n) and reuse across every path, not per-path.
pub(crate) fn hybrid_scheme_covariance(alpha: f64, kappa: usize, n: f64) -> Vec<f64> {
    let dim = kappa + 1;
    let mut sigma = vec![0.0; dim * dim];
    let at = |i: usize, j: usize| i * dim + j;

    sigma[at(0, 0)] = 1.0 / n;

    for j in 2..=kappa + 1 {
        let jf = j as f64;

        let s1j = ((jf - 1.0).powf(alpha + 1.0) - (jf - 2.0).powf(alpha + 1.0))
            / ((alpha + 1.0) * n.powf(alpha + 1.0));
        sigma[at(0, j - 1)] = s1j;
        sigma[at(j - 1, 0)] = s1j;

        let sjj = ((jf - 1.0).powf(2.0 * alpha + 1.0) - (jf - 2.0).powf(2.0 * alpha + 1.0))
            / ((2.0 * alpha + 1.0) * n.powf(2.0 * alpha + 1.0));
        sigma[at(j - 1, j - 1)] = sjj;
    }

    // off-diagonal near x near terms, needs the hypergeometric. only fires for
    // kappa >= 2 (kappa=1 has one near term, no pair to correlate).
    for j in 2..=kappa + 1 {
        for k in (j + 1)..=kappa + 1 {
            let (jf, kf) = (j as f64, k as f64);

            let term1 = (jf - 1.0).powf(alpha + 1.0) * (kf - 1.0).powf(alpha)
                * hyp2f1(-alpha, 1.0, alpha + 2.0, (jf - 1.0) / (kf - 1.0));
            // when j=2 this term is exactly zero, (jf-2)=0 and alpha+1>0, so
            // 0^(alpha+1)=0, no need to special-case skip the 2F1 call for it.
            let term2 = (jf - 2.0).powf(alpha + 1.0) * (kf - 2.0).powf(alpha)
                * hyp2f1(-alpha, 1.0, alpha + 2.0, (jf - 2.0) / (kf - 2.0));

            let sjk = (term1 - term2) / ((alpha + 1.0) * n.powf(2.0 * alpha + 1.0));
            sigma[at(j - 1, k - 1)] = sjk;
            sigma[at(k - 1, j - 1)] = sjk;
        }
    }

    sigma
}

// lower-triangular Cholesky factor L, L * L^T = matrix, flat dim x dim, zeros
// above the diagonal. None if a pivot goes non-positive, that's a real failure
// mode for a bad alpha/kappa/n combination (e.g. alpha outside (-0.5, 0) from
// a busted calibration upstream), not something to clamp and hide.
pub(crate) fn cholesky_lower(matrix: &[f64], dim: usize) -> Option<Vec<f64>> {
    assert_eq!(matrix.len(), dim * dim);
    let mut l = vec![0.0; dim * dim];

    for i in 0..dim {
        for j in 0..=i {
            let mut sum = matrix[i * dim + j];
            for p in 0..j {
                sum -= l[i * dim + p] * l[j * dim + p];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }

    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Table 1 in the paper, H=0.07. real crypto pre-event surfaces are
    // expected to be rougher than this, see the near-H-zero test below.
    const ALPHA_PAPER: f64 = -0.43;

    #[test]
    fn hyp2f1_matches_numeric_quadrature() {
        // Lemma 4.3: int_0^a (a-x)^alpha (b-x)^alpha dx
        //          = a^(alpha+1) b^alpha / (alpha+1) * 2F1(-alpha,1,alpha+2,a/b)
        // Simpson quadrature of the LHS, independent of the series above, this
        // is the actual correctness check, not a self-consistency one.
        //
        // (a-x)^alpha has an integrable singularity at x=a (alpha=-0.43 > -1),
        // equally-spaced Simpson chokes on that, first version of this test
        // dodged it with a single near-endpoint sample and silently carried
        // ~0.7% error, caught by cross-checking against mpmath independently,
        // not by inspection. fix: substitute t=a-x then u^2=t, the u^(2a+1)
        // term is continuous at u=0 since 2*alpha+1 > 0 for alpha > -0.5.
        let alpha = ALPHA_PAPER;
        let (a, b) = (1.0_f64, 2.0_f64);

        let g = |u: f64| if u == 0.0 { 0.0 } else {
            2.0 * u.powf(2.0 * alpha + 1.0) * (b - a + u * u).powf(alpha)
        };

        let panels = 200_000;
        let u_max  = a.sqrt();
        let h = u_max / panels as f64;
        let mut integral = g(0.0) + g(u_max);
        for i in 1..panels {
            let u = i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            integral += w * g(u);
        }
        integral *= h / 3.0;

        let closed_form = a.powf(alpha + 1.0) * b.powf(alpha) / (alpha + 1.0)
            * hyp2f1(-alpha, 1.0, alpha + 2.0, a / b);

        let rel_err = (integral - closed_form).abs() / closed_form.abs();
        assert!(rel_err < 1e-5, "integral={integral} closed_form={closed_form} rel_err={rel_err}");
    }

    #[test]
    fn optimal_eval_point_lies_in_cell() {
        for alpha in [-0.05, -0.2, -0.43, -0.49] {
            for k in 1..=5 {
                let b_star = optimal_eval_point(alpha, k);
                assert!(b_star > (k - 1) as f64 && b_star < k as f64,
                    "alpha={alpha} k={k} b*={b_star}, expected in ({}, {})", k - 1, k);
            }
        }
    }

    #[test]
    fn sigma_11_matches_ito_isometry() {
        // Var(W(1/n)) = 1/n, trivial but catches an index bug immediately.
        let sigma = hybrid_scheme_covariance(ALPHA_PAPER, 2, 100.0);
        assert!((sigma[0] - 0.01).abs() < 1e-12);
    }

    #[test]
    fn sigma_is_symmetric() {
        let dim = 3;
        let sigma = hybrid_scheme_covariance(ALPHA_PAPER, 2, 100.0);
        for i in 0..dim {
            for j in 0..dim {
                assert!((sigma[i * dim + j] - sigma[j * dim + i]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn cholesky_reconstructs_hybrid_scheme_covariance() {
        // same pattern as calibration.rs's jacobi_eigen_reconstructs_*x*.
        let dim = 3;
        let sigma = hybrid_scheme_covariance(ALPHA_PAPER, 2, 100.0);
        let l = cholesky_lower(&sigma, dim).expect("Sigma should be positive definite");

        for i in 0..dim {
            for j in 0..dim {
                let mut recon = 0.0;
                for p in 0..dim {
                    recon += l[i * dim + p] * l[j * dim + p];
                }
                let err = (recon - sigma[i * dim + j]).abs();
                assert!(err < 1e-10, "i={i} j={j} recon={recon} sigma={} err={err}", sigma[i * dim + j]);
            }
        }
    }

    #[test]
    fn cholesky_stays_positive_definite_near_h_zero() {
        // this is closer to the regime the crypto short-dated surfaces are
        // actually expected to live in, not the paper's H=0.07, worth its own check.
        for alpha in [-0.49, -0.45, -0.40] {
            let sigma = hybrid_scheme_covariance(alpha, 2, 500.0);
            assert!(cholesky_lower(&sigma, 3).is_some(), "alpha={alpha} should stay PD");
        }
    }

    #[test]
    fn covariance_scales_with_n_as_expected() {
        // Sigma_jj ~ n^-(2 alpha + 1), halving n should scale the diagonal by
        // 2^(2 alpha + 1). cheap sanity check on the n-dependence.
        let alpha = ALPHA_PAPER;
        let s1 = hybrid_scheme_covariance(alpha, 2, 100.0);
        let s2 = hybrid_scheme_covariance(alpha, 2, 200.0);
        let expected_ratio = 2.0_f64.powf(2.0 * alpha + 1.0);
        let actual_ratio   = s1[4] / s2[4]; // (1,1) in a flat 3x3, i*dim+j = 1*3+1
        assert!((actual_ratio - expected_ratio).abs() / expected_ratio < 1e-9,
            "actual={actual_ratio} expected={expected_ratio}");
    }
}
