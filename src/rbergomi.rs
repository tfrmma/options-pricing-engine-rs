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
// formulas checked against arxiv 1507.03004v4 directly, not from memory.
//
// path simulation (this commit): Yn(t) is the T BSS variant (paper's eq 3.7),
// sigma == 1 identically for rBergomi (the eta/vol-of-vol scaling happens
// outside the integral, in the variance formula below, not inside the
// kernel), so Xi_k = W^n_k, no separate vol-of-vol process to simulate for
// the kernel itself. Gamma/gamma the tail convolution weights via the FFT
// per Remark 3.1, kappa=2 near terms via the Sigma/Cholesky from this file.

use crate::types::{ForwardVarianceCurve, RoughBergomiParams};
use realfft::RealFftPlanner;
use realfft::num_complex::Complex64;
use std::sync::Arc;

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

// FFT plan for the tail convolution, built once per pricer call (see mc.rs's
// mc_rough_bergomi) and shared read-only across every path via Arc clones,
// not rebuilt per path. plan construction (twiddle factor tables) is the
// expensive part, the actual per-path transform is cheap in comparison.
pub(crate) struct ConvPlan {
    fft_len: usize,
    r2c: Arc<dyn realfft::RealToComplex<f64>>,
    c2r: Arc<dyn realfft::ComplexToReal<f64>>,
}

pub(crate) fn build_conv_plan(path_len: usize) -> ConvPlan {
    // zero-pad past 2*path_len-1 so the circular wraparound FFT convolution
    // gives, lands outside the region we read, gives a real (non-circular)
    // linear convolution back.
    let fft_len = (2 * path_len - 1).next_power_of_two();
    let mut planner = RealFftPlanner::<f64>::new();
    ConvPlan {
        fft_len,
        r2c: planner.plan_fft_forward(fft_len),
        c2r: planner.plan_fft_inverse(fft_len),
    }
}

// linear (not circular) causal convolution, conv[m] = sum_{k=0}^{m} gamma[k]*xi[m-k],
// m = 0..path_len-1. gamma and xi must both be exactly path_len long, zero-padded
// internally to fft_len before transforming.
pub(crate) fn convolve_causal(plan: &ConvPlan, gamma: &[f64], xi: &[f64]) -> Vec<f64> {
    let path_len = gamma.len();
    debug_assert_eq!(path_len, xi.len());

    let mut a = vec![0.0; plan.fft_len];
    let mut b = vec![0.0; plan.fft_len];
    a[..path_len].copy_from_slice(gamma);
    b[..path_len].copy_from_slice(xi);

    let mut spec_a: Vec<Complex64> = plan.r2c.make_output_vec();
    let mut spec_b: Vec<Complex64> = plan.r2c.make_output_vec();
    plan.r2c.process(&mut a, &mut spec_a).expect("forward fft");
    plan.r2c.process(&mut b, &mut spec_b).expect("forward fft");

    for (x, y) in spec_a.iter_mut().zip(spec_b.iter()) {
        *x *= y;
    }

    let mut out = vec![0.0; plan.fft_len];
    plan.c2r.process(&mut spec_a, &mut out).expect("inverse fft");

    // realfft doesn't normalize the inverse transform, that's on the caller.
    let scale = 1.0 / plan.fft_len as f64;
    out.truncate(path_len);
    out.iter_mut().for_each(|v| *v *= scale);
    out
}

// tail weights Gamma_k = g(b*_k/n), eq in Remark 3.1. Gamma_k = 0 for
// k=1,2 (kappa=2, those are the exact near terms from Sigma instead),
// index 0-based here: gamma[k-1] holds Gamma_k. same for every path at a
// given (alpha, n_steps, n), build once outside the per-path loop.
pub(crate) fn build_gamma(alpha: f64, path_len: usize, n: f64) -> Vec<f64> {
    let mut gamma = vec![0.0; path_len];
    for k in 3..=path_len {
        let b_star = optimal_eval_point(alpha, k);
        gamma[k - 1] = (b_star / n).powf(alpha);
    }
    gamma
}

// full variance path v(t_i), i=0..n_steps-1, plus the matching dZ increments
// for the price process, eq 3.4/3.7 (near+far) folded into the rBergomi
// variance formula from section 3.3. sign=-1.0 for the antithetic sibling,
// negating the underlying normals negates every derived quantity here since
// everything downstream (Cholesky, the convolution, Z) is linear in them.
//
// z0/z1/z2/zperp are raw iid N(0,1) draws, one point per step, RNG lives in
// mc.rs (draw_rbergomi_step), this stays a pure function of its inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_variance_and_dz(
    params: &RoughBergomiParams, curve: &ForwardVarianceCurve, dt: f64,
    l: &[f64], plan: &ConvPlan, gamma: &[f64],
    z0: &[f64], z1: &[f64], z2: &[f64], zperp: &[f64], sign: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_steps = z0.len();
    let alpha = params.alpha();
    let sqrt_2a1 = (2.0 * alpha + 1.0).sqrt();
    let sqdt = dt.sqrt();

    let mut w_plain = vec![0.0; n_steps]; // W^n_i
    let mut w1      = vec![0.0; n_steps]; // W^n_{i,1}
    let mut w2      = vec![0.0; n_steps]; // W^n_{i,2}
    let mut dz      = vec![0.0; n_steps];

    for i in 0..n_steps {
        let (a, b, c, d) = (sign * z0[i], sign * z1[i], sign * z2[i], sign * zperp[i]);
        // L * (a,b,c), L lower-triangular flat 3x3 from cholesky_lower
        let wi  = l[0] * a;
        let wi1 = l[3] * a + l[4] * b;
        let wi2 = l[6] * a + l[7] * b + l[8] * c;

        w_plain[i] = wi;
        w1[i] = wi1;
        w2[i] = wi2;
        dz[i] = params.rho * wi + (1.0 - params.rho * params.rho).sqrt() * d * sqdt;
    }

    let conv = convolve_causal(plan, gamma, &w_plain);

    let mut v = vec![0.0; n_steps];
    for i in 0..n_steps {
        // Yn(i/n) = Ycheck (near, kappa=2) + Yhat (far, the convolution).
        // min{i, kappa} per eq 3.7: i=0 has neither term, i=1 only the k=1 term.
        let y_near = match i {
            0 => 0.0,
            1 => w1[0],
            _ => w1[i - 1] + w2[i - 2],
        };
        let y_far = if i == 0 { 0.0 } else { conv[i - 1] };
        let y = y_near + y_far;

        let t = i as f64 * dt;
        v[i] = curve.variance_at(t) * (params.eta * sqrt_2a1 * y - 0.5 * params.eta * params.eta * t.powf(2.0 * alpha + 1.0)).exp();
    }

    (v, dz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    fn std_normal_pair<R: Rng>(rng: &mut R) -> (f64, f64) {
        let u1: f64 = rng.gen_range(1e-12..1.0);
        let u2: f64 = rng.gen();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }

    #[test]
    fn convolve_causal_matches_direct_sum() {
        let plan  = build_conv_plan(6);
        let gamma = vec![0.0, 0.0, 1.5, -0.7, 0.3, 2.1];
        let xi    = vec![1.0, -2.0, 0.5, 3.0, -1.5, 0.8];

        let fft_result = convolve_causal(&plan, &gamma, &xi);

        let mut direct = [0.0; 6];
        for m in 0..6 {
            for k in 0..=m {
                direct[m] += gamma[k] * xi[m - k];
            }
        }

        for m in 0..6 {
            assert!((fft_result[m] - direct[m]).abs() < 1e-9,
                "m={m} fft={} direct={}", fft_result[m], direct[m]);
        }
    }

    #[test]
    fn variance_of_simulated_y_matches_exact_scheme_variance() {
        // Yn(i/n) (the T BSS discretization, eq 3.7) is, by construction, an
        // exact linear combination of the iid-across-steps (W^n_m, W^n_m1,
        // W^n_m2) vectors. its variance UNDER THE SCHEME ITSELF (not the
        // continuous-time process it approximates, that comparison converges
        // much more slowly per Corollary 2.7 and isn't what this test is
        // checking) has a closed form: the near term (steps i-1, i-2) and
        // the far term (steps 0..i-3, weighted by Gamma_k which is exactly
        // zero for k=1,2) touch disjoint step indices with nonzero weight,
        // so they're independent, no cross term.
        //   Var(Yn(i/n)) = Sigma_{2,2} + Sigma_{3,3} + sum_{k=3}^{i} Gamma_k^2 / n
        // recovered empirically from the eta=1, flat xi0=1 variance path via
        // Y = (ln(v) + 0.5 t^(2a+1)) / sqrt(2a+1).
        let alpha = -0.43;
        let n_steps = 60;
        let dt = 1.0 / n_steps as f64;
        let n = 1.0 / dt;

        let params = RoughBergomiParams { eta: 1.0, rho: -0.5, hurst: alpha + 0.5 };
        let curve = ForwardVarianceCurve::new(vec![1.0], vec![1.0]);

        let sigma = hybrid_scheme_covariance(alpha, 2, n);
        let l     = cholesky_lower(&sigma, 3).unwrap();
        let plan  = build_conv_plan(n_steps);
        let gamma = build_gamma(alpha, n_steps, n);

        let check_i = 30usize;
        let t_i = check_i as f64 * dt;

        let exact_var = sigma[4] + sigma[8] // Sigma_{2,2}, Sigma_{3,3}: flat 3x3, (1,1)->4, (2,2)->8
            + (3..=check_i).map(|k| gamma[k - 1].powi(2) / n).sum::<f64>();

        let reps = 40_000;
        let mut rng = SmallRng::seed_from_u64(11);
        let (mut sum_y, mut sum_y2) = (0.0, 0.0);

        for _ in 0..reps {
            let mut z0 = vec![0.0; n_steps];
            let mut z1 = vec![0.0; n_steps];
            let mut z2 = vec![0.0; n_steps];
            let mut zperp = vec![0.0; n_steps];
            for i in 0..n_steps {
                let (a, b) = std_normal_pair(&mut rng);
                z0[i] = a; z1[i] = b;
                let (c, d) = std_normal_pair(&mut rng);
                z2[i] = c; zperp[i] = d;
            }
            let (v, _dz) = simulate_variance_and_dz(&params, &curve, dt, &l, &plan, &gamma,
                &z0, &z1, &z2, &zperp, 1.0);
            let y = (v[check_i].ln() + 0.5 * t_i.powf(2.0 * alpha + 1.0)) / (2.0 * alpha + 1.0).sqrt();
            sum_y += y;
            sum_y2 += y * y;
        }

        let mean_y = sum_y / reps as f64;
        let empirical_var = sum_y2 / reps as f64 - mean_y * mean_y;

        // se of a sample-variance estimator of a Gaussian target ~ var*sqrt(2/reps),
        // 6-sigma band, generous but not infinite.
        let se = exact_var * (2.0 / reps as f64).sqrt();
        assert!((empirical_var - exact_var).abs() < 6.0 * se,
            "empirical={empirical_var:.6} exact={exact_var:.6} se={se:.6}");
    }

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
