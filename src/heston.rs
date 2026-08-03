// Heston (1993) via characteristic function inversion.
//
// Using Albrecher et al. (2007) stable formulation, NOT the original.
// Original Heston CF has a branch-cut problem where the complex log
// jumps discontinuously. quadrature silently returns garbage. fun to debug at 2am.
//
// Integration: adaptive GK-15 (Gauss-7 embedded error estimate + subdivision).
// if you need it faster, calibrate offline and cache the surface.
//
// Greeks: bump-and-reprice by default, heston_greeks_ad (ad.rs) gives exact
// vega/vanna via forward-mode AD but isn't currently faster, see README.

use num_complex::Complex64;
use crate::types::{HestonParams, OptionType, PricingResult};
use crate::greeks::{BumpPriceable, bump_and_reprice_greeks};

impl BumpPriceable for HestonParams {
    #[inline]
    fn price(&self, spot: f64, strike: f64, expiry: f64, rate: f64, div_yield: f64, opt_type: OptionType) -> f64 {
        heston_price(spot, strike, expiry, rate, div_yield, self, opt_type)
    }
    #[inline]
    fn v0(&self) -> f64 { self.v0 }
    #[inline]
    fn with_v0(&self, v0: f64) -> Self { HestonParams { v0, ..*self } }
}

// standard GK-15 nodes/weights on [-1,1]
pub const GK_NODES: [f64; 15] = [
    0.0,
    0.2077849550078985, -0.2077849550078985,
    0.4058451513773972, -0.4058451513773972,
    0.5860872354676911, -0.5860872354676911,
    0.7415311855993945, -0.7415311855993945,
    0.8648644233597691, -0.8648644233597691,
    0.9491079123427585, -0.9491079123427585,
    0.9914553711208126, -0.9914553711208126,
];

pub const GK_WEIGHTS: [f64; 15] = [
    0.2094821410847278,
    0.2044329400752989, 0.2044329400752989,
    0.1903505780647854, 0.1903505780647854,
    0.1690047266392679, 0.1690047266392679,
    0.1406532597155259, 0.1406532597155259,
    0.1047900103222502, 0.1047900103222502,
    0.0630920926299786, 0.0630920926299786,
    0.0229353220105292, 0.0229353220105292,
];

// Gauss-7 weights for the embedded error estimate. The 7 Gauss nodes are the
// subset of GK_NODES at these indices; |K15 - G7| is the per-panel error.
pub(crate) const G7_IDX: [usize; 7] = [0, 3, 4, 7, 8, 11, 12];
pub(crate) const G7_WEIGHTS: [f64; 7] = [
    0.4179591836734694,
    0.3818300505051189, 0.3818300505051189,
    0.2797053914892766, 0.2797053914892766,
    0.1294849661688697, 0.1294849661688697,
];

pub fn heston_price(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &HestonParams, opt_type: OptionType,
) -> f64 {
    let call = heston_call(spot, strike, expiry, rate, div_yield, params);
    match opt_type {
        OptionType::Call => call,
        // put via parity, why integrate twice
        OptionType::Put  => call - spot*(-div_yield*expiry).exp() + strike*(-rate*expiry).exp(),
    }
}

pub fn heston_price_and_greeks(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &HestonParams, opt_type: OptionType,
) -> PricingResult {
    bump_and_reprice_greeks(spot, strike, expiry, rate, div_yield, params, opt_type)
}

fn heston_call(s: f64, k: f64, t: f64, r: f64, q: f64, p: &HestonParams) -> f64 {
    // Gil-Pelaez inversion: C = S*e^(-qT)*P1 - K*e^(-rT)*P2
    // x = ln(S/K), log-moneyness (NOT log-forward-moneyness; the rate term
    // is already carried by the CF itself).
    let x = (s/k).ln();

    // CF(-i) = e^{(r-q)T} is the normalizer that turns CF(u-i) into the
    // characteristic function under the stock-measure (needed for P1).
    let cf_mi = stable_cf(Complex64::new(0.0, -1.0), t, r, p);

    let i1 = gk_integrate(|u| cf_integrand(u, x, t, r, p, true, Some(cf_mi)));
    let i2 = gk_integrate(|u| cf_integrand(u, x, t, r, p, false, None));

    let p1 = 0.5 + i1 / std::f64::consts::PI;
    let p2 = 0.5 + i2 / std::f64::consts::PI;

    (s*(-q*t).exp()*p1 - k*(-r*t).exp()*p2).max(0.0)
}

fn cf_integrand(u: f64, x: f64, t: f64, r: f64, p: &HestonParams, is_p1: bool, cf_mi: Option<Complex64>) -> f64 {
    let phi = if is_p1 { Complex64::new(u, -1.0) } else { Complex64::new(u, 0.0) };
    let mut cf  = stable_cf(phi, t, r, p);
    if let Some(norm) = cf_mi {
        cf /= norm;
    }
    let num = Complex64::new(0.0, u * x).exp() * cf;
    (num / Complex64::new(0.0, u)).re
}

// Albrecher stable form. the g/(g-1) ratio avoids the log branch-cut issue
// that makes the original Heston formula blow up for longer maturities.
pub(crate) fn stable_cf(phi: Complex64, t: f64, r: f64, p: &HestonParams) -> Complex64 {
    let i = Complex64::i();
    let &HestonParams { v0, kappa, theta, sigma, rho } = p;

    let xi  = kappa - rho * sigma * phi * i;
    let d   = fast_csqrt(xi*xi + sigma*sigma * phi*(phi + i));
    let g   = (xi - d) / (xi + d);
    let edt = (-d * t).exp();
    let a   = (g*edt - 1.0) / (g - 1.0);

    let c  = (kappa*theta / (sigma*sigma)) * ((xi - d)*t - 2.0*a.ln());
    let dd = v0 * (xi - d) * (1.0 - edt) / (sigma*sigma * (1.0 - g*edt));

    (r * phi * i * t + c + dd).exp()
}

// num-complex's Complex64::sqrt() general branch goes through to_polar()/
// from_polar(), i.e. hypot + atan2 + sqrt + cos + sin (verified by reading
// the crate source, not assumed). stable_cf calls sqrt() once per CF
// evaluation, tens of millions of times over a calibration or surface
// update, and the general branch is the common case here, xi^2+sigma^2*
// phi*(phi+i) is essentially never exactly real or exactly imaginary for
// real market params.
//
// first version of this used the textbook re_out=sqrt((r+a)/2),
// im_out=sign(b)*sqrt((r-a)/2) formula directly and it LOOKED right, passed
// a 296-point correctness sweep at 1e-16 error. then zero_vol_of_vol_matches_bsm
// failed: sigma->0 makes b (the imaginary part going into this sqrt) tiny
// relative to a, so r-a is the difference of two nearly-equal quantities,
// catastrophic cancellation, garbage im_out. the sweep didn't catch it because
// its points were spread across the full circle, not concentrated near the
// real axis where this actually bites. to_polar/from_polar doesn't have this
// problem (atan2 of a tiny/large ratio is accurate, no subtraction of
// comparable magnitudes), that's the price you pay to drop it.
//
// fix: compute whichever output component is safe directly (re_out when
// a>=0, since r+a can't cancel; the imaginary-magnitude when a<0, since
// r-a can't cancel there either), then get the OTHER component via
// division (re_out*im_out = b/2 is an exact identity) instead of the
// subtraction that cancels. standard stable complex sqrt, this is not a
// novel trick, see e.g. Kahan's note on complex sqrt or any C99 csqrt impl.
#[inline]
pub(crate) fn fast_csqrt(z: Complex64) -> Complex64 {
    let (a, b) = (z.re, z.im);
    if b == 0.0 {
        return if a >= 0.0 {
            Complex64::new(a.sqrt(), b)
        } else {
            let m = (-a).sqrt();
            Complex64::new(0.0, if b.is_sign_positive() { m } else { -m })
        };
    }
    let r = a.hypot(b); // hypot-safe magnitude
    if a >= 0.0 {
        let re_out = ((r + a) / 2.0).sqrt();
        let im_out = b / (2.0 * re_out); // safe: re_out bounded away from 0 when a>=0 and r>0
        Complex64::new(re_out, im_out)
    } else {
        let im_out = ((r - a) / 2.0).sqrt().copysign(b); // safe: r-a can't cancel when a<0
        let re_out = b / (2.0 * im_out);
        Complex64::new(re_out, im_out)
    }
}

// One Gauss-Kronrod panel over [a,b]. Returns (K15 estimate, |K15 - G7| error
// estimate). Reusing the 15 node values for both rules is the whole point of the
// Gauss-Kronrod pair, the error estimate is free.
fn gk15_panel<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fv: [f64; 15] = std::array::from_fn(|i| f(c + h * GK_NODES[i]));
    let k: f64 = (0..15).map(|i| GK_WEIGHTS[i] * fv[i]).sum();
    let g: f64 = (0..7).map(|j| G7_WEIGHTS[j] * fv[G7_IDX[j]]).sum();
    (k * h, (k - g).abs() * h)
}

// Globally-adaptive Gauss-Kronrod over a finite [a,b] (QUADPACK QAG style):
// bisect the panel with the largest error estimate until the total estimated
// error is within tol. This is what GK-15 is designed for, a single fixed
// panel throws the error estimate away and aliases the oscillation.
fn adaptive_gk<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
    const MAX_PANELS: usize = 200;
    let (k0, e0) = gk15_panel(f, a, b);
    // (error, integral, a, b) per live panel
    let mut panels: Vec<(f64, f64, f64, f64)> = vec![(e0, k0, a, b)];
    let mut total = k0;
    let mut err = e0;
    while err > tol && panels.len() < MAX_PANELS {
        let w = (0..panels.len()).max_by(|&i, &j| panels[i].0.total_cmp(&panels[j].0)).unwrap();
        let (ew, kw, aw, bw) = panels.swap_remove(w);
        let m = 0.5 * (aw + bw);
        let (kl, el) = gk15_panel(f, aw, m);
        let (kr, er) = gk15_panel(f, m, bw);
        total += kl + kr - kw;
        err   += el + er - ew;
        panels.push((el, kl, aw, m));
        panels.push((er, kr, m, bw));
    }
    total
}

// Gil-Pelaez integral over u in [0, inf). The integrand oscillates (frequency
// set by log-moneyness) and, for short maturity or high vol-of-vol, decays
// slowly, so a fixed panel under-resolves it, producing arbitrage-violating
// prices in the wings. Map [0, inf) -> [0, 1] via u = (1 - t)/t and integrate
// adaptively; subdivision then resolves the whole frequency range regardless
// of maturity/moneyness.
pub(crate) fn gk_integrate<F: Fn(f64) -> f64>(f: F) -> f64 {
    let g = |t: f64| -> f64 {
        if t <= 0.0 { return 0.0; }   // u -> inf: integrand has already decayed
        let u = (1.0 - t) / t;
        if u < 1e-12 { return 0.0; }  // u -> 0: removable point of the kernel
        let v = f(u) / (t * t);
        if v.is_finite() { v } else { 0.0 }
    };
    adaptive_gk(&g, 0.0, 1.0, 1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionType;

    fn params() -> HestonParams {
        // 2*kappa*theta=0.16 > sigma^2=0.09, Feller satisfied
        HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
    }

    // fast_csqrt replaced Complex64::sqrt()'s to_polar/from_polar path in
    // stable_cf, load-bearing for every model in this crate. has to match
    // the builtin everywhere, not just at a couple of spot-checked points:
    // all four quadrants, near the negative real axis (the branch cut,
    // where sqrt is discontinuous), and across magnitudes from 1e-6 to 1e6.
    //
    // the near-axis points at very small angles are the important part.
    // the first version of this test used 37 points evenly spread around
    // the full circle (~10 degree spacing) and it passed at 1e-16 error,
    // then the actual pricer failed on sigma->0 because that regime puts a
    // TINY imaginary part next to a large real part, an angle far finer
    // than 10 degrees off the real axis, exactly where the naive algebraic
    // formula cancels catastrophically. a coarse angular sweep cannot catch
    // a bug that only bites within a fraction of a degree of the axis.
    #[test]
    fn fast_csqrt_matches_builtin() {
        let mags   = [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4, 1e6];
        let mut angles = vec![];
        let n = 37;
        for i in 0..n {
            angles.push(-std::f64::consts::PI + 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64);
        }
        // near-axis angles at decreasing scale, both sides of both axes,
        // this is what actually exercises the cancellation-prone branch
        for &tiny in &[1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
            for &base in &[0.0_f64, std::f64::consts::PI, std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
                angles.push(base + tiny);
                angles.push(base - tiny);
            }
        }

        let mut worst: f64 = 0.0;
        let mut checked = 0;
        for &mag in &mags {
            for &ang in &angles {
                let z = Complex64::new(mag * ang.cos(), mag * ang.sin());
                let expected = z.sqrt();
                let got      = fast_csqrt(z);
                let err = (got - expected).norm() / expected.norm().max(1e-300);
                worst = worst.max(err);
                checked += 1;
                assert!(err < 1e-9,
                    "fast_csqrt mismatch at z={z:?} (mag={mag}, ang={ang:.10}): got={got:?} expected={expected:?} rel_err={err:.2e}");
            }
        }
        // exact real axis, both signs, and exact imaginary axis, both signs
        for &(re, im) in &[(4.0, 0.0), (-4.0, 0.0), (-4.0, -0.0), (0.0, 4.0), (0.0, -4.0), (0.0, 0.0)] {
            let z = Complex64::new(re, im);
            let expected = z.sqrt();
            let got      = fast_csqrt(z);
            let err = (got - expected).norm();
            assert!(err < 1e-12, "fast_csqrt axis mismatch at z={z:?}: got={got:?} expected={expected:?}");
        }
        // the actual case that broke: xi ~= kappa (real, rho=0), tiny
        // sigma^2 perturbation from the jump/vol-of-vol term. reconstructed
        // from the zero_vol_of_vol_matches_bsm failure, kept as a named
        // regression point rather than trusting the sweep alone to cover it.
        {
            let z = Complex64::new(1.0, 0.0) + Complex64::new(1e-8, 0.0) * Complex64::new(0.5, -1.0) * (Complex64::new(0.5, -1.0) + Complex64::i());
            let expected = z.sqrt();
            let got      = fast_csqrt(z);
            let err = (got - expected).norm() / expected.norm().max(1e-300);
            assert!(err < 1e-9, "regression point mismatch: z={z:?} got={got:?} expected={expected:?} err={err:.2e}");
        }
        assert!(checked > 250, "sweep too small to trust, only checked {checked} points");
        eprintln!("fast_csqrt_matches_builtin: {checked} swept points + axis + regression point, worst rel err {worst:.2e}");
    }

    #[test]
    fn put_call_parity() {
        let p    = params();
        let call = heston_price(100.0, 100.0, 0.5, 0.03, 0.0, &p, OptionType::Call);
        let put  = heston_price(100.0, 100.0, 0.5, 0.03, 0.0, &p, OptionType::Put);
        let er   = (-0.03_f64 * 0.5).exp();
        assert!((call - put - (100.0 - 100.0*er)).abs() < 0.01);
    }

    #[test]
    fn price_sanity() {
        let p    = params();
        let call = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let put  = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Put);
        let er   = (-0.05_f64).exp();
        let pcp  = (call - put - 100.0 + 100.0*er).abs();
        assert!(pcp < 0.05, "pcp err = {pcp}");
        assert!(call >= 0.0);
    }

    #[test]
    fn feller_condition() {
        assert!(params().feller_ok());
    }

    #[test]
    fn greeks_signs() {
        let p = params();
        let r = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        assert!(r.delta > 0.0 && r.delta < 1.0, "delta={}", r.delta);
        assert!(r.gamma > 0.0, "gamma={}", r.gamma);
        assert!(r.vega  > 0.0, "vega={}", r.vega);
        assert!(r.rho   > 0.0, "rho={}", r.rho);
    }

    #[test]
    fn greeks_put_delta_negative() {
        let p = params();
        let r = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Put);
        assert!(r.delta < 0.0 && r.delta > -1.0, "put delta={}", r.delta);
        assert!(r.gamma > 0.0, "gamma={}", r.gamma);
    }

    // bump-and-reprice delta/vega should be close to BSM for low vol-of-vol
    // AND slow mean reversion. kappa matters here: with fast mean reversion
    // (large kappa), a bump in v0 barely moves the integrated variance over
    // [0,T] (d(IntVar)/dv0 ~ (1-e^{-kappa*T})/kappa -> 0), so Heston vega is
    // naturally much smaller than BSM vega even though delta/gamma still
    // line up. Use small kappa so v0 ~ integrated variance, like BSM assumes.
    #[test]
    fn delta_close_to_bsm() {
        use crate::bsm::bsm_price_and_greeks;
        use crate::types::OptionContract;
        // near-BSM params: low sigma (vol of vol), slow mean reversion, v0 = vol^2
        let p = HestonParams { v0: 0.04, kappa: 0.1, theta: 0.04, sigma: 0.01, rho: 0.0 };
        let h = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let b = bsm_price_and_greeks(&OptionContract {
            spot: 100.0, strike: 100.0, expiry: 1.0,
            rate: 0.05, div_yield: 0.0, vol: 0.2,
            opt_type: OptionType::Call,
        });
        assert!((h.delta - b.delta).abs() < 0.01, "heston delta={:.4} bsm delta={:.4}", h.delta, b.delta);
        // vega is damped by the (1-e^{-kT})/k factor above: a v0 bump moves the
        // integrated variance by less than one-for-one, so vega = bsm_vega * that
        // factor (~0.95 here), not bsm_vega itself.
        let damp = (1.0 - (-p.kappa * 1.0_f64).exp()) / p.kappa;
        assert!((h.vega - b.vega * damp).abs() < 0.5,
            "heston vega={:.4} bsm vega*{:.4}={:.4}", h.vega, damp, b.vega * damp);
    }

    // Static no-arbitrage across a maturity x strike grid: a call must sit in
    // [intrinsic, S*e^{-qT}] and decrease with strike. The fixed-panel rule broke
    // both in the short-dated / wing regime (calls below intrinsic, rising in K).
    #[test]
    fn no_static_arbitrage() {
        let sets = [
            HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 },
            HestonParams { v0: 0.09, kappa: 1.0, theta: 0.09, sigma: 0.8, rho: -0.5 },
            HestonParams { v0: 0.04, kappa: 0.5, theta: 0.04, sigma: 0.5, rho: -0.7 },
        ];
        let (s, r, q) = (100.0, 0.03, 0.0);
        let expiries = [0.02_f64, 0.1, 0.25, 0.5, 1.0, 2.0];
        let strikes  = [70.0_f64, 85.0, 100.0, 115.0, 130.0, 150.0, 200.0];
        for p in &sets {
            for &t in &expiries {
                let cap = s * (-q*t).exp();
                let mut prev = f64::INFINITY;
                for &k in &strikes {
                    let c = heston_price(s, k, t, r, q, p, OptionType::Call);
                    let intrinsic = (s*(-q*t).exp() - k*(-r*t).exp()).max(0.0);
                    assert!(c >= intrinsic - 1e-4, "call {c} < intrinsic {intrinsic} (T={t} K={k})");
                    assert!(c <= cap + 1e-4,       "call {c} > spot cap {cap} (T={t} K={k})");
                    assert!(c <= prev + 1e-4,      "call not monotone in K (T={t} K={k}): {c} > {prev}");
                    prev = c;
                }
            }
        }
    }

    // As vol-of-vol -> 0 the variance is frozen at v0, so Heston must collapse
    // onto BSM(vol = sqrt(v0)) at every strike. The fixed panel missed this by
    // up to ~0.4 in the wings; the adaptive rule recovers it to ~1e-6.
    #[test]
    fn zero_vol_of_vol_matches_bsm() {
        use crate::bsm::bsm_price;
        use crate::types::OptionContract;
        let p = HestonParams { v0: 0.04, kappa: 1.0, theta: 0.04, sigma: 1e-4, rho: 0.0 };
        let (s, r, q, vol) = (100.0, 0.03, 0.0, 0.2);
        for &t in &[0.25_f64, 1.0, 2.0] {
            for &k in &[70.0_f64, 85.0, 100.0, 115.0, 130.0] {
                let h = heston_price(s, k, t, r, q, &p, OptionType::Call);
                let b = bsm_price(&OptionContract {
                    spot: s, strike: k, expiry: t, rate: r, div_yield: q, vol,
                    opt_type: OptionType::Call,
                });
                assert!((h - b).abs() < 0.02, "heston {h:.5} vs bsm {b:.5} (T={t} K={k})");
            }
        }
    }
}
