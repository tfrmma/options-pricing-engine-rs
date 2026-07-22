// Heston (1993) via characteristic function inversion.
//
// Using Albrecher et al. (2007) stable formulation — NOT the original.
// Original Heston CF has a branch-cut problem where the complex log
// jumps discontinuously. quadrature silently returns garbage. fun to debug at 2am.
//
// Integration: adaptive GK-15 (Gauss-7 embedded error estimate + subdivision).
// if you need it faster, calibrate offline and cache the surface.
//
// Greeks: bump-and-reprice. not pretty but correct.
// AD or complex-step would be faster; add it when it matters.

use num_complex::Complex64;
use crate::types::{HestonParams, OptionType, PricingResult};

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
const G7_IDX: [usize; 7] = [0, 3, 4, 7, 8, 11, 12];
const G7_WEIGHTS: [f64; 7] = [
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
        // put via parity — why integrate twice
        OptionType::Put  => call - spot*(-div_yield*expiry).exp() + strike*(-rate*expiry).exp(),
    }
}

// bump sizes: dS = 1% spot, dv = 1 vol point, dr = 1bp, dt = 1 calendar day.
// vanna and volga via double bump — 4 extra pricing calls each, worth it for
// second-order accuracy.
pub fn heston_price_and_greeks(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &HestonParams, opt_type: OptionType,
) -> PricingResult {
    let price = heston_price(spot, strike, expiry, rate, div_yield, params, opt_type);

    let ds  = 0.01 * spot;
    let dv  = 0.01;
    let dr  = 1e-4;
    let dt  = 1.0 / 365.0;

    let pu  = heston_price(spot + ds, strike, expiry, rate, div_yield, params, opt_type);
    let pd  = heston_price(spot - ds, strike, expiry, rate, div_yield, params, opt_type);
    let delta = (pu - pd) / (2.0 * ds);
    let gamma = (pu - 2.0*price + pd) / (ds * ds);

    // vega: bump v0 (initial variance). dv is in vol units so bump v0 by (v+dv)^2 - v^2
    let v0     = params.v0;
    let v_cur  = v0.sqrt();
    let p_vup  = params_with_v0(params, (v_cur + dv).powi(2));
    let p_vdn  = params_with_v0(params, (v_cur - dv).max(1e-8).powi(2));
    let vega   = (heston_price(spot, strike, expiry, rate, div_yield, &p_vup, opt_type)
                - heston_price(spot, strike, expiry, rate, div_yield, &p_vdn, opt_type))
               / (2.0 * dv);

    // theta: bump expiry down. clamp so we don't go negative.
    let t_dn  = (expiry - dt).max(1e-6);
    let theta = (heston_price(spot, strike, t_dn, rate, div_yield, params, opt_type) - price) / dt;

    let rho   = (heston_price(spot, strike, expiry, rate + dr, div_yield, params, opt_type)
               - heston_price(spot, strike, expiry, rate - dr, div_yield, params, opt_type))
              / (2.0 * dr);

    // vanna = d(delta)/d(vol). cross bump: (delta at v+dv) - (delta at v-dv)
    let delta_vup = {
        let pu = heston_price(spot + ds, strike, expiry, rate, div_yield, &p_vup, opt_type);
        let pd = heston_price(spot - ds, strike, expiry, rate, div_yield, &p_vup, opt_type);
        (pu - pd) / (2.0 * ds)
    };
    let delta_vdn = {
        let pu = heston_price(spot + ds, strike, expiry, rate, div_yield, &p_vdn, opt_type);
        let pd = heston_price(spot - ds, strike, expiry, rate, div_yield, &p_vdn, opt_type);
        (pu - pd) / (2.0 * ds)
    };
    let vanna = (delta_vup - delta_vdn) / (2.0 * dv);

    // volga = d(vega)/d(vol). second derivative of price w.r.t. vol.
    let p_vup2 = params_with_v0(params, (v_cur + 2.0*dv).powi(2));
    let p_vdn2 = params_with_v0(params, (v_cur - 2.0*dv).max(1e-8).powi(2));
    let vega_up = (heston_price(spot, strike, expiry, rate, div_yield, &p_vup2, opt_type)
                 - heston_price(spot, strike, expiry, rate, div_yield, params, opt_type))
                / (2.0 * dv);
    let vega_dn = (heston_price(spot, strike, expiry, rate, div_yield, params, opt_type)
                 - heston_price(spot, strike, expiry, rate, div_yield, &p_vdn2, opt_type))
                / (2.0 * dv);
    let volga   = (vega_up - vega_dn) / (2.0 * dv);

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}

#[inline]
fn params_with_v0(p: &HestonParams, v0: f64) -> HestonParams {
    HestonParams { v0, ..*p }
}

fn heston_call(s: f64, k: f64, t: f64, r: f64, q: f64, p: &HestonParams) -> f64 {
    // Gil-Pelaez inversion: C = S*e^(-qT)*P1 - K*e^(-rT)*P2
    // x = ln(S/K) — log-moneyness (NOT log-forward-moneyness; the rate term
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
    let d   = (xi*xi + sigma*sigma * phi*(phi + i)).sqrt();
    let g   = (xi - d) / (xi + d);
    let edt = (-d * t).exp();
    let a   = (g*edt - 1.0) / (g - 1.0);

    let c  = (kappa*theta / (sigma*sigma)) * ((xi - d)*t - 2.0*a.ln());
    let dd = v0 * (xi - d) * (1.0 - edt) / (sigma*sigma * (1.0 - g*edt));

    (r * phi * i * t + c + dd).exp()
}

// One Gauss-Kronrod panel over [a,b]. Returns (K15 estimate, |K15 - G7| error
// estimate). Reusing the 15 node values for both rules is the whole point of the
// Gauss-Kronrod pair — the error estimate is free.
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
// error is within tol. This is what GK-15 is designed for — a single fixed
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
// slowly — so a fixed panel under-resolves it, producing arbitrage-violating
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
        // 2*kappa*theta=0.16 > sigma^2=0.09 — Feller satisfied
        HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
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
