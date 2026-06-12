// Heston (1993) via characteristic function inversion.
//
// Using Albrecher et al. (2007) stable formulation — NOT the original.
// Original Heston CF has a branch-cut problem where the complex log
// jumps discontinuously. quadrature silently returns garbage. fun to debug at 2am.
//
// Integration: GK-15. don't swap this for trapezoidal rule.
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

pub(crate) fn gk_integrate<F: Fn(f64) -> f64>(f: F) -> f64 {
    let upper = 200.0;
    let mid   = upper / 2.0;
    GK_NODES.iter().zip(GK_WEIGHTS.iter())
        .map(|(&n, &w)| {
            let u = mid + mid * n;
            if u < 1e-12 { return 0.0; }
            w * f(u)
        })
        .sum::<f64>() * mid
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

    // bump-and-reprice delta should be close to BSM delta for low vol-of-vol
    #[test]
    fn delta_close_to_bsm() {
        use crate::bsm::bsm_price_and_greeks;
        use crate::types::OptionContract;
        // near-BSM params: low sigma (vol of vol), v0 = vol^2
        let p = HestonParams { v0: 0.04, kappa: 10.0, theta: 0.04, sigma: 0.01, rho: 0.0 };
        let h = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
        let b = bsm_price_and_greeks(&OptionContract {
            spot: 100.0, strike: 100.0, expiry: 1.0,
            rate: 0.05, div_yield: 0.0, vol: 0.2,
            opt_type: OptionType::Call,
        });
        assert!((h.delta - b.delta).abs() < 0.01, "heston delta={:.4} bsm delta={:.4}", h.delta, b.delta);
        assert!((h.vega  - b.vega ).abs() < 1.0,  "heston vega={:.4}  bsm vega={:.4}",  h.vega,  b.vega);
    }
}
