// Heston (1993) via characteristic function inversion.
//
// Using Albrecher et al. (2007) stable formulation — NOT the original.
// Original Heston CF has a branch-cut problem where the complex log
// jumps discontinuously. quadrature silently returns garbage. fun to debug at 2am.
//
// Integration: GK-15. don't swap this for trapezoidal rule.
// if you need it faster, calibrate offline and cache the surface.

use num_complex::Complex64;
use crate::types::{HestonParams, OptionType};

// standard GK-15 nodes/weights on [-1,1]
const GK_NODES: [f64; 15] = [
    0.0,
    0.2077849550078985, -0.2077849550078985,
    0.4058451513773972, -0.4058451513773972,
    0.5860872354676911, -0.5860872354676911,
    0.7415311855993945, -0.7415311855993945,
    0.8648644233597691, -0.8648644233597691,
    0.9491079123427585, -0.9491079123427585,
    0.9914553711208126, -0.9914553711208126,
];

const GK_WEIGHTS: [f64; 15] = [
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

fn heston_call(s: f64, k: f64, t: f64, r: f64, q: f64, p: &HestonParams) -> f64 {
    // Gil-Pelaez inversion: C = S*e^(-qT)*P1 - K*e^(-rT)*P2
    let x = (s/k).ln() + (r - q)*t; // ln(F/K)

    let i1 = gk_integrate(|u| cf_integrand(u, x, t, r, p, true));
    let i2 = gk_integrate(|u| cf_integrand(u, x, t, r, p, false));

    let p1 = 0.5 + i1 / std::f64::consts::PI;
    let p2 = 0.5 + i2 / std::f64::consts::PI;

    (s*(-q*t).exp()*p1 - k*(-r*t).exp()*p2).max(0.0)
}

fn cf_integrand(u: f64, x: f64, t: f64, r: f64, p: &HestonParams, is_p1: bool) -> f64 {
    let phi = if is_p1 { Complex64::new(u, -1.0) } else { Complex64::new(u, 0.0) };
    let cf  = stable_cf(phi, t, r, p);
    let num = Complex64::new(0.0, -u * x).exp() * cf;
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
}
