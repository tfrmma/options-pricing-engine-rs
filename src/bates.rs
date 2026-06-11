// Bates (1996): Heston + Merton log-normal jumps.
// CF for Bates = Heston CF * jump CF. that's the whole trick.
// keep them separate — bolting jumps onto Heston post-integration doesn't work.
//
// stable_cf and gk_integrate live in heston.rs — imported directly below.

use num_complex::Complex64;
use crate::types::{BatesParams, OptionType};
use crate::heston::{heston_price, stable_cf, gk_integrate};

pub fn bates_price(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &BatesParams, opt_type: OptionType,
) -> f64 {
    let call = bates_call(spot, strike, expiry, rate, div_yield, params);
    match opt_type {
        OptionType::Call => call,
        OptionType::Put  => call - spot*(-div_yield*expiry).exp() + strike*(-rate*expiry).exp(),
    }
}

fn bates_call(s: f64, k: f64, t: f64, r: f64, q: f64, bp: &BatesParams) -> f64 {
    let x  = (s/k).ln() + (r - q)*t;
    let i1 = gk_integrate(|u| bates_integrand(u, x, t, r, bp, true));
    let i2 = gk_integrate(|u| bates_integrand(u, x, t, r, bp, false));
    let p1 = 0.5 + i1 / std::f64::consts::PI;
    let p2 = 0.5 + i2 / std::f64::consts::PI;
    (s*(-q*t).exp()*p1 - k*(-r*t).exp()*p2).max(0.0)
}

fn bates_integrand(u: f64, x: f64, t: f64, r: f64, bp: &BatesParams, is_p1: bool) -> f64 {
    let phi = if is_p1 { Complex64::new(u, -1.0) } else { Complex64::new(u, 0.0) };
    let cf  = stable_cf(phi, t, r, &bp.heston) * jump_cf(phi, t, bp);
    let num = Complex64::new(0.0, -u * x).exp() * cf;
    (num / Complex64::new(0.0, u)).re
}

// jump component: Merton log-normal.
// mu_j_bar is the compensation term that keeps the process a Q-martingale.
fn jump_cf(phi: Complex64, t: f64, bp: &BatesParams) -> Complex64 {
    let i    = Complex64::i();
    let &BatesParams { lambda, mu_j, sigma_j, .. } = bp;
    let comp = (mu_j + 0.5*sigma_j*sigma_j).exp() - 1.0;
    let jump = (phi*i*mu_j - 0.5*phi*phi*sigma_j*sigma_j).exp();
    (lambda * t * (jump - 1.0 - i*phi*comp)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HestonParams, BatesParams, OptionType};

    fn base() -> HestonParams {
        HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
    }

    #[test]
    fn recovers_heston_no_jumps() {
        let bp = BatesParams { heston: base(), lambda: 0.0, mu_j: 0.0, sigma_j: 1e-8 };
        let bates_px  = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &bp, OptionType::Call);
        let heston_px = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &base(), OptionType::Call);
        assert!((bates_px - heston_px).abs() < 0.02,
            "bates={bates_px:.4} heston={heston_px:.4}");
    }

    #[test]
    fn put_call_parity() {
        let bp   = BatesParams { heston: base(), lambda: 0.5, mu_j: -0.1, sigma_j: 0.15 };
        let call = bates_price(100.0, 100.0, 0.5, 0.03, 0.0, &bp, OptionType::Call);
        let put  = bates_price(100.0, 100.0, 0.5, 0.03, 0.0, &bp, OptionType::Put);
        let er   = (-0.03_f64 * 0.5).exp();
        assert!((call - put - (100.0 - 100.0*er)).abs() < 0.02);
    }
}
