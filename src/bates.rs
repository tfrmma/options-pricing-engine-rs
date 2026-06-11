// Bates (1996): Heston + Merton log-normal jumps.
// CF for Bates = Heston CF * jump CF. that's the whole trick.
// keep them separate — bolting jumps onto Heston post-integration doesn't work.
//
// stable_cf and gk_integrate live in heston.rs — imported directly below.
//
// Greeks: bump-and-reprice on bates_price. jumps affect vega and vanna so
// we can't just delegate to heston_price_and_greeks — the bump has to go
// through the full Bates pricer.

use num_complex::Complex64;
use crate::types::{BatesParams, OptionType, PricingResult};
use crate::heston::{stable_cf, gk_integrate};

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

pub fn bates_price_and_greeks(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &BatesParams, opt_type: OptionType,
) -> PricingResult {
    let price = bates_price(spot, strike, expiry, rate, div_yield, params, opt_type);

    let ds = 0.01 * spot;
    let dv = 0.01;
    let dr = 1e-4;
    let dt = 1.0 / 365.0;

    let pu    = bates_price(spot + ds, strike, expiry, rate, div_yield, params, opt_type);
    let pd    = bates_price(spot - ds, strike, expiry, rate, div_yield, params, opt_type);
    let delta = (pu - pd) / (2.0 * ds);
    let gamma = (pu - 2.0*price + pd) / (ds * ds);

    // vega bumps v0 in vol units, same convention as heston_price_and_greeks
    let v_cur = params.heston.v0.sqrt();
    let p_vup = params_with_v0(params, (v_cur + dv).powi(2));
    let p_vdn = params_with_v0(params, (v_cur - dv).max(1e-8).powi(2));
    let vega  = (bates_price(spot, strike, expiry, rate, div_yield, &p_vup, opt_type)
               - bates_price(spot, strike, expiry, rate, div_yield, &p_vdn, opt_type))
              / (2.0 * dv);

    let t_dn  = (expiry - dt).max(1e-6);
    let theta = (bates_price(spot, strike, t_dn, rate, div_yield, params, opt_type) - price) / dt;

    let rho   = (bates_price(spot, strike, expiry, rate + dr, div_yield, params, opt_type)
               - bates_price(spot, strike, expiry, rate - dr, div_yield, params, opt_type))
              / (2.0 * dr);

    let delta_vup = {
        let pu = bates_price(spot + ds, strike, expiry, rate, div_yield, &p_vup, opt_type);
        let pd = bates_price(spot - ds, strike, expiry, rate, div_yield, &p_vup, opt_type);
        (pu - pd) / (2.0 * ds)
    };
    let delta_vdn = {
        let pu = bates_price(spot + ds, strike, expiry, rate, div_yield, &p_vdn, opt_type);
        let pd = bates_price(spot - ds, strike, expiry, rate, div_yield, &p_vdn, opt_type);
        (pu - pd) / (2.0 * ds)
    };
    let vanna = (delta_vup - delta_vdn) / (2.0 * dv);

    let p_vup2 = params_with_v0(params, (v_cur + 2.0*dv).powi(2));
    let p_vdn2 = params_with_v0(params, (v_cur - 2.0*dv).max(1e-8).powi(2));
    let vega_up = (bates_price(spot, strike, expiry, rate, div_yield, &p_vup2, opt_type)
                 - bates_price(spot, strike, expiry, rate, div_yield, params, opt_type))
                / (2.0 * dv);
    let vega_dn = (bates_price(spot, strike, expiry, rate, div_yield, params, opt_type)
                 - bates_price(spot, strike, expiry, rate, div_yield, &p_vdn2, opt_type))
                / (2.0 * dv);
    let volga   = (vega_up - vega_dn) / (2.0 * dv);

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}

#[inline]
fn params_with_v0(p: &BatesParams, v0: f64) -> BatesParams {
    BatesParams { heston: crate::types::HestonParams { v0, ..p.heston }, ..*p }
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

    fn bp_with_jumps() -> BatesParams {
        BatesParams { heston: base(), lambda: 0.5, mu_j: -0.1, sigma_j: 0.15 }
    }

    #[test]
    fn recovers_heston_no_jumps() {
        use crate::heston_price;
        let bp = BatesParams { heston: base(), lambda: 0.0, mu_j: 0.0, sigma_j: 1e-8 };
        let bates_px  = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &bp, OptionType::Call);
        let heston_px = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &base(), OptionType::Call);
        assert!((bates_px - heston_px).abs() < 0.02,
            "bates={bates_px:.4} heston={heston_px:.4}");
    }

    #[test]
    fn put_call_parity() {
        let bp   = bp_with_jumps();
        let call = bates_price(100.0, 100.0, 0.5, 0.03, 0.0, &bp, OptionType::Call);
        let put  = bates_price(100.0, 100.0, 0.5, 0.03, 0.0, &bp, OptionType::Put);
        let er   = (-0.03_f64 * 0.5).exp();
        assert!((call - put - (100.0 - 100.0*er)).abs() < 0.02);
    }

    #[test]
    fn greeks_signs() {
        let r = bates_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &bp_with_jumps(), OptionType::Call);
        assert!(r.delta > 0.0 && r.delta < 1.0, "delta={}", r.delta);
        assert!(r.gamma > 0.0, "gamma={}", r.gamma);
        assert!(r.vega  > 0.0, "vega={}", r.vega);
        assert!(r.rho   > 0.0, "rho={}", r.rho);
    }

    #[test]
    fn greeks_put_delta_negative() {
        let r = bates_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &bp_with_jumps(), OptionType::Put);
        assert!(r.delta < 0.0 && r.delta > -1.0, "put delta={}", r.delta);
        assert!(r.gamma > 0.0);
    }

    // with no jumps, bates greeks should match heston greeks closely
    #[test]
    fn greeks_match_heston_no_jumps() {
        use crate::heston::heston_price_and_greeks;
        let bp = BatesParams { heston: base(), lambda: 0.0, mu_j: 0.0, sigma_j: 1e-8 };
        let b  = bates_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &bp, OptionType::Call);
        let h  = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &base(), OptionType::Call);
        assert!((b.delta - h.delta).abs() < 0.01, "delta: bates={:.4} heston={:.4}", b.delta, h.delta);
        assert!((b.vega  - h.vega ).abs() < 1.0,  "vega:  bates={:.4} heston={:.4}", b.vega,  h.vega);
    }
}
