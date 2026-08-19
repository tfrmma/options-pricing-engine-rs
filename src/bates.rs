// Bates (1996): Heston + Merton log-normal jumps.
// CF for Bates = Heston CF * jump CF. that's the whole trick.
// keep them separate, bolting jumps onto Heston post-integration doesn't work.
//
// stable_cf and gk_integrate live in heston.rs, imported directly below.
//
// Greeks: bump-and-reprice on bates_price. jumps affect vega and vanna so
// we can't just delegate to heston_price_and_greeks, the bump has to go
// through the full Bates pricer.

use num_complex::Complex64;
use crate::types::{BatesParams, HestonParams, OptionType, PricingResult};
use crate::heston::{stable_cf, gk_integrate};
use crate::greeks::{BumpPriceable, bump_and_reprice_greeks};

impl BumpPriceable for BatesParams {
    #[inline]
    fn price(&self, spot: f64, strike: f64, expiry: f64, rate: f64, div_yield: f64, opt_type: OptionType) -> f64 {
        bates_price(spot, strike, expiry, rate, div_yield, self, opt_type)
    }
    #[inline]
    fn v0(&self) -> f64 { self.heston.v0 }
    #[inline]
    fn with_v0(&self, v0: f64) -> Self {
        BatesParams { heston: HestonParams { v0, ..self.heston }, ..*self }
    }
}

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
    bump_and_reprice_greeks(spot, strike, expiry, rate, div_yield, params, opt_type)
}

fn bates_call(s: f64, k: f64, t: f64, r: f64, q: f64, bp: &BatesParams) -> f64 {
    // x = ln(S/K), see heston.rs for the derivation of why this (and not
    // ln(F/K)) plus a positive exponential sign is the correct combination.
    let x  = (s/k).ln();

    // CF(-i) normalizer for P1, including the jump component.
    //
    // the CF drift is r-q, not r — same defect and same fix as heston_call
    // (see the comment there): the payoff below discounts the stock leg by
    // e^{-qT}, so a q-free drift computes P1/P2 in a world without dividends
    // and then discounts as if there were some.
    let mu = r - q;
    let phi_mi = Complex64::new(0.0, -1.0);
    let cf_mi  = stable_cf(phi_mi, t, mu, &bp.heston) * jump_cf(phi_mi, t, bp);

    let i1 = gk_integrate(|u| bates_integrand(u, x, t, mu, bp, true, Some(cf_mi)));
    let i2 = gk_integrate(|u| bates_integrand(u, x, t, mu, bp, false, None));
    let p1 = 0.5 + i1 / std::f64::consts::PI;
    let p2 = 0.5 + i2 / std::f64::consts::PI;
    (s*(-q*t).exp()*p1 - k*(-r*t).exp()*p2).max(0.0)
}

fn bates_integrand(u: f64, x: f64, t: f64, mu: f64, bp: &BatesParams, is_p1: bool, cf_mi: Option<Complex64>) -> f64 {
    let phi = if is_p1 { Complex64::new(u, -1.0) } else { Complex64::new(u, 0.0) };
    let mut cf  = stable_cf(phi, t, mu, &bp.heston) * jump_cf(phi, t, bp);
    if let Some(norm) = cf_mi {
        cf /= norm;
    }
    let num = Complex64::new(0.0, u * x).exp() * cf;
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

    // Bates prices through the same CF-inversion quadrature as Heston, so the
    // fixed-panel under-resolution produced the same arbitrage-violating prices
    // here. A call must sit in [intrinsic, S*e^{-qT}] and decrease with strike.
    #[test]
    fn no_static_arbitrage() {
        let sets = [
            BatesParams { heston: base(), lambda: 0.5, mu_j: -0.1, sigma_j: 0.15 },
            BatesParams {
                heston: HestonParams { v0: 0.09, kappa: 1.0, theta: 0.09, sigma: 0.8, rho: -0.5 },
                lambda: 1.0, mu_j: -0.2, sigma_j: 0.25,
            },
        ];
        let (s, r, q) = (100.0, 0.03, 0.0);
        let expiries = [0.02_f64, 0.1, 0.25, 0.5, 1.0, 2.0];
        let strikes  = [70.0_f64, 85.0, 100.0, 115.0, 130.0, 150.0, 200.0];
        for bp in &sets {
            for &t in &expiries {
                let cap = s * (-q*t).exp();
                let mut prev = f64::INFINITY;
                for &k in &strikes {
                    let c = bates_price(s, k, t, r, q, bp, OptionType::Call);
                    let intrinsic = (s*(-q*t).exp() - k*(-r*t).exp()).max(0.0);
                    assert!(c >= intrinsic - 1e-4, "bates call {c} < intrinsic {intrinsic} (T={t} K={k})");
                    assert!(c <= cap + 1e-4,       "bates call {c} > spot cap {cap} (T={t} K={k})");
                    assert!(c <= prev + 1e-4,      "bates call not monotone in K (T={t} K={k}): {c} > {prev}");
                    prev = c;
                }
            }
        }
    }
}
