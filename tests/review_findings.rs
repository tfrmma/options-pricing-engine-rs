// regression tests for the review findings on the theta/calibration branch:
//
//   1. bump-and-reprice greeks NaN'd at the domain edges the bump clamps
//      created: v0 = 0 collapsed the vol-bump span to zero (vega/vanna 0/0),
//      expiry = 0 collapsed the theta time span to zero.
//   2. the near-expiry theta fix (central difference, shrinking step) went
//      into greeks.rs but not the AD entry points, so the same option had
//      two materially different thetas depending on which public API you
//      called (~1.9x apart at 1 DTE).

use options_pricing_engine::{
    bates_greeks_ad, bates_price_and_greeks, heston_greeks_ad, heston_greeks_ad5,
    heston_price_and_greeks, BatesParams, HestonParams, OptionType,
};

fn params() -> HestonParams {
    HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
}

fn assert_all_finite(tag: &str, g: &options_pricing_engine::PricingResult) {
    for (name, v) in [
        ("price", g.price), ("delta", g.delta), ("gamma", g.gamma), ("vega", g.vega),
        ("theta", g.theta), ("rho", g.rho), ("vanna", g.vanna), ("volga", g.volga),
    ] {
        assert!(v.is_finite(), "{tag}: {name} = {v}");
    }
}

// v0 = 0 is representable (plain pub f64, nothing on the pricing path
// validates it) and the CF handles it fine — the greeks must too. the old
// clamp order made v_up == v_dn == 1e-8, dividing by a zero span.
#[test]
fn greeks_finite_at_zero_v0() {
    let p = HestonParams { v0: 0.0, ..params() };
    let g = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &p, OptionType::Call);
    assert_all_finite("v0=0", &g);
}

// expiry = 0 made t_hi == t_lo == 1e-9: theta = -0/0 = NaN, which then
// poisons any portfolio-level theta sum. an expired option has no time
// value left to decay, theta should be 0, not NaN.
#[test]
fn theta_finite_at_zero_expiry() {
    let g = heston_price_and_greeks(100.0, 100.0, 0.0, 0.05, 0.0, &params(), OptionType::Call);
    assert!(g.theta.is_finite(), "theta = {}", g.theta);
}

// the three AD entry points must agree with the bump-and-reprice path on
// theta near expiry. they kept the one-sided `(P(T-1d) - P)/1d` form after
// greeks.rs switched to a central difference with a shrinking step.
fn assert_theta_close(tag: &str, ad: f64, std: f64) {
    let rel = (ad - std).abs() / std.abs().max(1e-8);
    assert!(rel < 0.02, "{tag}: ad theta {ad:.6} vs bump-reprice {std:.6} (rel {rel:.4})");
}

#[test]
fn ad_theta_matches_bump_reprice_at_one_dte() {
    let p = params();
    let t = 1.0 / 365.0;
    let std = heston_price_and_greeks(100.0, 100.0, t, 0.05, 0.0, &p, OptionType::Call);
    let ad  = heston_greeks_ad(100.0, 100.0, t, 0.05, 0.0, &p, OptionType::Call);
    assert_theta_close("heston_greeks_ad", ad.theta, std.theta);
}

#[test]
fn ad5_theta_matches_bump_reprice_at_one_dte() {
    let p = params();
    let t = 1.0 / 365.0;
    let std = heston_price_and_greeks(100.0, 100.0, t, 0.05, 0.0, &p, OptionType::Call);
    let ad  = heston_greeks_ad5(100.0, 100.0, t, 0.05, 0.0, &p, OptionType::Call);
    assert_theta_close("heston_greeks_ad5", ad.theta, std.theta);
}

#[test]
fn bates_ad_theta_matches_bump_reprice_at_one_dte() {
    let p = params();
    let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
    let bp = BatesParams { heston: p, lambda, mu_j, sigma_j };
    let t = 1.0 / 365.0;
    let std = bates_price_and_greeks(100.0, 100.0, t, 0.05, 0.0, &bp, OptionType::Call);
    let ad  = bates_greeks_ad(100.0, 100.0, t, 0.05, 0.0, &p, lambda, mu_j, sigma_j, OptionType::Call);
    assert_theta_close("bates_greeks_ad", ad.theta, std.theta);
}
