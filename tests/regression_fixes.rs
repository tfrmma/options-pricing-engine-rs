// regression tests for three defects found by numerical cross-checking
// against closed forms (degenerate-Heston-vs-BSM, FD reprice):
//
//   1. black76 theta: carry/discount term had the same sign flip that
//      theta_calc in bsm.rs had before d88a1ff. error = 2*r*price, always
//      more negative than truth, so a sign-only test can't catch it.
//   2. bates.rs / ad.rs CF drift: heston.rs got the r -> r-q fix (d88a1ff)
//      but the same defect stayed in bates_call and the AD forward passes,
//      so bates and AD prices disagree with heston at q > 0.
//   3. ad5 additionally dropped the e^{-qT} on the stock leg entirely.

use options_pricing_engine::{
    bates_greeks_ad, bates_price, black76_price_and_greeks, bsm_price_and_greeks,
    heston_greeks_ad, heston_greeks_ad5, heston_price, BatesParams, HestonParams,
    OptionContract, OptionType,
};

fn heston_params() -> HestonParams {
    HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
}

// black76 theta has to match a reprice in expiry, same test shape as
// theta_matches_finite_difference in bsm.rs. the flipped carry term is
// invisible to sign checks (a call's theta stays negative either way).
#[test]
fn black76_theta_matches_fd() {
    let (fwd, k, t, v) = (100.0, 100.0, 1.0, 0.2);
    let dt = 1e-5;
    for r in [0.0, 0.05] {
        for ot in [OptionType::Call, OptionType::Put] {
            let analytic = black76_price_and_greeks(fwd, k, t, r, v, ot).theta;
            let fd = (black76_price_and_greeks(fwd, k, t - dt, r, v, ot).price
                    - black76_price_and_greeks(fwd, k, t + dt, r, v, ot).price) / (2.0 * dt);
            assert!((analytic - fd).abs() < 1e-4,
                    "r={r} {ot:?}: analytic {analytic:.6} vs fd {fd:.6}");
        }
    }
}

// consistency with BSM at r=q=0 where theta is pure time-value decay
// (no carry term to hide in). pins the vega-decay part independently.
#[test]
fn black76_theta_matches_bsm_at_zero_rate() {
    let c = OptionContract {
        spot: 100.0, strike: 100.0, expiry: 1.0,
        rate: 0.0, div_yield: 0.0, vol: 0.2,
        opt_type: OptionType::Call,
    };
    let b76 = black76_price_and_greeks(100.0, 100.0, 1.0, 0.0, 0.2, OptionType::Call);
    let bsm = bsm_price_and_greeks(&c);
    assert!((b76.theta - bsm.theta).abs() < 1e-10,
            "b76 {:.6} vs bsm {:.6}", b76.theta, bsm.theta);
}

// with lambda = 0 the jump CF is identically 1, so bates must equal heston
// for any q. before the fix bates_call still fed drift r (not r-q) to the
// CF, so this disagreed by ~0.2 at q=5%.
#[test]
fn bates_lambda0_with_dividend_matches_heston() {
    let hp = heston_params();
    let bp = BatesParams { heston: hp, lambda: 0.0, mu_j: 0.0, sigma_j: 0.1 };
    for q in [0.0, 0.02, 0.05] {
        for k in [90.0, 100.0, 110.0] {
            for ot in [OptionType::Call, OptionType::Put] {
                let h = heston_price(100.0, k, 1.0, 0.05, q, &hp, ot);
                let b = bates_price(100.0, k, 1.0, 0.05, q, &bp, ot);
                assert!((h - b).abs() < 1e-6,
                        "q={q} K={k} {ot:?}: heston {h:.6} vs bates {b:.6}");
            }
        }
    }
}

// AD forward passes must price identically to the standard pricer at q > 0.
// the AD paths kept drift r after heston.rs was fixed, so they drifted apart.
#[test]
fn ad_heston_price_matches_pricer_with_dividend() {
    let hp = heston_params();
    for q in [0.0, 0.05] {
        for ot in [OptionType::Call, OptionType::Put] {
            let px = heston_price(100.0, 100.0, 1.0, 0.05, q, &hp, ot);
            let ad = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, q, &hp, ot).price;
            assert!((px - ad).abs() < 1e-4,
                    "q={q} {ot:?}: pricer {px:.6} vs ad {ad:.6}");
        }
    }
}

// ad5 had two q defects: the CF drift AND a missing e^{-qT} on the stock leg
// (it effectively priced q=0 outright, off by +3.15 at q=5% ATM).
#[test]
fn ad5_price_matches_pricer_with_dividend() {
    let hp = heston_params();
    for q in [0.0, 0.05] {
        for ot in [OptionType::Call, OptionType::Put] {
            let px = heston_price(100.0, 100.0, 1.0, 0.05, q, &hp, ot);
            let ad = heston_greeks_ad5(100.0, 100.0, 1.0, 0.05, q, &hp, ot).price;
            assert!((px - ad).abs() < 1e-4,
                    "q={q} {ot:?}: pricer {px:.6} vs ad5 {ad:.6}");
        }
    }
}

// same drift defect in the bates AD path, with jumps on so the jump CF and
// the heston CF are both exercised.
#[test]
fn ad_bates_price_matches_pricer_with_dividend() {
    let hp = heston_params();
    let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
    let bp = BatesParams { heston: hp, lambda, mu_j, sigma_j };
    for q in [0.0, 0.05] {
        for ot in [OptionType::Call, OptionType::Put] {
            let px = bates_price(100.0, 100.0, 1.0, 0.05, q, &bp, ot);
            let ad = bates_greeks_ad(100.0, 100.0, 1.0, 0.05, q, &hp, lambda, mu_j, sigma_j, ot).price;
            assert!((px - ad).abs() < 1e-4,
                    "q={q} {ot:?}: pricer {px:.6} vs ad {ad:.6}");
        }
    }
}
