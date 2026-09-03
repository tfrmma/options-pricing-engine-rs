// Deribit inverse (coin-settled) option: price and Greeks in coin terms.
//
// Deribit publishes their own BS formula for inverse options:
// https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options
// C = X*N(d1) - K*N(d2)*e^(-RT), R = ln(F/X)/T, X = index, F = forward (the
// corresponding future's mark price). the coin price on the order book is
// that USD price divided by X.
//
// substitute e^(-RT) = X/F into C, divide by X, the X dependence cancels
// completely, leaves a forward-only formula:
//   call_coin = N(d1) - (K/F)*N(d2)
//   put_coin  = (K/F)*N(-d2) - N(-d1)
// unified via phi = opt_type.sign(), same pattern bsm.rs already uses for
// the regular pricer instead of a match arm per Greek:
//   price_coin = phi*N(phi*d1) - phi*(K/F)*N(phi*d2)
//
// Greeks worked out by hand from that (chain rule through the /F, not
// reused BS Greeks), cross-checked against put-call parity (call - put =
// 1 - K/F, a coin forward struck at K in USD is (F-K)/F in coin terms) and
// against finite differences below. Alexander & Imeraj (2021) call this
// the "naive" inverse parametrization and derive a quanto-corrected
// version with an extra convexity term; this implements what Deribit's
// own docs actually use, matching the venue's mark price matters more
// than the more theoretically complete academic version.
//
// ported from options-market-making-engine-rs (book-risk::inverse_option),
// same formulas and same test cross-checks, adapted to this crate's
// OptionType/npdf/ncdf and the phi-unified call/put pattern instead of a
// match per Greek. re-verified here after porting, not assumed identical
// just because the source formulas were already trusted.
//
// sanity check against a number that's actually out in the wild: a
// Deribit/Laevitas writeup on "true" inverse delta for an ATM-ish BTC
// option quotes a coin delta of ~0.0000063 per $1 move, same order of
// magnitude this formula gives for an ATM strike (roughly 0.5/F). a naive
// USD-style delta would be off by a factor of F from that.

use crate::math::{ncdf, npdf};
use crate::types::OptionType;

#[derive(Debug, Clone, Copy, Default)]
pub struct InverseGreeks {
    pub price_coin: f64,
    // d(price)/dF: hedge ratio against the tradable future/perp, not the index
    pub delta: f64,
    pub gamma: f64,
    // per 1.0 change in vol (100 vol points), divide by 100 for per-point
    pub vega: f64,
    pub theta: f64, // d(price)/d(calendar time), negative for a long option
    pub vanna: f64, // d(delta)/dvol, equivalently d(vega)/dF, both agree below
    pub volga: f64, // d(vega)/dvol
}

#[inline]
fn d1_d2(forward: f64, strike: f64, vol: f64, t: f64) -> (f64, f64) {
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
    (d1, d1 - vol * sqrt_t)
}

pub fn price_coin(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
    let phi = opt_type.sign();
    if t <= 0.0 || vol <= 0.0 {
        // at/past expiry: intrinsic in coin terms is (F-K)/F, not (F-K),
        // the /F is the whole point of this module existing
        return (phi * (forward - strike) / forward).max(0.0);
    }
    let (d1, d2) = d1_d2(forward, strike, vol, t);
    phi * ncdf(phi * d1) - phi * (strike / forward) * ncdf(phi * d2)
}

// theta: differentiate price_coin w.r.t. T holding F/K/vol fixed (standard
// Greek convention), the same n(d1)=(K/F)n(d2) identity that kills the
// ln(F/K) cross-terms in delta/gamma leaves d(price_coin)/dT =
// (K/F)*n(d2)*vol/(2*sqrt(T)), same for both call and put since call-put =
// 1-K/F doesn't depend on T at all, so their T-derivatives have to match,
// verified against that parity below, not just against FD.
//
// vanna derived two independent ways (had to agree before trusting it):
// d(vega)/dF from vega=(K/F)*n(d2)*sqrt(T), and d(delta)/dvol from
// delta=(K/F^2)*N(d2) using d(d2)/dvol=-d1/vol. both collapse to
// vanna=-(K/F^2)*n(d2)*d1/vol.
//
// volga=d(vega)/dvol works out to vega*d1*d2/vol, same structural form as
// the standard BS volga identity (Haug), a useful external check beyond
// the FD tests below.
pub fn greeks(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> InverseGreeks {
    let price_coin_now = price_coin(opt_type, forward, strike, vol, t);
    if t <= 0.0 || vol <= 0.0 {
        return InverseGreeks { price_coin: price_coin_now, ..Default::default() };
    }

    let phi = opt_type.sign();
    let (d1, d2) = d1_d2(forward, strike, vol, t);
    let sqrt_t = t.sqrt();
    let k_over_f2 = strike / (forward * forward);
    let k_over_f3 = k_over_f2 / forward;
    let nd2  = ncdf(phi * d2);
    let npd2 = npdf(d2); // npdf is even, npdf(phi*d2) == npdf(d2) regardless of phi

    let delta = phi * k_over_f2 * nd2;
    let gamma = k_over_f3 * (npd2 / (vol * sqrt_t) - phi * 2.0 * nd2);

    // same closed form for calls and puts, same parity argument as theta above
    let vega  = (strike / forward) * npd2 * sqrt_t;
    let theta = -(strike / forward) * npd2 * vol / (2.0 * sqrt_t);
    let vanna = -k_over_f2 * npd2 * d1 / vol;
    let volga = vega * d1 * d2 / vol;

    InverseGreeks { price_coin: price_coin_now, delta, gamma, vega, theta, vanna, volga }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_delta(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = forward * 1e-6;
        let up = price_coin(opt_type, forward + eps, strike, vol, t);
        let down = price_coin(opt_type, forward - eps, strike, vol, t);
        (up - down) / (2.0 * eps)
    }

    fn fd_gamma(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = forward * 1e-4; // gamma needs a wider bump than delta or noise dominates
        let up = price_coin(opt_type, forward + eps, strike, vol, t);
        let mid = price_coin(opt_type, forward, strike, vol, t);
        let down = price_coin(opt_type, forward - eps, strike, vol, t);
        (up - 2.0 * mid + down) / (eps * eps)
    }

    fn fd_vega(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = 1e-6;
        let up = price_coin(opt_type, forward, strike, vol + eps, t);
        let down = price_coin(opt_type, forward, strike, vol - eps, t);
        (up - down) / (2.0 * eps)
    }

    fn fd_theta(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = t * 1e-6;
        let up = price_coin(opt_type, forward, strike, vol, t + eps); // theta = -dV/dT
        let down = price_coin(opt_type, forward, strike, vol, t - eps);
        -(up - down) / (2.0 * eps)
    }

    // vanna two independent ways: d(delta)/dvol and d(vega)/dF, both have
    // to land on the same number if the closed form is right
    fn fd_vanna_via_delta(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = 1e-6;
        let up = greeks(opt_type, forward, strike, vol + eps, t).delta;
        let down = greeks(opt_type, forward, strike, vol - eps, t).delta;
        (up - down) / (2.0 * eps)
    }

    fn fd_vanna_via_vega(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = forward * 1e-6;
        let up = greeks(opt_type, forward + eps, strike, vol, t).vega;
        let down = greeks(opt_type, forward - eps, strike, vol, t).vega;
        (up - down) / (2.0 * eps)
    }

    fn fd_volga(opt_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = 1e-6;
        let up = greeks(opt_type, forward, strike, vol + eps, t).vega;
        let down = greeks(opt_type, forward, strike, vol - eps, t).vega;
        (up - down) / (2.0 * eps)
    }

    #[test]
    fn put_call_parity_holds_in_coin_terms() {
        let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
        let call = price_coin(OptionType::Call, forward, strike, vol, t);
        let put = price_coin(OptionType::Put, forward, strike, vol, t);
        let expected = 1.0 - strike / forward;
        assert!((call - put - expected).abs() < 1e-12, "call={call} put={put} expected={expected}");
    }

    #[test]
    fn analytic_delta_matches_finite_difference() {
        for (forward, strike) in [(65000.0, 65000.0), (65000.0, 70000.0), (65000.0, 55000.0)] {
            let (vol, t) = (0.65, 14.0 / 365.0);
            for opt_type in [OptionType::Call, OptionType::Put] {
                let analytic = greeks(opt_type, forward, strike, vol, t).delta;
                let fd = fd_delta(opt_type, forward, strike, vol, t);
                let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
                assert!(rel_diff < 1e-4, "{opt_type:?} F={forward} K={strike} analytic={analytic} fd={fd}");
            }
        }
    }

    #[test]
    fn analytic_gamma_matches_finite_difference() {
        for (forward, strike) in [(65000.0, 65000.0), (65000.0, 70000.0), (65000.0, 55000.0)] {
            let (vol, t) = (0.65, 14.0 / 365.0);
            for opt_type in [OptionType::Call, OptionType::Put] {
                let analytic = greeks(opt_type, forward, strike, vol, t).gamma;
                let fd = fd_gamma(opt_type, forward, strike, vol, t);
                let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
                assert!(rel_diff < 1e-2, "{opt_type:?} F={forward} K={strike} analytic={analytic} fd={fd}");
            }
        }
    }

    #[test]
    fn analytic_vega_matches_finite_difference() {
        for opt_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(opt_type, forward, strike, vol, t).vega;
            let fd = fd_vega(opt_type, forward, strike, vol, t);
            let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
            assert!(rel_diff < 1e-4, "{opt_type:?} analytic={analytic} fd={fd}");
        }
    }

    #[test]
    fn coin_delta_is_orders_of_magnitude_smaller_than_a_direct_option_delta() {
        // ATM coin delta should sit near 0.5/F, not near 0.5 the way a
        // direct/USD-settled option's delta would.
        let forward = 65000.0;
        let d = greeks(OptionType::Call, forward, forward, 0.6, 30.0 / 365.0).delta;
        assert!(d > 0.0 && d < 1.0 / forward, "coin delta {d} should be well under 1/F");
    }

    #[test]
    fn analytic_theta_matches_finite_difference() {
        for opt_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(opt_type, forward, strike, vol, t).theta;
            let fd = fd_theta(opt_type, forward, strike, vol, t);
            let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
            assert!(rel_diff < 1e-4, "{opt_type:?} analytic={analytic} fd={fd}");
        }
    }

    #[test]
    fn theta_is_negative_for_a_long_option() {
        let theta = greeks(OptionType::Call, 65000.0, 65000.0, 0.6, 14.0 / 365.0).theta;
        assert!(theta < 0.0, "long option should decay in value as T shrinks, got {theta}");
    }

    #[test]
    fn call_and_put_theta_are_equal() {
        // same reasoning as vega: C - P = 1 - K/F doesn't depend on T
        let (forward, strike, vol, t) = (65000.0, 62000.0, 0.7, 45.0 / 365.0);
        let call_theta = greeks(OptionType::Call, forward, strike, vol, t).theta;
        let put_theta = greeks(OptionType::Put, forward, strike, vol, t).theta;
        assert!((call_theta - put_theta).abs() < 1e-10);
    }

    #[test]
    fn vanna_matches_finite_difference_both_ways() {
        for opt_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(opt_type, forward, strike, vol, t).vanna;
            let via_delta = fd_vanna_via_delta(opt_type, forward, strike, vol, t);
            let via_vega = fd_vanna_via_vega(opt_type, forward, strike, vol, t);
            assert!((analytic - via_delta).abs() / via_delta.abs().max(1e-12) < 1e-4,
                "{opt_type:?} analytic={analytic} via_delta={via_delta}");
            assert!((analytic - via_vega).abs() / via_vega.abs().max(1e-12) < 1e-4,
                "{opt_type:?} analytic={analytic} via_vega={via_vega}");
        }
    }

    #[test]
    fn volga_matches_finite_difference() {
        for opt_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(opt_type, forward, strike, vol, t).volga;
            let fd = fd_volga(opt_type, forward, strike, vol, t);
            let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
            assert!(rel_diff < 1e-4, "{opt_type:?} analytic={analytic} fd={fd}");
        }
    }

    #[test]
    fn call_and_put_vanna_and_volga_are_equal() {
        // delta_call - delta_put = K/F^2 doesn't depend on vol, and vega is
        // already equal for both, so both cross-greeks inherit the same parity
        let (forward, strike, vol, t) = (65000.0, 62000.0, 0.7, 45.0 / 365.0);
        let call = greeks(OptionType::Call, forward, strike, vol, t);
        let put = greeks(OptionType::Put, forward, strike, vol, t);
        assert!((call.vanna - put.vanna).abs() < 1e-10);
        assert!((call.volga - put.volga).abs() < 1e-10);
    }

    #[test]
    fn vanna_and_volga_are_zero_at_or_past_expiry() {
        let g = greeks(OptionType::Call, 65000.0, 65000.0, 0.6, 0.0);
        assert_eq!(g.vanna, 0.0);
        assert_eq!(g.volga, 0.0);
    }

    #[test]
    fn call_and_put_vega_are_equal() {
        let (forward, strike, vol, t) = (65000.0, 62000.0, 0.7, 45.0 / 365.0);
        let call_vega = greeks(OptionType::Call, forward, strike, vol, t).vega;
        let put_vega = greeks(OptionType::Put, forward, strike, vol, t).vega;
        assert!((call_vega - put_vega).abs() < 1e-10);
    }

    #[test]
    fn at_expiry_price_is_intrinsic_in_coin_terms() {
        let forward = 65000.0;
        let itm_call = price_coin(OptionType::Call, forward, 60000.0, 0.6, 0.0);
        assert!((itm_call - (forward - 60000.0) / forward).abs() < 1e-12);
        let otm_call = price_coin(OptionType::Call, forward, 70000.0, 0.6, 0.0);
        assert_eq!(otm_call, 0.0);
    }
}
