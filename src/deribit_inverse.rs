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
use crate::mc::McResult;
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

// implied vol from a coin-denominated market price, same Halley-with-
// bisection-fallback architecture as iv.rs, forward-based instead of
// spot/rate/div_yield since that's what a coin-settled quote actually
// gives you (Deribit quotes premium in coin against the mark price of
// the corresponding future, not against spot).
pub struct CoinIvProblem {
    pub forward: f64,
    pub strike: f64,
    pub expiry: f64,
    pub opt_type: OptionType,
    pub market_price_coin: f64,
}

const MAX_ITER: usize = 10;
const TOL: f64 = 1e-10;

pub fn implied_vol_coin(prob: &CoinIvProblem) -> Option<f64> {
    if !price_in_bounds(prob) { return None; }
    let v0 = initial_guess(prob)?;
    halley_solve(prob, v0)
}

fn price_in_bounds(p: &CoinIvProblem) -> bool {
    if p.market_price_coin <= 0.0 { return false; }
    let phi = p.opt_type.sign();
    let intrinsic = (phi * (p.forward - p.strike) / p.forward).max(0.0);
    if p.market_price_coin < intrinsic { return false; }
    // a coin call is worth strictly less than one coin: the payoff itself,
    // (S_T-K)^+/S_T = max(0, 1-K/S_T), is bounded above by 1 pointwise for
    // any K>0, so its expectation is too. checked numerically across
    // 200k random (F,K,vol,T) draws before relying on it, max observed
    // 1.0 (the asymptotic limit, never exceeded).
    //
    // puts get no such bound. (K-S_T)^+/S_T = max(0, K/S_T-1) is genuinely
    // unbounded as S_T->0, and so is its price: a deep ITM put (K/F=100)
    // priced out to ~99 coin in the same sweep, scaling with K/F, not
    // capped at 1. real asymmetry inverse options have that linear ones
    // don't, not a bug to paper over with a symmetric bound.
    if p.opt_type == OptionType::Call && p.market_price_coin >= 1.0 { return false; }
    true
}

// Brenner-Subrahmanyam ATM approximation, coin version. near ATM (F~K),
// price_coin(Call) = N(d1) - (K/F)N(d2) ~ N(d1)-N(d2) ~ npdf(0)*(d1-d2) =
// vol*sqrt(T)/sqrt(2pi), since d1-d2 = vol*sqrt(T) always and K/F~1. no
// fwd/discount-factor rescale needed the way iv.rs's USD version has,
// price_coin is already unitless (coin per coin, not USD per share), the
// ATM approximation is just its inverse directly. same moneyness
// adjustment and same 0.2*price fallback-to-bisection tolerance as iv.rs.
fn initial_guess(p: &CoinIvProblem) -> Option<f64> {
    let bs = p.market_price_coin * (2.0 * std::f64::consts::PI / p.expiry).sqrt();
    let x  = (p.forward / p.strike).ln();
    let v0 = (bs / (-0.5 * x * x).exp().max(0.01)).clamp(0.001, 10.0);

    if (price_coin(p.opt_type, p.forward, p.strike, v0, p.expiry) - p.market_price_coin).abs() < 0.2 * p.market_price_coin {
        return Some(v0);
    }
    bisect(p)
}

fn bisect(p: &CoinIvProblem) -> Option<f64> {
    let mut lo = 1e-4_f64;
    let mut hi = 10.0_f64;
    let f = |v: f64| price_coin(p.opt_type, p.forward, p.strike, v, p.expiry) - p.market_price_coin;
    if f(lo) * f(hi) > 0.0 { return None; }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 { lo = mid; } else { hi = mid; }
        if hi - lo < 1e-9 { return Some(mid); }
    }
    Some(0.5 * (lo + hi))
}

fn halley_solve(p: &CoinIvProblem, v0: f64) -> Option<f64> {
    let mut v = v0;
    for _ in 0..MAX_ITER {
        let err = price_coin(p.opt_type, p.forward, p.strike, v, p.expiry) - p.market_price_coin;
        if err.abs() < TOL { return Some(v); }
        let g = greeks(p.opt_type, p.forward, p.strike, v, p.expiry); // vega/volga, reuses the already-tested closed form instead of a second copy of it
        if g.vega.abs() < 1e-14 { return None; }
        let denom = (1.0 - err * g.volga / (2.0 * g.vega * g.vega)).clamp(0.5, 2.0);
        v -= err / (g.vega * denom);
        if v <= 0.0 { v = 1e-8; }
    }
    if (price_coin(p.opt_type, p.forward, p.strike, v, p.expiry) - p.market_price_coin).abs() < 1e-6 {
        Some(v)
    } else {
        None
    }
}

// converts an existing USD-settled MC price (mc.rs, any of Heston/Bates/
// rBergomi, any of European/AsianArithmetic/UpAndOut) into the
// coin-settled equivalent. NOT price/spot_T (dividing by the terminal
// price, path by path), that was the first version of this and it's
// wrong: it silently priced under the wrong probability measure and
// missed the Deribit closed-form cross-check by a z-score in the 40s-60s,
// caught by that cross-check, not by inspection.
//
// derivation: a coin-settled payoff is h(path)/S_T for some USD payoff
// h(path) (h = (S_T-K)^+ for a European call, the running average minus
// strike for an Asian, etc). the correct price is the expectation under
// the SHARE numeraire measure Q^S (numeraire = the asset itself, not the
// money-market account), not the standard risk-neutral measure Q, that's
// what "coin-settled" actually means, the premium and the payoff are
// denominated in a numeraire that moves with the asset. the Radon-Nikodym
// derivative for switching from the money-market numeraire to the share
// numeraire is dQ^S/dQ = S_T*e^{-(r-q)T}/S_0 (a completely general
// change-of-numeraire result, only requires S_t*e^{-(r-q)t} to be a
// Q-martingale, true by construction for every model in mc.rs, not
// specific to flat vol or to European payoffs):
//
//   E^{Q^S}[h(path)/S_T] = E^Q[h(path)/S_T * S_T*e^{-(r-q)T}/S_0]
//                        = (e^{-(r-q)T}/S_0) * E^Q[h(path)]      <- S_T cancels
//
// so the coin price is just (e^{q*T}/S_0) times the ALREADY-DISCOUNTED
// standard USD MC price, no new simulation, no new payoff logic, no
// change of drift needed. verified against deribit_inverse::price_coin
// directly (q=0 case: BSM call / S_0 matched the closed form to 6 decimal
// places, see the crate root test), and the S_T-cancellation step above
// never used flat vol, so the same rescale is correct for Heston/Bates/
// rBergomi and for path-dependent payoffs (Asian, barrier) too, not just
// vanilla European.
//
// std_error scales by the same constant: it's a linear rescale of the
// whole per-path estimator, correlation structure across paths is
// unaffected.
pub fn mc_result_to_coin(usd_result: McResult, spot: f64, div_yield: f64, expiry: f64) -> McResult {
    let scale = (div_yield * expiry).exp() / spot;
    McResult { price: usd_result.price * scale, std_error: usd_result.std_error * scale }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coin_iv_roundtrip(forward: f64, strike: f64, vol: f64, t: f64, opt_type: OptionType) {
        let price = price_coin(opt_type, forward, strike, vol, t);
        let iv = implied_vol_coin(&CoinIvProblem { forward, strike, expiry: t, opt_type, market_price_coin: price })
            .unwrap_or_else(|| panic!("solver bailed on vol={vol}"));
        assert!((iv - vol).abs() < 1e-7, "got {iv:.8} expected {vol}");
    }

    #[test]
    fn coin_iv_roundtrip_atm() {
        coin_iv_roundtrip(65000.0, 65000.0, 0.6, 30.0 / 365.0, OptionType::Call);
        coin_iv_roundtrip(65000.0, 65000.0, 0.6, 30.0 / 365.0, OptionType::Put);
    }

    #[test]
    fn coin_iv_roundtrip_otm_and_itm() {
        coin_iv_roundtrip(65000.0, 75000.0, 0.7, 14.0 / 365.0, OptionType::Call); // OTM call
        coin_iv_roundtrip(65000.0, 55000.0, 0.7, 14.0 / 365.0, OptionType::Call); // ITM call
        coin_iv_roundtrip(65000.0, 55000.0, 0.9, 90.0 / 365.0, OptionType::Put);  // deep ITM put (K << F, small side)
    }

    #[test]
    fn coin_iv_roundtrip_deep_itm_put() {
        // K/F=3, price_coin ~2.0, comfortably past the call's <1 bound,
        // this is exactly the case with no upper bound. checked vega isn't
        // near zero before picking these numbers, a first attempt at K/F=60
        // put the option so deep ITM the price is flat at intrinsic across
        // vol=0.05 to vol=3.0 (T=30d), genuinely ill-posed for inversion,
        // not a solver bug, replaced rather than forced to converge on a
        // number the price barely depends on.
        coin_iv_roundtrip(30000.0, 90000.0, 0.8, 180.0 / 365.0, OptionType::Put);
    }

    #[test]
    fn coin_iv_roundtrip_low_vol() {
        coin_iv_roundtrip(65000.0, 65000.0, 0.15, 7.0 / 365.0, OptionType::Call);
    }

    #[test]
    fn coin_iv_rejects_bad_price() {
        let base = CoinIvProblem { forward: 65000.0, strike: 65000.0, expiry: 30.0 / 365.0, opt_type: OptionType::Call, market_price_coin: -1.0 };
        assert!(implied_vol_coin(&base).is_none());
    }

    #[test]
    fn coin_iv_rejects_call_priced_at_or_above_one() {
        // a coin call can never be worth a full coin (see price_in_bounds),
        // 1.0 exactly and anything above it should both bounce.
        for bad_price in [1.0, 1.5, 100.0] {
            let p = CoinIvProblem { forward: 65000.0, strike: 65000.0, expiry: 30.0 / 365.0, opt_type: OptionType::Call, market_price_coin: bad_price };
            assert!(implied_vol_coin(&p).is_none(), "should reject call price {bad_price}");
        }
    }

    #[test]
    fn coin_iv_accepts_put_priced_above_one() {
        // puts have no such bound, this must NOT be rejected just because
        // it's over 1.0, only the intrinsic-value floor applies to them.
        coin_iv_roundtrip(30000.0, 90000.0, 0.8, 180.0 / 365.0, OptionType::Put);
    }

    #[test]
    fn coin_iv_rejects_price_below_intrinsic() {
        // forward=65000, strike=60000, call intrinsic = (65000-60000)/65000 ~ 0.0769
        let p = CoinIvProblem { forward: 65000.0, strike: 60000.0, expiry: 30.0 / 365.0, opt_type: OptionType::Call, market_price_coin: 0.01 };
        assert!(implied_vol_coin(&p).is_none());
    }

    // K/F=60, T=30d: price_coin is flat at intrinsic (59.0) across
    // vol=0.05..3.0, vega effectively zero over that whole range, this is
    // genuinely ill-posed for inversion, not a solver bug (found while
    // picking test parameters above, a first attempt used exactly this
    // combination and the solver landed on vol=2.396 instead of the 0.8
    // that generated the price). doesn't assert which vol comes back,
    // only that whatever does reprices consistently, that's the honest
    // thing to check in a region where many different vols are
    // observationally indistinguishable within tolerance.
    #[test]
    fn coin_iv_near_zero_vega_returns_a_self_consistent_root_not_necessarily_the_generating_one() {
        let (forward, strike, t) = (1000.0, 60000.0, 30.0 / 365.0);
        let price = price_coin(OptionType::Put, forward, strike, 0.8, t);
        if let Some(v) = implied_vol_coin(&CoinIvProblem { forward, strike, expiry: t, opt_type: OptionType::Put, market_price_coin: price }) {
            let repriced = price_coin(OptionType::Put, forward, strike, v, t);
            assert!((repriced - price).abs() < 1e-6, "solver returned vol={v} which reprices to {repriced}, not {price}");
        }
        // None is also an acceptable outcome here, bailing on an
        // ill-conditioned problem beats a confident wrong answer.
    }

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

    // mc_result_to_coin against rBergomi with near-zero vol-of-vol, which
    // collapses to flat BSM (same trick as mc.rs's own
    // rbergomi_matches_black_scholes_when_vol_of_vol_is_tiny). two
    // independently-built pricers, closed-form here, hybrid-scheme MC
    // there, agreeing is a real cross-check on the rescale, not a
    // self-consistency check on either one alone. this is the test that
    // caught the first (wrong) implementation, z was in the 40s-60s, not a
    // rounding-error mismatch.
    #[test]
    fn rbergomi_rescale_matches_closed_form_when_vol_of_vol_is_tiny() {
        use crate::mc::{mc_rough_bergomi, McConfig, Payoff, VarianceScheme};
        use crate::types::{RoughBergomiParams, ForwardVarianceCurve};

        let params = RoughBergomiParams { eta: 0.001, rho: -0.9, hurst: 0.07 };
        let xi0 = 0.04;
        let curve = ForwardVarianceCurve::new(vec![2.0], vec![xi0]);
        let (s, k, t, r, q): (f64, f64, f64, f64, f64) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let forward = s * ((r - q) * t).exp();

        let closed = price_coin(OptionType::Call, forward, k, xi0.sqrt(), t);

        let cfg = McConfig { n_paths: 100_000, n_steps: 64, seed: 3, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };
        let usd = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::European { strike: k, opt_type: OptionType::Call }, &cfg);
        let coin = mc_result_to_coin(usd, s, q, t);

        let z = (coin.price - closed).abs() / coin.std_error;
        assert!(z < 4.0, "closed={closed:.6} coin={:.6} se={:.6} z={z:.2}", coin.price, coin.std_error);
    }

    // same cross-check, Heston MC instead of rBergomi (near-zero vol-of-vol
    // sigma this time), confirms the rescale is correct for a genuinely
    // different stochastic-vol model, not just one whose implementation
    // happens to agree.
    #[test]
    fn heston_rescale_matches_closed_form_when_vol_of_vol_is_tiny() {
        use crate::mc::{mc_heston, McConfig, Payoff, VarianceScheme};
        use crate::types::HestonParams;

        let heston_params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.001, rho: -0.5 };
        let (s, k, t, r, q): (f64, f64, f64, f64, f64) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let forward = s * ((r - q) * t).exp();

        let closed = price_coin(OptionType::Call, forward, k, 0.04_f64.sqrt(), t);

        let cfg = McConfig { n_paths: 100_000, n_steps: 100, seed: 7, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };
        let usd = mc_heston(s, t, r, q, &heston_params,
            Payoff::European { strike: k, opt_type: OptionType::Call }, &cfg);
        let coin = mc_result_to_coin(usd, s, q, t);

        let z = (coin.price - closed).abs() / coin.std_error;
        assert!(z < 4.0, "closed={closed:.6} coin={:.6} se={:.6} z={z:.2}", coin.price, coin.std_error);
    }
}
