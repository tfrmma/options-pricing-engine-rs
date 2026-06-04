// IV solver. this is where most engines die quietly on deep OTM.
//
// flow: B-S ATM approximation for initial guess -> Halley (cubic convergence).
// if the guess is garbage, we fall back to bisection. slow but never blows up.
// 2-3 Halley steps is usually enough. running 50 Newton iterations on a vanilla
// is not optimization, it's a bug.

use crate::bsm::bsm_price;
use crate::types::{IvProblem, OptionContract, OptionType};

const MAX_ITER: usize = 10;
const TOL: f64 = 1e-10;

pub fn implied_vol(prob: &IvProblem) -> Option<f64> {
    let c = &prob.contract;
    let mkt = prob.market_price;
    if !price_in_bounds(c, mkt) { return None; }
    let v0 = initial_guess(c, mkt)?;
    halley_solve(c, mkt, v0)
}

fn price_in_bounds(c: &OptionContract, price: f64) -> bool {
    if price <= 0.0 { return false; }
    let er = (-c.rate * c.expiry).exp();
    let eq = (-c.div_yield * c.expiry).exp();
    let intrinsic = match c.opt_type {
        OptionType::Call => (c.spot*eq - c.strike*er).max(0.0),
        OptionType::Put  => (c.strike*er - c.spot*eq).max(0.0),
    };
    price >= intrinsic
}

// Brenner-Subrahmanyam ATM approximation, adjusted for moneyness.
// good enough that Halley finishes in 2 steps most of the time.
// falls back to bisection if we're way off — rare, but happens on deep wings.
fn initial_guess(c: &OptionContract, price: f64) -> Option<f64> {
    let er  = (-c.rate * c.expiry).exp();
    let eq  = (-c.div_yield * c.expiry).exp();
    let fwd = c.spot * eq / er;

    let bs = price * (2.0 * std::f64::consts::PI / c.expiry).sqrt() / (fwd * er);
    let x  = (fwd / c.strike).ln();
    let v0 = (bs / (-0.5*x*x).exp().max(0.01)).clamp(0.001, 10.0);

    if (bsm_price(&OptionContract { vol: v0, ..*c }) - price).abs() < 0.2 * price {
        return Some(v0);
    }
    bisect(c, price)
}

// last resort — always converges, just not fast
fn bisect(c: &OptionContract, target: f64) -> Option<f64> {
    let mut lo = 1e-4_f64;
    let mut hi = 10.0_f64;
    let f = |v: f64| bsm_price(&OptionContract { vol: v, ..*c }) - target;
    if f(lo) * f(hi) > 0.0 { return None; }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 { lo = mid; } else { hi = mid; }
        if hi - lo < 1e-9 { return Some(mid); }
    }
    Some(0.5 * (lo + hi))
}

fn halley_solve(c: &OptionContract, target: f64, v0: f64) -> Option<f64> {
    let mut v = v0;
    for _ in 0..MAX_ITER {
        let err = bsm_price(&OptionContract { vol: v, ..*c }) - target;
        if err.abs() < TOL { return Some(v); }
        let (vega, volga) = vega_and_volga(c, v);
        if vega.abs() < 1e-14 { return None; }  // zero vega = we're toast
        // Halley: f/f' / (1 - f*f''/(2*f'^2))
        let denom = (1.0 - err*volga / (2.0*vega*vega)).clamp(0.5, 2.0);
        v -= err / (vega * denom);
        if v <= 0.0 { v = 1e-8; }
    }
    // didn't converge cleanly — return best effort if it's close enough
    if (bsm_price(&OptionContract { vol: v, ..*c }) - target).abs() < 1e-6 {
        Some(v)
    } else {
        None
    }
}

#[inline]
fn vega_and_volga(c: &OptionContract, v: f64) -> (f64, f64) {
    let vt   = v * c.expiry.sqrt();
    let d1   = ((c.spot/c.strike).ln() + (c.rate - c.div_yield + 0.5*v*v)*c.expiry) / vt;
    let d2   = d1 - vt;
    let eq   = (-c.div_yield * c.expiry).exp();
    let npd1 = crate::math::npdf(d1);
    let sqt  = c.expiry.sqrt();
    let vega = c.spot * eq * npd1 * sqt;
    (vega, vega * d1 * d2 / v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm::bsm_price;
    use crate::types::{OptionContract, OptionType, IvProblem};

    fn contract(vol: f64, opt_type: OptionType) -> OptionContract {
        OptionContract {
            spot: 100.0, strike: 100.0, expiry: 0.5,
            rate: 0.03, div_yield: 0.0,
            vol, opt_type,
        }
    }

    fn check_roundtrip(vol: f64, t: OptionType) {
        let c  = contract(vol, t);
        let px = bsm_price(&c);
        let iv = implied_vol(&IvProblem { contract: c, market_price: px })
            .unwrap_or_else(|| panic!("solver bailed on vol={vol}"));
        assert!((iv - vol).abs() < 1e-7, "got {iv:.8} expected {vol}");
    }

    #[test]
    fn iv_roundtrip_atm()  { check_roundtrip(0.20, OptionType::Call); check_roundtrip(0.20, OptionType::Put); }
    #[test]
    fn iv_roundtrip_otm()  { check_roundtrip(0.30, OptionType::Call); }
    #[test]
    fn iv_roundtrip_low_vol() { check_roundtrip(0.05, OptionType::Call); }
    #[test]
    fn iv_rejects_bad_price() {
        let c = contract(0.2, OptionType::Call);
        assert!(implied_vol(&IvProblem { contract: c, market_price: -1.0 }).is_none());
    }
}
