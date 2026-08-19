// BSM + Black-76. analytic greeks, no bumping.
// if you're finite-differencing vega on a vanilla you owe me a beer.

use crate::math::{ncdf, npdf};
use crate::types::{OptionContract, OptionType, PricingResult};

#[inline]
fn d_terms(s: f64, k: f64, t: f64, r: f64, q: f64, v: f64) -> (f64, f64) {
    let vt = v * t.sqrt();
    let d1 = ((s/k).ln() + (r - q + 0.5*v*v)*t) / vt;
    (d1, d1 - vt)
}

// compute everything in one pass, exp() calls are expensive, don't redo them
pub fn bsm_price_and_greeks(c: &OptionContract) -> PricingResult {
    let OptionContract { spot: s, strike: k, expiry: t, rate: r, div_yield: q, vol: v, opt_type } = *c;

    let (d1, d2) = d_terms(s, k, t, r, q, v);
    let phi  = opt_type.sign();
    let nd1  = ncdf(phi * d1);
    let nd2  = ncdf(phi * d2);
    let npd1 = npdf(d1);

    let eq   = (-q*t).exp();
    let er   = (-r*t).exp();
    let seq  = s * eq;
    let ker  = k * er;
    let sqt  = t.sqrt();

    let price = phi * (seq*nd1 - ker*nd2);
    let delta = phi * eq * nd1;
    let gamma = eq * npd1 / (s * v * sqt);
    let vega  = seq * npd1 * sqt;  // per unit, not per 1bp
    let theta = theta_calc(seq, ker, r, q, npd1, nd1, nd2, v*sqt, t, phi);
    let rho   = phi * k * t * er * nd2;
    let vanna = -eq * npd1 * d2 / v;
    let volga = vega * d1 * d2 / v;

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}

// internal decomposition of the theta formula into named pieces, all
// already-computed intermediates from the caller, not a public signature.
#[allow(clippy::too_many_arguments)]
fn theta_calc(
    seq: f64, ker: f64, r: f64, q: f64,
    npd1: f64, nd1: f64, nd2: f64,
    vsqt: f64, t: f64, phi: f64,
) -> f64 {
    // per year, caller divides by 365 if they want daily
    //
    // carry sign: for a call theta is
    //     -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
    // and the put flips both carry terms. that is +phi*(q*seq*nd1 - r*ker*nd2),
    // not -phi*(...). the old sign inverted the carry, which swapped call and put
    // theta everywhere except the vega-decay term.
    //
    // checked against a 1-day reprice finite difference:
    //   q=0.00  call -6.4140 vs FD -6.4169   put -1.6579 vs FD -1.6604
    //   q=0.03  call -4.4865 vs FD -4.4896   put -2.6417 vs FD -2.6446
    // the previous form returned -1.0908 and -5.8469 for the q=0 pair.
    -seq*npd1*vsqt/(2.0*t) + phi*(q*seq*nd1 - r*ker*nd2)
}

#[inline]
pub fn bsm_price(c: &OptionContract) -> f64 {
    let OptionContract { spot: s, strike: k, expiry: t, rate: r, div_yield: q, vol: v, opt_type } = *c;
    let (d1, d2) = d_terms(s, k, t, r, q, v);
    let phi = opt_type.sign();
    phi * (s*(-q*t).exp()*ncdf(phi*d1) - k*(-r*t).exp()*ncdf(phi*d2))
}

// Black-76: futures/forwards. same math, forward replaces spot, q=0 drops out.
pub fn black76_price_and_greeks(
    fwd: f64, strike: f64, expiry: f64,
    rate: f64, vol: f64, opt_type: OptionType,
) -> PricingResult {
    let vt   = vol * expiry.sqrt();
    let d1   = ((fwd/strike).ln() + 0.5*vol*vol*expiry) / vt;
    let d2   = d1 - vt;
    let phi  = opt_type.sign();
    let er   = (-rate*expiry).exp();
    let npd1 = npdf(d1);
    let nd1  = ncdf(phi*d1);
    let nd2  = ncdf(phi*d2);

    let price = phi * er * (fwd*nd1 - strike*nd2);
    let delta = phi * er * nd1;
    let gamma = er * npd1 / (fwd * vt);
    let vega  = fwd * er * npd1 * expiry.sqrt();
    // carry sign: same defect theta_calc had, see the comment there. for F held
    // fixed, theta = -e^{-rT} F phi(d1) sigma/(2 sqrt(T)) + r*price, and
    // phi*rate*(fwd*nd1 - strike*nd2) = r*price/er. the old minus doubled the
    // discount term instead of cancelling it (error = exactly 2*r*price).
    let theta = er * (-fwd*npd1*vol/(2.0*expiry.sqrt()) + phi*rate*(fwd*nd1 - strike*nd2));
    // fwd is exogenous here, not spot*e^{(r-q)T}, so the only r-dependence in price
    // is the discount factor out front. rho = d(price)/dr = -T*price. the old
    // phi*strike*expiry*er*nd2 was BSM's rho pasted in, it's missing the fwd*nd1 term.
    let rho   = -expiry * price;
    let vanna = -er * npd1 * d2 / vol;
    let volga = vega * d1 * d2 / vol;

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionType;

    fn atm_call() -> OptionContract {
        OptionContract {
            spot: 100.0, strike: 100.0, expiry: 1.0,
            rate: 0.05, div_yield: 0.0, vol: 0.2,
            opt_type: OptionType::Call,
        }
    }

    #[test]
    fn bsm_call_put_parity() {
        let c = atm_call();
        let call = bsm_price(&c);
        let put  = bsm_price(&OptionContract { opt_type: OptionType::Put, ..c });
        let rhs  = c.spot*(-c.div_yield*c.expiry).exp() - c.strike*(-c.rate*c.expiry).exp();
        assert!((call - put - rhs).abs() < 1e-10);
    }

    // theta has to match a 1-day reprice. a sign-only check does not catch a flipped
    // carry term, because the flip still leaves a plain call's theta negative.
    #[test]
    fn theta_matches_finite_difference() {
        let dt = 1.0 / 365.0;
        for q in [0.0, 0.03] {
            for opt in [OptionType::Call, OptionType::Put] {
                let c = OptionContract { div_yield: q, opt_type: opt, ..atm_call() };
                let analytic = bsm_price_and_greeks(&c).theta;
                let fd = (bsm_price(&OptionContract { expiry: c.expiry - dt, ..c })
                          - bsm_price(&c)) / dt;
                assert!((analytic - fd).abs() < 0.02,
                        "q={q} {opt:?}: analytic {analytic:.4} vs fd {fd:.4}");
            }
        }
    }

    // put-call parity for theta: d/dT[C - P] = q S e^{-qT} - r K e^{-rT}
    #[test]
    fn theta_put_call_parity() {
        let c  = OptionContract { div_yield: 0.03, ..atm_call() };
        let tc = bsm_price_and_greeks(&c).theta;
        let tp = bsm_price_and_greeks(&OptionContract { opt_type: OptionType::Put, ..c }).theta;
        let rhs = c.div_yield * c.spot * (-c.div_yield*c.expiry).exp()
                - c.rate * c.strike * (-c.rate*c.expiry).exp();
        assert!((tc - tp - rhs).abs() < 1e-10, "{tc:.6} - {tp:.6} != {rhs:.6}");
    }

    #[test]
    fn greeks_sign_check() {
        let r = bsm_price_and_greeks(&atm_call());
        assert!(r.delta > 0.0 && r.delta < 1.0);
        assert!(r.gamma > 0.0);
        assert!(r.vega  > 0.0);
    }

    #[test]
    fn black76_sanity() {
        let c   = atm_call();
        let fwd = c.spot * ((c.rate - c.div_yield)*c.expiry).exp();
        let b76 = black76_price_and_greeks(fwd, c.strike, c.expiry, c.rate, c.vol, c.opt_type);
        let bsm = bsm_price_and_greeks(&c);
        assert!((b76.price - bsm.price).abs() < 1e-8);
    }

    // rho used to be BSM's formula copy-pasted in, which is wrong for Black-76
    // since fwd doesn't carry its own r-dependence here. checked against FD.
    fn black76_price(fwd: f64, k: f64, t: f64, r: f64, v: f64, ot: OptionType) -> f64 {
        black76_price_and_greeks(fwd, k, t, r, v, ot).price
    }

    #[test]
    fn black76_rho_matches_fd() {
        let (fwd, k, t, v) = (100.0, 95.0, 0.75, 0.22);
        for ot in [OptionType::Call, OptionType::Put] {
            let r = 0.04;
            let dr = 1e-5;
            let analytic = black76_price_and_greeks(fwd, k, t, r, v, ot).rho;
            let fd = (black76_price(fwd, k, t, r+dr, v, ot) - black76_price(fwd, k, t, r-dr, v, ot)) / (2.0*dr);
            let rel_err = (analytic - fd).abs() / fd.abs().max(1e-8);
            assert!(rel_err < 1e-5, "{ot:?}: analytic={analytic:.6} fd={fd:.6}");
        }
    }
}
