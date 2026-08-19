// shared bump-and-reprice greeks for models where we don't have (or haven't
// wired up) an analytic/AD path. heston.rs and bates.rs used to each carry
// their own copy of this, same bumps, same 14 pricing calls, same bugs
// waiting to diverge the moment someone tweaks one and forgets the other.
//
// bump sizes: dS = 1% spot, dv = 1 vol point, dr = 1bp, dt = 1 calendar day.
// vanna/volga via double bump, 4 extra calls each, worth it for second-order accuracy.

use crate::types::{OptionType, PricingResult};

// anything with a variance-driving param (v0) that can price itself.
// heston_price/bates_price implement this trivially, see heston.rs/bates.rs.
pub(crate) trait BumpPriceable: Copy {
    fn price(&self, spot: f64, strike: f64, expiry: f64, rate: f64, div_yield: f64, opt_type: OptionType) -> f64;
    fn v0(&self) -> f64;
    fn with_v0(&self, v0: f64) -> Self;
}

// vol bump step, shared with the AD entry points in ad.rs (they bump vol for
// their own vanna/volga), one implementation so the two paths can't drift
// apart again. adaptive: 1 vol point, capped at half the current level (a
// fixed point swamps a sub-1% surface), floored at 1e-4 so a zero or
// near-zero v0 still yields a usable span instead of the 0/0 the old 1e-8
// floor produced (dv = 1e-8 made v_up == v_dn == 1e-8 at v0 = 0).
pub(crate) fn vol_step(v_cur: f64) -> f64 {
    0.01_f64.min(v_cur * 0.5).max(1e-4)
}

// symmetric bump levels around v_cur, clamped at zero from below (variance
// can't go negative; v0 = 0 prices fine through the CF). returns
// (up, dn, span) where span is the REALIZED distance: callers divide by it,
// not by 2*h, because at small v_cur the lower level clamps and the span is
// h, not 2h. span >= vol_step > 0 always, so no denominator can vanish.
pub(crate) fn vol_bump_levels(v_cur: f64, mult: f64) -> (f64, f64, f64) {
    let h  = mult * vol_step(v_cur);
    let up = v_cur + h;
    let dn = (v_cur - h).max(0.0);
    (up, dn, up - dn)
}

// central-difference theta with a step that shrinks near expiry, shared with
// the AD entry points in ad.rs for the same can't-diverge reason as above.
//
// two things were wrong with the one-sided `(P(T-dt) - P(T)) / dt`:
//
// 1. a forward difference has O(dt) truncation error, and dt is one calendar day.
//    at T=1 day that step *is* the entire remaining life, where P is at its most
//    nonlinear, so the error explodes: measured against a central difference on
//    Heston ATM the ratio ran 1.007 at 30 DTE, 1.096 at 3 DTE, and 1.932 at 1 DTE.
//    this was the dominant error, not the clamp below.
// 2. `t_dn` was clamped to 1e-6 while the denominator stayed `dt`, so under one day
//    the step silently shrank but the divisor did not: at 0.25 DTE the real step is
//    a quarter of dt, scaling theta to 0.477 of its true value (the two errors push
//    in opposite directions, which is why the ratio crosses 1 near 0.5 DTE).
//
// the step is a quarter of the remaining life, not half: the central difference
// error is O(h^2), and at half-life the residual bias was still a flat 3.4%
// across 0.25..2 DTE. h <= expiry/4 also keeps t_lo strictly positive.
//
// expiry at (numerically) zero has no life left to difference over — the
// span degenerates and the quotient is 0/0. an expired option has no time
// value left to decay: theta is 0, not NaN.
pub(crate) fn central_theta<F: Fn(f64) -> f64>(expiry: f64, price_at: F) -> f64 {
    if expiry <= 1e-8 { return 0.0; }
    let dt   = 1.0_f64 / 365.0;
    let h_t  = dt.min(expiry * 0.25).max(1e-9);
    let t_hi = expiry + h_t;
    let t_lo = expiry - h_t;
    -(price_at(t_hi) - price_at(t_lo)) / (t_hi - t_lo)
}

pub(crate) fn bump_and_reprice_greeks<P: BumpPriceable>(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &P, opt_type: OptionType,
) -> PricingResult {
    let price = params.price(spot, strike, expiry, rate, div_yield, opt_type);

    let ds = 0.01 * spot;
    let dr = 1e-4;

    let pu    = params.price(spot + ds, strike, expiry, rate, div_yield, opt_type);
    let pd    = params.price(spot - ds, strike, expiry, rate, div_yield, opt_type);
    let delta = (pu - pd) / (2.0 * ds);
    let gamma = (pu - 2.0*price + pd) / (ds * ds);

    // vega bumps v0, which is in variance units, so we go through (vol+dv)^2 - vol^2.
    // step rule and realized-span division live in vol_bump_levels above, same rule
    // the calibration Jacobian already follows (calibration.rs, `h = vu - vd`).
    let v_cur = params.v0().sqrt();
    let (v_up, v_dn, v_span) = vol_bump_levels(v_cur, 1.0);
    let p_vup = params.with_v0(v_up.powi(2));
    let p_vdn = params.with_v0(v_dn.powi(2));
    let vega  = (p_vup.price(spot, strike, expiry, rate, div_yield, opt_type)
               - p_vdn.price(spot, strike, expiry, rate, div_yield, opt_type))
              / v_span;

    // theta: central difference with a shrinking step, see central_theta.
    let theta = central_theta(expiry, |t| params.price(spot, strike, t, rate, div_yield, opt_type));

    let rho = (params.price(spot, strike, expiry, rate + dr, div_yield, opt_type)
             - params.price(spot, strike, expiry, rate - dr, div_yield, opt_type))
            / (2.0 * dr);

    // vanna = d(delta)/d(vol): cross bump, delta at v+dv minus delta at v-dv
    let delta_vup = {
        let pu = p_vup.price(spot + ds, strike, expiry, rate, div_yield, opt_type);
        let pd = p_vup.price(spot - ds, strike, expiry, rate, div_yield, opt_type);
        (pu - pd) / (2.0 * ds)
    };
    let delta_vdn = {
        let pu = p_vdn.price(spot + ds, strike, expiry, rate, div_yield, opt_type);
        let pd = p_vdn.price(spot - ds, strike, expiry, rate, div_yield, opt_type);
        (pu - pd) / (2.0 * ds)
    };
    let vanna = (delta_vup - delta_vdn) / v_span;

    // volga = d(vega)/d(vol), i.e. second derivative of price w.r.t. vol.
    // divide by the realized spans here too, for the same reason as vega above.
    let (v_up2, v_dn2, _) = vol_bump_levels(v_cur, 2.0);
    let p_vup2  = params.with_v0(v_up2.powi(2));
    let span_up = v_up2 - v_cur; // = 2*step, always > 0
    let span_dn = v_cur - v_dn2; // 0 when v_cur sits at (or under) the zero clamp
    let vega_up = (p_vup2.price(spot, strike, expiry, rate, div_yield, opt_type) - price)
                / span_up;
    // no room below the current level means no curvature information: report
    // volga 0 rather than dividing by the degenerate lower span (0/0 -> NaN,
    // or a sign-flipped quotient when v_cur is barely above the clamp).
    let volga = if span_dn > 0.0 {
        let p_vdn2  = params.with_v0(v_dn2.powi(2));
        let vega_dn = (price - p_vdn2.price(spot, strike, expiry, rate, div_yield, opt_type))
                    / span_dn;
        (vega_up - vega_dn) / (0.5 * (span_up + span_dn))
    } else {
        0.0
    };

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}
