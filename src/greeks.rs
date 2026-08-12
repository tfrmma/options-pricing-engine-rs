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

pub(crate) fn bump_and_reprice_greeks<P: BumpPriceable>(
    spot: f64, strike: f64, expiry: f64,
    rate: f64, div_yield: f64,
    params: &P, opt_type: OptionType,
) -> PricingResult {
    let price = params.price(spot, strike, expiry, rate, div_yield, opt_type);

    let ds = 0.01 * spot;
    let dv = 0.01;
    let dr = 1e-4;
    let dt = 1.0 / 365.0;

    let pu    = params.price(spot + ds, strike, expiry, rate, div_yield, opt_type);
    let pd    = params.price(spot - ds, strike, expiry, rate, div_yield, opt_type);
    let delta = (pu - pd) / (2.0 * ds);
    let gamma = (pu - 2.0*price + pd) / (ds * ds);

    // vega bumps v0, which is in variance units, so we go through (vol+dv)^2 - vol^2.
    // the denominator is the vol span actually used, not 2*dv: for v0 below dv the
    // lower bump clamps at 1e-8 and a 2*dv denominator understates vega by that ratio.
    // same rule the calibration Jacobian already follows (calibration.rs, `h = vu - vd`).
    // the step also has to shrink with v0 itself. a fixed 1 vol point is a 5% nudge at
    // vol=20% but swamps a vol=0.7% surface, where the curvature over that span makes
    // the difference quotient miss by ~20%. cap it at half the level, same idea as the
    // theta step below.
    let v_cur = params.v0().sqrt();
    let dv    = (dv as f64).min(v_cur * 0.5).max(1e-8);
    let v_up  = v_cur + dv;
    let v_dn  = (v_cur - dv).max(1e-8);
    let p_vup = params.with_v0(v_up.powi(2));
    let p_vdn = params.with_v0(v_dn.powi(2));
    let vega  = (p_vup.price(spot, strike, expiry, rate, div_yield, opt_type)
               - p_vdn.price(spot, strike, expiry, rate, div_yield, opt_type))
              / (v_up - v_dn);

    // theta: central difference with a step that shrinks near expiry.
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
    // capping the step at half the remaining life keeps t_lo strictly positive and the
    // error second-order, and dividing by the realized span handles whatever clamping
    // still happens.
    // a quarter of the remaining life, not half: the central difference error is O(h^2),
    // and at half-life the residual bias was still a flat 3.4% across 0.25..2 DTE.
    let h_t   = (dt as f64).min(expiry * 0.25).max(1e-9);
    let t_hi  = expiry + h_t;
    let t_lo  = (expiry - h_t).max(1e-9);
    let theta = -(params.price(spot, strike, t_hi, rate, div_yield, opt_type)
                - params.price(spot, strike, t_lo, rate, div_yield, opt_type))
              / (t_hi - t_lo);

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
    let vanna = (delta_vup - delta_vdn) / (v_up - v_dn);

    // volga = d(vega)/d(vol), i.e. second derivative of price w.r.t. vol.
    // divide by the realized spans here too, for the same reason as vega above.
    let v_up2 = v_cur + 2.0*dv;
    let v_dn2 = (v_cur - 2.0*dv).max(1e-8);
    let p_vup2 = params.with_v0(v_up2.powi(2));
    let p_vdn2 = params.with_v0(v_dn2.powi(2));
    let vega_up = (p_vup2.price(spot, strike, expiry, rate, div_yield, opt_type) - price)
                / (v_up2 - v_cur);
    let vega_dn = (price - p_vdn2.price(spot, strike, expiry, rate, div_yield, opt_type))
                / (v_cur - v_dn2);
    let volga   = (vega_up - vega_dn) / (0.5 * ((v_up2 - v_cur) + (v_cur - v_dn2)));

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}
