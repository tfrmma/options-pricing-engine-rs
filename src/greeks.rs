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

    // vega bumps v0, which is in variance units, so we go through (vol+dv)^2 - vol^2
    let v_cur = params.v0().sqrt();
    let p_vup = params.with_v0((v_cur + dv).powi(2));
    let p_vdn = params.with_v0((v_cur - dv).max(1e-8).powi(2));
    let vega  = (p_vup.price(spot, strike, expiry, rate, div_yield, opt_type)
               - p_vdn.price(spot, strike, expiry, rate, div_yield, opt_type))
              / (2.0 * dv);

    let t_dn  = (expiry - dt).max(1e-6);
    let theta = (params.price(spot, strike, t_dn, rate, div_yield, opt_type) - price) / dt;

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
    let vanna = (delta_vup - delta_vdn) / (2.0 * dv);

    // volga = d(vega)/d(vol), i.e. second derivative of price w.r.t. vol
    let p_vup2 = params.with_v0((v_cur + 2.0*dv).powi(2));
    let p_vdn2 = params.with_v0((v_cur - 2.0*dv).max(1e-8).powi(2));
    let vega_up = (p_vup2.price(spot, strike, expiry, rate, div_yield, opt_type) - price) / (2.0 * dv);
    let vega_dn = (price - p_vdn2.price(spot, strike, expiry, rate, div_yield, opt_type)) / (2.0 * dv);
    let volga   = (vega_up - vega_dn) / (2.0 * dv);

    PricingResult { price, delta, gamma, vega, theta, rho, vanna, volga }
}
