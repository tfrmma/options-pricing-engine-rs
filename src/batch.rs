// batch pricing. call these instead of looping externally.
// BSM is embarrassingly parallel, rayon handles it.
// Heston is heavier per option (~15 GK evals of a complex CF),
// so the parallel speedup matters even for chains of 50.

use rayon::prelude::*;
use crate::types::{OptionContract, PricingResult, HestonParams, BatesParams};
use crate::bsm::{bsm_price_and_greeks, bsm_price};
use crate::heston::heston_price;
use crate::bates::bates_price;
use crate::iv::implied_vol;

pub fn batch_bsm(contracts: &[OptionContract]) -> Vec<PricingResult> {
    contracts.par_iter().map(|c| bsm_price_and_greeks(c)).collect()
}

// price-only — skip greeks if you just need marks
pub fn batch_bsm_price(contracts: &[OptionContract]) -> Vec<f64> {
    contracts.par_iter().map(|c| bsm_price(c)).collect()
}

pub fn batch_heston(contracts: &[OptionContract], params: &HestonParams) -> Vec<f64> {
    contracts.par_iter()
        .map(|c| heston_price(c.spot, c.strike, c.expiry, c.rate, c.div_yield, params, c.opt_type))
        .collect()
}

pub fn batch_bates(contracts: &[OptionContract], params: &BatesParams) -> Vec<f64> {
    contracts.par_iter()
        .map(|c| bates_price(c.spot, c.strike, c.expiry, c.rate, c.div_yield, params, c.opt_type))
        .collect()
}

// None = solver bailed. check your input surface if you're seeing a lot of these.
pub fn batch_implied_vol(contracts: &[OptionContract], market_prices: &[f64]) -> Vec<Option<f64>> {
    debug_assert_eq!(contracts.len(), market_prices.len());
    contracts.par_iter().zip(market_prices.par_iter())
        .map(|(c, &px)| implied_vol(&crate::types::IvProblem { contract: *c, market_price: px }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionType;
    use crate::bsm::bsm_price;

    fn chain(n: usize) -> Vec<OptionContract> {
        (0..n).map(|i| OptionContract {
            spot: 100.0, strike: 80.0 + i as f64 * 2.0,
            expiry: 0.5, rate: 0.03, div_yield: 0.0,
            vol: 0.20, opt_type: OptionType::Call,
        }).collect()
    }

    #[test]
    fn batch_matches_single() {
        let c   = chain(20);
        let out = batch_bsm_price(&c);
        for (contract, &px) in c.iter().zip(out.iter()) {
            assert!((px - bsm_price(contract)).abs() < 1e-12);
        }
    }

    #[test]
    fn iv_roundtrip() {
        let c      = chain(10);
        let prices: Vec<f64> = c.iter().map(|x| bsm_price(x)).collect();
        let ivs    = batch_implied_vol(&c, &prices);
        for (iv, contract) in ivs.iter().zip(c.iter()) {
            let iv = iv.expect("solver bailed");
            assert!((iv - contract.vol).abs() < 1e-6, "got {iv:.8}");
        }
    }
}
