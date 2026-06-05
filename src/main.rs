// Quick smoke test + timing. Not a unit test — just something you can
// run to sanity-check the whole stack after a refactor.
// Run with: cargo run --release

//! @file main.rs
//! @author Taha - Algorithmic Trader
//! @brief Institutional-grade Options Pricing Engine.
//! 
//! @note This is a public structural showcase. For full production-grade 
//!       deployment, architecture consulting, or recruitment inquiries:
//!       Contact: email: fadilrezokt@gmail.com / linkedin.com/in/tahaotc

use options_pricing_engine::*;
use std::time::Instant;

fn main() {
    println!("=== options-pricing-engine-rs ===\n");

    bsm_demo();
    black76_demo();
    iv_demo();
    heston_demo();
    bates_demo();
    local_vol_demo();
    batch_bench();
}

fn bsm_demo() {
    let contract = OptionContract {
        spot: 100.0, strike: 100.0, expiry: 1.0,
        rate: 0.05, div_yield: 0.02, vol: 0.20,
        opt_type: OptionType::Call,
    };

    let r = bsm_price_and_greeks(&contract);
    println!("[BSM] ATM Call  price={:.4}  delta={:.4}  gamma={:.4}  vega={:.4}  theta={:.6}",
        r.price, r.delta, r.gamma, r.vega, r.theta);

    let put = bsm_price_and_greeks(&OptionContract { opt_type: OptionType::Put, ..contract });
    println!("[BSM] ATM Put   price={:.4}  delta={:.4}", put.price, put.delta);

    let er = (-contract.rate * contract.expiry).exp();
    let eq = (-contract.div_yield * contract.expiry).exp();
    let parity_err = (r.price - put.price - contract.spot * eq + contract.strike * er).abs();
    println!("[BSM] Put-call parity err = {:.2e}", parity_err);
    println!();
}

fn black76_demo() {
    let fwd = 100.0_f64 * (0.05_f64 * 1.0_f64).exp();
    let r = black76_price_and_greeks(fwd, 100.0, 1.0, 0.05, 0.20, OptionType::Call);
    println!("[B76] Fwd={:.2}  Call price={:.4}  delta={:.4}", fwd, r.price, r.delta);
    println!();
}

fn iv_demo() {
    let contract = OptionContract {
        spot: 100.0, strike: 105.0, expiry: 0.5,
        rate: 0.03, div_yield: 0.0, vol: 0.25,
        opt_type: OptionType::Call,
    };

    let market_px = bsm_price(&contract);
    let iv = implied_vol(&IvProblem { contract, market_price: market_px });

    println!("[IV]  True vol=0.2500  Solved={:.6}  Market px={:.4}",
        iv.unwrap_or(f64::NAN), market_px);
    println!();
}

fn heston_demo() {
    let params = HestonParams {
        v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7,
    };
    println!("[Heston] Feller ok: {}", params.feller_ok());

    let call = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);
    let put  = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Put);
    let er = (-0.05_f64).exp();
    let parity = (call - put - 100.0 + 100.0 * er).abs();

    println!("[Heston] Call={:.4}  Put={:.4}  PCP_err={:.2e}", call, put, parity);
    println!();
}

fn bates_demo() {
    let params = BatesParams {
        heston: HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 },
        lambda: 0.5,
        mu_j: -0.10,
        sigma_j: 0.15,
    };

    let call = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);
    let put  = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Put);
    let er = (-0.05_f64).exp();
    let parity = (call - put - 100.0 + 100.0 * er).abs();

    println!("[Bates] Call={:.4}  Put={:.4}  PCP_err={:.2e}", call, put, parity);
    println!();
}

fn local_vol_demo() {
    let strikes  = vec![80.0, 90.0, 100.0, 110.0, 120.0];
    let expiries = vec![0.25, 0.5, 1.0, 2.0];
    let ivs: Vec<f64> = strikes.iter().flat_map(|&k| {
        expiries.iter().map(move |&_t| {
            let moneyness = (k / 100.0 - 1.0_f64).abs();
            0.18 + 0.08 * moneyness
        })
    }).collect();

    let surf = LocalVolSurface::new(strikes, expiries, ivs);
    let lv = dupire_local_vol(&surf, 100.0, 0.03, 0.0, 2, 1);
    println!("[LocalVol] ATM local vol = {:.4}", lv);
    println!();
}

fn batch_bench() {
    let chain: Vec<OptionContract> = (0..500).map(|i| OptionContract {
        spot: 100.0,
        strike: 60.0 + i as f64 * 0.2,
        expiry: 0.5,
        rate: 0.03,
        div_yield: 0.0,
        vol: 0.20,
        opt_type: if i % 2 == 0 { OptionType::Call } else { OptionType::Put },
    }).collect();

    let t0 = Instant::now();
    let prices = batch_bsm_price(&chain);
    let bsm_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let heston_params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
    let heston_prices = batch_heston(&chain, &heston_params);
    let heston_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("[Batch] {} options:", chain.len());
    println!("  BSM    : {:.2}ms  ({:.0} opts/ms)", bsm_ms, chain.len() as f64 / bsm_ms);
    println!("  Heston : {:.2}ms  ({:.0} opts/ms)", heston_ms, chain.len() as f64 / heston_ms);
    println!("  BSM[0]={:.4}  Heston[0]={:.4}", prices[0], heston_prices[0]);
}

// Quick smoke test + timing. Not a unit test — just something you can
// run to sanity-check the whole stack after a refactor.
// Run with: cargo run --release

//! @file main.rs
//! @author Taha - Algorithmic Trader
//! @brief Institutional-grade Options Pricing Engine.
//! 
//! @note This is a public structural showcase. For full production-grade 
//!       deployment, architecture consulting, or recruitment inquiries:
//!       Contact: email: fadilrezokt@gmail.com / linkedin.com/in/tahaotc
