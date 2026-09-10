use options_pricing_engine::{bates_price_and_greeks, bates_greeks_ad, HestonParams, BatesParams, OptionType};

fn params() -> BatesParams {
    BatesParams {
        heston: HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 },
        lambda: 0.3, mu_j: -0.1, sigma_j: 0.15,
    }
}

#[test]
fn ad_vega_matches_bump_vega() {
    for ot in [OptionType::Call, OptionType::Put] {
        for &k in &[80.0, 100.0, 120.0] {
            let p = params();
            let ad = bates_greeks_ad(100.0, k, 1.0, 0.05, 0.0, &p.heston, p.lambda, p.mu_j, p.sigma_j, ot);
            let bump = bates_price_and_greeks(100.0, k, 1.0, 0.05, 0.0, &p, ot);
            let rel = (ad.vega - bump.vega).abs() / bump.vega.abs().max(1e-6);
            println!("{:?} k={}: ad_vega={:.6} bump_vega={:.6} rel_err={:.4}", ot, k, ad.vega, bump.vega, rel);
            assert!(rel < 0.02, "{:?} k={}: ad={:.6} bump={:.6}", ot, k, ad.vega, bump.vega);
        }
    }
}
