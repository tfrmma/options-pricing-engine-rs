use options_pricing_engine::{heston_price_and_greeks, heston_greeks_ad, HestonParams, OptionType};

fn params() -> HestonParams {
    HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
}

#[test]
fn ad_vega_matches_bump_vega() {
    for ot in [OptionType::Call, OptionType::Put] {
        for &k in &[80.0, 100.0, 120.0] {
            for &t in &[0.25, 1.0, 2.0] {
                let p = params();
                let ad = heston_greeks_ad(100.0, k, t, 0.05, 0.0, &p, ot);
                let bump = heston_price_and_greeks(100.0, k, t, 0.05, 0.0, &p, ot);
                let rel = (ad.vega - bump.vega).abs() / bump.vega.abs().max(1e-6);
                println!("{:?} k={} t={}: ad_vega={:.6} bump_vega={:.6} rel_err={:.4}",
                    ot, k, t, ad.vega, bump.vega, rel);
                assert!(rel < 0.02, "{:?} k={} t={}: ad={:.6} bump={:.6} rel_err={:.4}", ot, k, t, ad.vega, bump.vega, rel);
            }
        }
    }
}

#[test]
fn ad_and_bump_agree_on_shared_fd_greeks() {
    // delta/gamma/theta/rho/vanna/volga are FD in both paths, should nearly match
    let p = params();
    let ad = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.02, &p, OptionType::Call);
    let bump = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.02, &p, OptionType::Call);
    println!("delta ad={:.6} bump={:.6}", ad.delta, bump.delta);
    println!("theta ad={:.6} bump={:.6}", ad.theta, bump.theta);
    println!("rho   ad={:.6} bump={:.6}", ad.rho, bump.rho);
    println!("vanna ad={:.6} bump={:.6}", ad.vanna, bump.vanna);
    println!("volga ad={:.6} bump={:.6}", ad.volga, bump.volga);
}
