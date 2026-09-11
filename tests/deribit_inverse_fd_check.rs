use options_pricing_engine::deribit_inverse::{price_coin, greeks};
use options_pricing_engine::OptionType;

fn fd_delta(ot: OptionType, f: f64, k: f64, v: f64, t: f64, df: f64) -> f64 {
    (price_coin(ot, f+df, k, v, t) - price_coin(ot, f-df, k, v, t)) / (2.0*df)
}
fn fd_theta(ot: OptionType, f: f64, k: f64, v: f64, t: f64, dt: f64) -> f64 {
    -(price_coin(ot, f, k, v, t+dt) - price_coin(ot, f, k, v, t-dt)) / (2.0*dt)
}
fn fd_vega(ot: OptionType, f: f64, k: f64, v: f64, t: f64, dv: f64) -> f64 {
    (price_coin(ot, f, k, v+dv, t) - price_coin(ot, f, k, v-dv, t)) / (2.0*dv)
}
fn fd_gamma(ot: OptionType, f: f64, k: f64, v: f64, t: f64, df: f64) -> f64 {
    (price_coin(ot, f+df, k, v, t) - 2.0*price_coin(ot, f, k, v, t) + price_coin(ot, f-df, k, v, t)) / (df*df)
}
fn fd_vanna(ot: OptionType, f: f64, k: f64, v: f64, t: f64, df: f64, dv: f64) -> f64 {
    (fd_delta(ot, f, k, v+dv, t, df) - fd_delta(ot, f, k, v-dv, t, df)) / (2.0*dv)
}
fn fd_volga(ot: OptionType, f: f64, k: f64, v: f64, t: f64, dv: f64) -> f64 {
    (price_coin(ot, f, k, v+dv, t) - 2.0*price_coin(ot, f, k, v, t) + price_coin(ot, f, k, v-dv, t)) / (dv*dv)
}

fn assert_close(name: &str, a: f64, b: f64, tol: f64) {
    let rel = (a - b).abs() / b.abs().max(1e-8);
    assert!(rel < tol, "{name}: analytic={a:.8} fd={b:.8} rel_err={rel:.6}");
}

#[test]
fn inverse_greeks_match_fd_grid() {
    for ot in [OptionType::Call, OptionType::Put] {
        for &f in &[80.0, 100.0, 120.0] {
            for &t in &[0.1, 0.5, 1.5] {
                for &v in &[0.4, 0.8] {   // BTC-scale vols
                    let k = 100.0;
                    let g = greeks(ot, f, k, v, t);
                    assert_close("delta", g.delta, fd_delta(ot, f, k, v, t, 1e-3), 1e-3);
                    assert_close("gamma", g.gamma, fd_gamma(ot, f, k, v, t, 1e-2), 1e-2);
                    assert_close("vega",  g.vega,  fd_vega(ot, f, k, v, t, 1e-4), 1e-3);
                    assert_close("theta", g.theta, fd_theta(ot, f, k, v, t, 1e-5), 1e-3);
                    assert_close("vanna", g.vanna, fd_vanna(ot, f, k, v, t, 1e-2, 1e-4), 2e-2);
                    assert_close("volga", g.volga, fd_volga(ot, f, k, v, t, 1e-3), 2e-2);
                }
            }
        }
    }
}
