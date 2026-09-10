use options_pricing_engine::{bsm_price, bsm_price_and_greeks, black76_price_and_greeks, OptionContract, OptionType};

fn contract(t: f64, ot: OptionType, q: f64, r: f64, v: f64) -> OptionContract {
    OptionContract { spot: 100.0, strike: 100.0, expiry: t, rate: r, div_yield: q, vol: v, opt_type: ot }
}

fn fd_theta(c: &OptionContract, dt: f64) -> f64 {
    let up = OptionContract { expiry: c.expiry + dt, ..*c };
    let dn = OptionContract { expiry: c.expiry - dt, ..*c };
    -(bsm_price(&up) - bsm_price(&dn)) / (2.0 * dt)
}
fn fd_delta(c: &OptionContract, ds: f64) -> f64 {
    let up = OptionContract { spot: c.spot + ds, ..*c };
    let dn = OptionContract { spot: c.spot - ds, ..*c };
    (bsm_price(&up) - bsm_price(&dn)) / (2.0 * ds)
}
fn fd_vega(c: &OptionContract, dv: f64) -> f64 {
    let up = OptionContract { vol: c.vol + dv, ..*c };
    let dn = OptionContract { vol: c.vol - dv, ..*c };
    (bsm_price(&up) - bsm_price(&dn)) / (2.0 * dv)
}
fn fd_rho(c: &OptionContract, dr: f64) -> f64 {
    let up = OptionContract { rate: c.rate + dr, ..*c };
    let dn = OptionContract { rate: c.rate - dr, ..*c };
    (bsm_price(&up) - bsm_price(&dn)) / (2.0 * dr)
}
fn fd_vanna(c: &OptionContract, ds: f64, dv: f64) -> f64 {
    // d(delta)/d(vol)
    let up = OptionContract { vol: c.vol + dv, ..*c };
    let dn = OptionContract { vol: c.vol - dv, ..*c };
    (fd_delta(&up, ds) - fd_delta(&dn, ds)) / (2.0 * dv)
}
fn fd_volga(c: &OptionContract, dv: f64) -> f64 {
    // d(vega)/d(vol) = d2(price)/d(vol)^2
    let up = OptionContract { vol: c.vol + dv, ..*c };
    let dn = OptionContract { vol: c.vol - dv, ..*c };
    (bsm_price(&up) - 2.0*bsm_price(c) + bsm_price(&dn)) / (dv * dv)
}

fn assert_close(name: &str, analytic: f64, fd: f64, tol: f64) {
    let rel = (analytic - fd).abs() / fd.abs().max(1.0);
    assert!(rel < tol, "{name}: analytic={analytic:.6} fd={fd:.6} rel_err={rel:.6}");
}

#[test]
fn all_bsm_greeks_match_fd() {
    for ot in [OptionType::Call, OptionType::Put] {
        for &q in &[0.0, 0.02, 0.06] {
            for &r in &[0.0, 0.01, 0.05] {
                for &v in &[0.15, 0.25, 0.45] {
                    let c = contract(1.0, ot, q, r, v);
                    let g = bsm_price_and_greeks(&c);
                    assert_close("delta", g.delta, fd_delta(&c, 1e-4), 1e-4);
                    assert_close("vega",  g.vega,  fd_vega(&c, 1e-4),  1e-4);
                    assert_close("theta", g.theta, fd_theta(&c, 1e-5), 1e-3);
                    assert_close("rho",   g.rho,   fd_rho(&c, 1e-5),   1e-3);
                    assert_close("vanna", g.vanna, fd_vanna(&c, 1e-3, 1e-4), 1e-2);
                    assert_close("volga", g.volga, fd_volga(&c, 1e-3), 1e-2);
                }
            }
        }
    }
}

#[test]
fn black76_greeks_match_fd() {
    let (k, t, v) = (100.0, 1.0, 0.25);
    for ot in [OptionType::Call, OptionType::Put] {
        for &fwd in &[80.0, 100.0, 120.0] {
            for &r in &[0.0, 0.03, 0.07] {
                let g = black76_price_and_greeks(fwd, k, t, r, v, ot);
                let price = |f: f64, tt: f64| black76_price_and_greeks(f, k, tt, r, v, ot).price;
                let df = 1e-4;
                let fd_delta = (price(fwd+df, t) - price(fwd-df, t)) / (2.0*df);
                let dt = 1e-5;
                let fd_theta = -(price(fwd, t+dt) - price(fwd, t-dt)) / (2.0*dt);
                assert_close("b76 delta", g.delta, fd_delta, 1e-4);
                assert_close("b76 theta", g.theta, fd_theta, 1e-3);
            }
        }
    }
}
