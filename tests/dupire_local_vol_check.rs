use options_pricing_engine::{LocalVolSurface, dupire_local_vol, monotone_cubic_interp};

// Independent reference implementation of Gatheral's Dupire formula,
// differentiating directly w.r.t. log-moneyness x (not raw strike K),
// using the crate's own monotone_cubic_interp so both paths share the
// same interpolant and only the differentiation variable differs.
fn reference_local_vol(
    strikes: &[f64], expiries: &[f64], local_vols: &[f64],
    spot: f64, rate: f64, div_yield: f64,
    i_k: usize, j_t: usize,
) -> f64 {
    let nk = strikes.len();
    let k = strikes[i_k];
    let t = expiries[j_t];

    // total variance as a function of K at this maturity slice (via spline)
    let iv_col: Vec<f64> = (0..nk).map(|i| local_vols[i*expiries.len() + j_t]).collect();
    let ivar_at_k = |kk: f64| -> f64 {
        let iv = monotone_cubic_interp(strikes, &iv_col, kk);
        iv*iv*t
    };
    // total variance as function of maturity at this strike (via spline over expiries)
    let ne = expiries.len();
    let iv_row: Vec<f64> = (0..ne).map(|j| local_vols[i_k*ne + j]).collect();
    let ivar_at_t = |tt: f64| -> f64 {
        let iv = monotone_cubic_interp(expiries, &iv_row, tt);
        iv*iv*tt
    };

    let x0 = (spot/k).ln() + (rate - div_yield)*t;
    let w0 = ivar_at_k(k);

    // K(x) = spot * exp((r-q)T - x)
    let k_of_x = |x: f64| spot * ((rate-div_yield)*t - x).exp();

    let h = 1e-4;
    let dw_dx   = (ivar_at_k(k_of_x(x0+h)) - ivar_at_k(k_of_x(x0-h))) / (2.0*h);
    let d2w_dx2 = (ivar_at_k(k_of_x(x0+h)) - 2.0*w0 + ivar_at_k(k_of_x(x0-h))) / (h*h);

    let ht = 1e-4;
    let dw_dt = (ivar_at_t(t+ht) - ivar_at_t(t-ht)) / (2.0*ht);

    let num = dw_dt;
    let denom = 1.0 - (x0/w0)*dw_dx
              + 0.25*(-0.25 - 1.0/w0 + x0*x0/(w0*w0))*dw_dx*dw_dx
              + 0.5*d2w_dx2;
    if denom <= 1e-10 || num <= 0.0 { return 0.0; }
    (num/denom).max(0.0).sqrt()
}

#[test]
fn dupire_matches_independent_x_based_reference() {
    // realistic equity-style skew: IV decreasing in strike, term structure rising
    let strikes: Vec<f64>  = vec![70.0, 85.0, 100.0, 115.0, 130.0];
    let expiries: Vec<f64> = vec![0.25, 0.5, 1.0, 2.0];
    let spot = 100.0_f64;
    let mut local_vols = vec![0.0; strikes.len()*expiries.len()];
    for (i, &k) in strikes.iter().enumerate() {
        for (j, &t) in expiries.iter().enumerate() {
            let moneyness = (k/spot).ln();
            // skew: -0.15 per unit log-moneyness, term structure: base rises with sqrt(T)
            let iv = 0.25 - 0.15*moneyness + 0.03*t.sqrt();
            local_vols[i*expiries.len()+j] = iv;
        }
    }
    let surf = LocalVolSurface::new(strikes.clone(), expiries.clone(), local_vols.clone());
    let (rate, div_yield) = (0.03, 0.01);

    for i_k in 1..strikes.len()-1 {
        for j_t in 1..expiries.len()-1 {
            let repo_val = dupire_local_vol(&surf, spot, rate, div_yield, i_k, j_t);
            let ref_val  = reference_local_vol(&strikes, &expiries, &local_vols, spot, rate, div_yield, i_k, j_t);
            println!("K={} T={}: repo={:.6} reference(x-based)={:.6}", strikes[i_k], expiries[j_t], repo_val, ref_val);
        }
    }
}
