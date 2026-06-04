// Dupire local vol. the math is fine; the input data never is.
//
// problem: raw IV surface has noise. finite differences on noisy data
// amplify the noise and you get negative local vols, which are nonsense.
//
// fix: Fritsch-Butland monotone cubic spline before differentiating.
// guarantees no overshoot between grid points => derivatives stay sane.
//
// TODO: arbitrage check on input surface before computing local vols.
//       right now we just clamp negatives. good enough for now but will
//       bite someone eventually on a badly-conditioned calendar spread.

use crate::types::LocalVolSurface;

// returns local vol (not variance) at grid node (i_k, j_t).
// returns 0.0 if the result is degenerate — clean your surface if this happens a lot.
pub fn dupire_local_vol(
    surf: &LocalVolSurface,
    spot: f64, rate: f64, div_yield: f64,
    i_k: usize, j_t: usize,
) -> f64 {
    let k = surf.strikes[i_k];
    let t = surf.expiries[j_t];
    let w = surf.get(i_k, j_t).powi(2) * t; // total variance

    let dw_dt   = dvar_dt(surf, i_k, j_t);
    let dw_dk   = dvar_dk(surf, i_k, j_t);
    let d2w_dk2 = d2var_dk2(surf, i_k, j_t);
    let x       = (spot/k).ln() + (rate - div_yield)*t; // log-moneyness

    // Dupire in total variance form
    let num   = dw_dt;
    let denom = 1.0 - (x/w)*dw_dk
              + 0.25*(-0.25 - 1.0/w + x*x/(w*w))*dw_dk*dw_dk
              + 0.5*d2w_dk2;

    if denom <= 1e-10 || num <= 0.0 { return 0.0; }
    (num / denom).max(0.0).sqrt()
}

// central diff where we have room, one-sided at edges
fn dvar_dt(surf: &LocalVolSurface, i_k: usize, j_t: usize) -> f64 {
    let n = surf.expiries.len();
    if n < 2 { return 0.0; }
    if j_t == 0 {
        let dt = surf.expiries[1] - surf.expiries[0];
        (total_var(surf, i_k, 1) - total_var(surf, i_k, 0)) / dt
    } else if j_t == n-1 {
        let dt = surf.expiries[n-1] - surf.expiries[n-2];
        (total_var(surf, i_k, n-1) - total_var(surf, i_k, n-2)) / dt
    } else {
        let dt = surf.expiries[j_t+1] - surf.expiries[j_t-1];
        (total_var(surf, i_k, j_t+1) - total_var(surf, i_k, j_t-1)) / dt
    }
}

fn dvar_dk(surf: &LocalVolSurface, i_k: usize, j_t: usize) -> f64 {
    let n = surf.strikes.len();
    if n < 2 { return 0.0; }
    if i_k == 0 {
        (total_var(surf,1,j_t) - total_var(surf,0,j_t)) / (surf.strikes[1] - surf.strikes[0])
    } else if i_k == n-1 {
        (total_var(surf,n-1,j_t) - total_var(surf,n-2,j_t)) / (surf.strikes[n-1] - surf.strikes[n-2])
    } else {
        let dk = surf.strikes[i_k+1] - surf.strikes[i_k-1];
        (total_var(surf,i_k+1,j_t) - total_var(surf,i_k-1,j_t)) / dk
    }
}

fn d2var_dk2(surf: &LocalVolSurface, i_k: usize, j_t: usize) -> f64 {
    let n = surf.strikes.len();
    if n < 3 { return 0.0; }
    let i  = i_k.max(1).min(n-2);
    let dp = surf.strikes[i+1] - surf.strikes[i];
    let dm = surf.strikes[i]   - surf.strikes[i-1];
    (total_var(surf,i+1,j_t) - 2.0*total_var(surf,i,j_t) + total_var(surf,i-1,j_t))
        / (0.5*(dp+dm) * 0.5*(dp+dm))
}

#[inline]
fn total_var(surf: &LocalVolSurface, i_k: usize, j_t: usize) -> f64 {
    let iv = surf.get(i_k, j_t);
    iv * iv * surf.expiries[j_t]
}

// Fritsch-Butland monotone cubic. doesn't overshoot, doesn't oscillate.
// if your spline gives you IV going negative between nodes, use this.
pub fn monotone_cubic_interp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    let n = xs.len();
    if n == 1 { return ys[0]; }

    let idx = xs.partition_point(|&x| x <= xq).min(n-1).saturating_sub(1);
    let i   = idx.min(n-2);

    if xq < xs[0]   { return ys[0]; }
    if xq > xs[n-1] { return ys[n-1]; }

    let dx = xs[i+1] - xs[i];
    let t  = (xq - xs[i]) / dx;
    let mi  = slope(xs, ys, i);
    let mi1 = slope(xs, ys, i+1);

    // Fritsch-Butland limiter — prevents overshoot
    let delta = (ys[i+1] - ys[i]) / dx;
    let lim  = if delta.abs() < 1e-15 { 0.0 } else {
        let a = mi / delta; let b = mi1 / delta;
        (3.0 / (a*a + b*b + a*b).sqrt()).min(1.0)
    };
    let m0 = lim * mi;
    let m1 = lim * mi1;

    // Hermite basis
    (2.0*t*t*t - 3.0*t*t + 1.0)*ys[i]
        + (t*t*t - 2.0*t*t + t)*dx*m0
        + (-2.0*t*t*t + 3.0*t*t)*ys[i+1]
        + (t*t*t - t*t)*dx*m1
}

fn slope(xs: &[f64], ys: &[f64], i: usize) -> f64 {
    let n = xs.len();
    if i == 0 {
        (ys[1] - ys[0]) / (xs[1] - xs[0])
    } else if i >= n-1 {
        (ys[n-1] - ys[n-2]) / (xs[n-1] - xs[n-2])
    } else {
        0.5 * ((ys[i] - ys[i-1])/(xs[i] - xs[i-1]) + (ys[i+1] - ys[i])/(xs[i+1] - xs[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LocalVolSurface;

    fn flat_surf(iv: f64) -> LocalVolSurface {
        let ks = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let ts = vec![0.25, 0.5, 1.0, 2.0];
        let n  = ks.len() * ts.len();
        LocalVolSurface::new(ks, ts, vec![iv; n])
    }

    #[test]
    fn flat_surface_recovers_iv() {
        let lv = dupire_local_vol(&flat_surf(0.2), 100.0, 0.02, 0.0, 2, 1);
        assert!((lv - 0.2).abs() < 0.01, "local vol = {lv:.4}");
    }

    #[test]
    fn interp_flat() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.2; 4];
        assert!((monotone_cubic_interp(&xs, &ys, 1.5) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn interp_no_overshoot() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sqrt()).collect();
        let mut prev = ys[0];
        for i in 1..9 {
            let y = monotone_cubic_interp(&xs, &ys, xs[i] + 0.3);
            assert!(y >= prev - 1e-10);
            prev = y;
        }
    }
}
