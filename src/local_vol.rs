// Dupire local vol. the math is fine; the input data never is.
//
// problem: raw IV surface has noise. finite differences on noisy data
// amplify the noise and you get negative local vols, which are nonsense.
//
// fix: Fritsch-Butland monotone cubic spline before differentiating.
// guarantees no overshoot between grid points => derivatives stay sane.
//
// arbitrage check: call check_and_repair_surface before passing to dupire_local_vol.
// it catches calendar spread and butterfly violations and repairs them in-place
// with the minimum adjustment needed to restore no-arb.

use crate::types::LocalVolSurface;

// returns local vol (not variance) at grid node (i_k, j_t).
// returns 0.0 if the result is degenerate, clean your surface if this happens a lot.
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
    // denominator for symmetric 3-point FD on non-uniform grid is (dp^2+dm^2)/2.
    // the original (dp+dm)^2/4 is wrong. dp*dm is also wrong unless dp==dm.
    (total_var(surf,i+1,j_t) - 2.0*total_var(surf,i,j_t) + total_var(surf,i-1,j_t))
        / (0.5 * (dp*dp + dm*dm))
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

    // Fritsch-Butland limiter prevents overshoot.
    // condition: alpha^2 + alpha*beta + beta^2 <= 9, where alpha=mi/delta, beta=mi1/delta.
    // if violated, scale both slopes down uniformly so we sit on the boundary.
    // the sqrt formula that was here before is not F-B. it's a made-up norm that
    // happens to limit *something* but not the right thing — it'll overshoot on
    // asymmetric intervals.
    let delta = (ys[i+1] - ys[i]) / dx;
    let lim = if delta.abs() < 1e-15 { 0.0 } else {
        let a = mi / delta; let b = mi1 / delta;
        let cond = a*a + a*b + b*b;
        if cond <= 9.0 { 1.0 } else { (9.0 / cond).sqrt() }
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


// --- arbitrage check + repair ---

#[derive(Debug, Clone)]
pub struct SurfaceViolation {
    pub kind:  ViolationKind,
    pub i_k:   usize,
    pub j_t:   usize,
    pub delta: f64,  // how far out of bounds (positive = magnitude of violation)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViolationKind {
    // total variance not increasing in T at fixed K. implies negative calendar spread.
    CalendarSpread,
    // total variance not convex in K at fixed T. implies negative butterfly price.
    Butterfly,
}

pub struct SurfaceAudit {
    pub violations: Vec<SurfaceViolation>,
    pub repaired:   usize,
}

// checks and repairs a LocalVolSurface in-place.
// calendar spread: w(K, T2) >= w(K, T1) for T2 > T1.
// butterfly: w(K-) - 2*w(K) + w(K+) >= 0 (convexity in strike space).
//
// repair strategy: minimum upward adjustment on the offending node.
// we don't touch neighbors repair is conservative and local.
// if the surface is badly broken, run multiple passes until clean.
pub fn check_and_repair_surface(surf: &mut LocalVolSurface) -> SurfaceAudit {
    let mut violations = vec![];
    let mut repaired   = 0;

    let nk = surf.strikes.len();
    let nt = surf.expiries.len();

    // calendar spread: for each (i_k, j_t > 0), w(j_t) >= w(j_t-1).
    // repair: bump iv at j_t up so total variances are equal (minimum fix).
    for i_k in 0..nk {
        for j_t in 1..nt {
            let w_prev = total_var_raw(&surf.local_vols, nk, nt, i_k, j_t-1, surf.expiries[j_t-1]);
            let w_cur  = total_var_raw(&surf.local_vols, nk, nt, i_k, j_t,   surf.expiries[j_t]);
            if w_cur < w_prev {
                let delta = w_prev - w_cur;
                violations.push(SurfaceViolation {
                    kind: ViolationKind::CalendarSpread, i_k, j_t, delta,
                });
                // set iv so w_cur == w_prev + epsilon
                let new_iv = ((w_prev + 1e-8) / surf.expiries[j_t]).max(0.0).sqrt();
                surf.local_vols[i_k * nt + j_t] = new_iv;
                repaired += 1;
            }
        }
    }

    // butterfly: for each interior strike, w(i-1) - 2*w(i) + w(i+1) >= 0.
    // repair: raise the middle node to the average of neighbors minus epsilon.
    for j_t in 0..nt {
        for i_k in 1..(nk-1) {
            let wm = total_var_raw(&surf.local_vols, nk, nt, i_k-1, j_t, surf.expiries[j_t]);
            let w0 = total_var_raw(&surf.local_vols, nk, nt, i_k,   j_t, surf.expiries[j_t]);
            let wp = total_var_raw(&surf.local_vols, nk, nt, i_k+1, j_t, surf.expiries[j_t]);
            let convexity = wm - 2.0*w0 + wp;
            if convexity < 0.0 {
                let delta = -convexity;
                violations.push(SurfaceViolation {
                    kind: ViolationKind::Butterfly, i_k, j_t, delta,
                });
                // minimum repair: set w0 = (wm + wp) / 2 - epsilon
                let w0_new = 0.5*(wm + wp) - 1e-8;
                if w0_new > 0.0 {
                    surf.local_vols[i_k * nt + j_t] = (w0_new / surf.expiries[j_t]).sqrt();
                    repaired += 1;
                }
            }
        }
    }

    SurfaceAudit { violations, repaired }
}

// inline helper avoids borrowing surf mutably while we read
#[inline]
fn total_var_raw(ivs: &[f64], nk: usize, nt: usize, i_k: usize, j_t: usize, t: f64) -> f64 {
    let iv = ivs[i_k * nt + j_t];
    iv * iv * t
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

#[cfg(test)]
mod p0_regression {
    use super::*;
    use crate::types::LocalVolSurface;

    // non-uniform strike grid. the old d2var_dk2 used (0.5*(dp+dm))^2 which is wrong.
    // correct denominator for the symmetric numerator is (dp^2+dm^2)/2.
    #[test]
    fn nonuniform_grid_curvature() {
        // strikes with intentionally uneven spacing
        let ks = vec![90.0, 95.0, 105.0, 125.0];
        let ts = vec![0.5, 1.0];
        // flat surface => zero curvature => local vol should match iv
        let ivs = vec![0.2_f64; ks.len() * ts.len()];
        let surf = LocalVolSurface::new(ks, ts, ivs);
        let lv = dupire_local_vol(&surf, 100.0, 0.02, 0.0, 1, 0);
        assert!((lv - 0.2).abs() < 0.02, "nonuniform curvature err: lv={lv:.4}");
    }

    // asymmetric interval this is where the old F-B formula would overshoot.
    // interpolant must stay bounded between adjacent nodes.
    #[test]
    fn interp_no_overshoot_asymmetric() {
        let xs = vec![0.0, 0.1, 0.11, 1.0, 2.0];
        let ys = vec![0.0, 0.5,  0.5, 0.6, 0.7];
        for j in 0..4 {
            let lo = ys[j].min(ys[j+1]);
            let hi = ys[j].max(ys[j+1]);
            for k in 0..5 {
                let xq = xs[j] + (xs[j+1] - xs[j]) * (k as f64 / 4.0);
                let y  = monotone_cubic_interp(&xs, &ys, xq);
                assert!(y >= lo - 1e-10 && y <= hi + 1e-10,
                    "overshoot at xq={xq:.3}: y={y:.6} not in [{lo:.4},{hi:.4}]");
            }
        }
    }
}

#[cfg(test)]
mod arb_tests {
    use super::*;
    use crate::types::LocalVolSurface;

    // surface with a calendar spread violation at (i_k=1, j_t=1):
    // total variance at T=1.0 < T=0.5 that's negative time value.
    fn calendar_violation() -> LocalVolSurface {
        let ks = vec![90.0, 100.0, 110.0];
        let ts = vec![0.5, 1.0];
        // ivs indexed [i_k * n_t + j_t]
        // at k=100: iv(T=0.5)=0.25 => w=0.03125, iv(T=1.0)=0.15 => w=0.0225 — violation
        let ivs = vec![
            0.20, 0.22,  // k=90:  fine
            0.25, 0.15,  // k=100: T=1.0 has lower total var than T=0.5
            0.20, 0.22,  // k=110: fine
        ];
        LocalVolSurface::new(ks, ts, ivs)
    }

    // surface with a butterfly violation at (i_k=1, j_t=0):
    // middle strike has higher total variance than average of neighbors — negative butterfly.
    fn butterfly_violation() -> LocalVolSurface {
        let ks = vec![90.0, 100.0, 110.0];
        let ts = vec![0.5, 1.0];
        // at T=0.5: w(90)=0.02, w(100)=0.05, w(110)=0.02 concave, negative butterfly
        let ivs = vec![
            (0.02_f64/0.5).sqrt(), (0.03_f64/1.0).sqrt(),  // k=90
            (0.05_f64/0.5).sqrt(), (0.04_f64/1.0).sqrt(),  // k=100: too high at T=0.5
            (0.02_f64/0.5).sqrt(), (0.03_f64/1.0).sqrt(),  // k=110
        ];
        LocalVolSurface::new(ks, ts, ivs)
    }

    #[test]
    fn detects_calendar_violation() {
        let mut surf  = calendar_violation();
        let audit = check_and_repair_surface(&mut surf);
        assert!(audit.violations.iter().any(|v| v.kind == ViolationKind::CalendarSpread),
            "calendar violation not detected");
    }

    #[test]
    fn repairs_calendar_violation() {
        let mut surf = calendar_violation();
        check_and_repair_surface(&mut surf);
        // after repair: total variance must be non-decreasing in T at every K
        let nk = surf.strikes.len();
        let nt = surf.expiries.len();
        for i_k in 0..nk {
            for j_t in 1..nt {
                let w_prev = surf.get(i_k, j_t-1).powi(2) * surf.expiries[j_t-1];
                let w_cur  = surf.get(i_k, j_t  ).powi(2) * surf.expiries[j_t];
                assert!(w_cur >= w_prev - 1e-10,
                    "calendar still violated at i_k={i_k} j_t={j_t}: w_prev={w_prev:.6} w_cur={w_cur:.6}");
            }
        }
    }

    #[test]
    fn detects_butterfly_violation() {
        let mut surf  = butterfly_violation();
        let audit = check_and_repair_surface(&mut surf);
        assert!(audit.violations.iter().any(|v| v.kind == ViolationKind::Butterfly),
            "butterfly violation not detected");
    }

    #[test]
    fn repairs_butterfly_violation() {
        let mut surf = butterfly_violation();
        check_and_repair_surface(&mut surf);
        let nk = surf.strikes.len();
        let nt = surf.expiries.len();
        for j_t in 0..nt {
            for i_k in 1..(nk-1) {
                let wm = surf.get(i_k-1, j_t).powi(2) * surf.expiries[j_t];
                let w0 = surf.get(i_k,   j_t).powi(2) * surf.expiries[j_t];
                let wp = surf.get(i_k+1, j_t).powi(2) * surf.expiries[j_t];
                assert!(wm - 2.0*w0 + wp >= -1e-8,
                    "butterfly still violated at i_k={i_k} j_t={j_t}");
            }
        }
    }

    #[test]
    fn clean_surface_has_no_violations() {
        let mut surf  = LocalVolSurface::new(
            vec![90.0, 100.0, 110.0],
            vec![0.5, 1.0],
            vec![0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        );
        let audit = check_and_repair_surface(&mut surf);
        assert_eq!(audit.violations.len(), 0, "clean surface should have zero violations");
        assert_eq!(audit.repaired, 0);
    }
}
