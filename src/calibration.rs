// Heston calibration via Levenberg-Marquardt.
// fits (v0, kappa, theta, sigma, rho) to a slice of (contract, iv_market) pairs.
//
// why IVs and not prices? prices overweight ITM options by ~10x. fitting in
// vol space treats a 10d wing the same as an ATM — which is what you want
// when you care about the shape of the surface, not just the center.
//
// LM in a nutshell: Newton with a damping term that makes it behave like
// gradient descent when the Jacobian is badly conditioned. lambda up = more
// cautious, lambda down = more aggressive. standard Marquardt update rule.
//
// Jacobian: FD central differences on each param. analytic would be faster
// but this isn't on the hot path — calibration runs offline or on surface updates.

use crate::types::{HestonParams, OptionContract, IvProblem};
use crate::heston::heston_price;
use crate::iv::implied_vol;

const MAX_ITER:   usize = 200;
const TOL_GRAD:   f64   = 1e-8;   // stop when gradient norm < this
const TOL_PARAMS: f64   = 1e-10;  // stop when param step < this
const LM_INIT:    f64   = 1e-3;   // initial damping factor
const LM_UP:      f64   = 10.0;   // damping multiplier on bad step
const LM_DOWN:    f64   = 0.1;    // damping multiplier on good step
const LM_MAX:     f64   = 1e8;    // bail if damping gets this large

// bump sizes for Jacobian FD. small enough for accuracy, big enough to avoid
// numerical noise from the GK integrator.
const D_V0:    f64 = 1e-4;
const D_KAPPA: f64 = 1e-3;
const D_THETA: f64 = 1e-4;
const D_SIGMA: f64 = 1e-3;
const D_RHO:   f64 = 1e-4;

pub struct CalibInput<'a> {
    pub contract:  &'a OptionContract,
    pub iv_market: f64,
    pub weight:    f64,  // typically 1/vega or uniform. set to 1.0 if you don't care.
}

pub struct CalibResult {
    pub params:    HestonParams,
    pub rmse:      f64,   // weighted RMSE in vol points
    pub max_err:   f64,   // worst single option error
    pub iters:     usize,
    pub converged: bool,
}

// main entry point. pass a reasonable initial guess — ATM vol^2 for v0/theta,
// kappa=1-3, sigma=0.3-0.5, rho=-0.5 to -0.7 is usually fine.
pub fn calibrate_heston(
    quotes:  &[CalibInput],
    p0:      HestonParams,
) -> CalibResult {
    assert!(!quotes.is_empty(), "nothing to calibrate");

    let mut p   = params_to_vec(&p0);
    let mut lam = LM_INIT;
    let mut res = residuals(quotes, &vec_to_params(&p));
    let mut sse = weighted_sse(&res, quotes);

    let mut iters     = 0;
    let mut converged = false;

    for iter in 0..MAX_ITER {
        iters = iter + 1;

        let j   = jacobian(quotes, &vec_to_params(&p));
        let jtr = jtj_and_grad(&j, &res, quotes);
        let (jtj, grad) = jtr;

        if grad.iter().map(|g| g*g).sum::<f64>().sqrt() < TOL_GRAD {
            converged = true;
            break;
        }

        // LM update: (J'J + lambda*diag(J'J)) * dp = -J'r
        let dp = lm_step(&jtj, &grad, lam);

        let p_new    = apply_step(&p, &dp);
        let hp       = vec_to_params(&p_new);

        // reject any step that violates Feller or param bounds
        if !hp.feller_ok() || !bounds_ok(&hp) {
            lam = (lam * LM_UP).min(LM_MAX);
            continue;
        }

        let res_new = residuals(quotes, &hp);
        let sse_new = weighted_sse(&res_new, quotes);

        if sse_new < sse {
            // good step
            p   = p_new;
            res = res_new;
            sse = sse_new;
            lam = (lam * LM_DOWN).max(1e-12);

            let step_norm = dp.iter().map(|d| d*d).sum::<f64>().sqrt();
            if step_norm < TOL_PARAMS {
                converged = true;
                break;
            }
        } else {
            // bad step — increase damping and retry with same params
            lam = (lam * LM_UP).min(LM_MAX);
            if lam >= LM_MAX { break; }
        }
    }

    let hp     = vec_to_params(&p);
    let wmse   = sse / quotes.iter().map(|q| q.weight * q.weight).sum::<f64>();
    let rmse   = wmse.sqrt();
    let max_err = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

    CalibResult { params: hp, rmse, max_err, iters, converged }
}

// residual for one option: iv_heston(p) - iv_market.
// returns 0.0 if heston or iv solver bails — don't let one bad quote blow up the fit.
fn single_residual(contract: &OptionContract, iv_mkt: f64, p: &HestonParams) -> f64 {
    let px = heston_price(
        contract.spot, contract.strike, contract.expiry,
        contract.rate, contract.div_yield, p, contract.opt_type,
    );
    // need a contract with a vol field to run the iv solver — use iv_mkt as placeholder
    let c_for_iv = OptionContract { vol: iv_mkt, ..*contract };
    match implied_vol(&IvProblem { contract: c_for_iv, market_price: px }) {
        Some(iv) => iv - iv_mkt,
        None     => 0.0,
    }
}

fn residuals(quotes: &[CalibInput], p: &HestonParams) -> Vec<f64> {
    quotes.iter()
        .map(|q| single_residual(q.contract, q.iv_market, p))
        .collect()
}

fn weighted_sse(res: &[f64], quotes: &[CalibInput]) -> f64 {
    res.iter().zip(quotes.iter())
        .map(|(r, q)| (q.weight * r).powi(2))
        .sum()
}

// central FD Jacobian. one column per param, one row per option.
fn jacobian(quotes: &[CalibInput], p: &HestonParams) -> Vec<Vec<f64>> {
    let bumps = [
        (HestonParams { v0:    p.v0    + D_V0,    ..*p }, HestonParams { v0:    (p.v0    - D_V0).max(1e-8),    ..*p }, 2.0*D_V0),
        (HestonParams { kappa: p.kappa + D_KAPPA, ..*p }, HestonParams { kappa: (p.kappa - D_KAPPA).max(1e-6), ..*p }, 2.0*D_KAPPA),
        (HestonParams { theta: p.theta + D_THETA, ..*p }, HestonParams { theta: (p.theta - D_THETA).max(1e-8), ..*p }, 2.0*D_THETA),
        (HestonParams { sigma: p.sigma + D_SIGMA, ..*p }, HestonParams { sigma: (p.sigma - D_SIGMA).max(1e-6), ..*p }, 2.0*D_SIGMA),
        (HestonParams { rho:   (p.rho  + D_RHO).min(0.9999), ..*p },
         HestonParams { rho:   (p.rho  - D_RHO).max(-0.9999), ..*p }, 2.0*D_RHO),
    ];

    let mut j = vec![vec![0.0; 5]; quotes.len()];
    for (col, (pu, pd, h)) in bumps.iter().enumerate() {
        let ru = residuals(quotes, pu);
        let rd = residuals(quotes, pd);
        for row in 0..quotes.len() {
            j[row][col] = (ru[row] - rd[row]) / h;
        }
    }
    j
}

// J'J and J'r in one pass
fn jtj_and_grad(j: &[Vec<f64>], res: &[f64], quotes: &[CalibInput]) -> ([[f64; 5]; 5], [f64; 5]) {
    let mut jtj  = [[0.0_f64; 5]; 5];
    let mut grad = [0.0_f64; 5];
    for (row, (r, q)) in res.iter().zip(quotes.iter()).enumerate() {
        let w2 = q.weight * q.weight;
        for c1 in 0..5 {
            grad[c1] += w2 * j[row][c1] * r;
            for c2 in 0..5 {
                jtj[c1][c2] += w2 * j[row][c1] * j[row][c2];
            }
        }
    }
    (jtj, grad)
}

// solve (J'J + lam*diag(J'J)) * dp = -grad via Cholesky-ish (just Gaussian
// elimination — 5x5 system, not worth pulling in a linear algebra crate).
fn lm_step(jtj: &[[f64; 5]; 5], grad: &[f64; 5], lam: f64) -> [f64; 5] {
    let mut a = *jtj;
    let mut b = *grad;

    // damp diagonal
    for i in 0..5 {
        a[i][i] *= 1.0 + lam;
        b[i] = -b[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..5 {
        let pivot = (col..5).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);

        if a[col][col].abs() < 1e-15 { continue; }
        let inv = 1.0 / a[col][col];
        for row in (col+1)..5 {
            let f = a[row][col] * inv;
            for k in col..5 { a[row][k] -= f * a[col][k]; }
            b[row] -= f * b[col];
        }
    }

    // back substitution
    let mut dp = [0.0_f64; 5];
    for i in (0..5).rev() {
        let mut s = b[i];
        for j in (i+1)..5 { s -= a[i][j] * dp[j]; }
        dp[i] = if a[i][i].abs() > 1e-15 { s / a[i][i] } else { 0.0 };
    }
    dp
}

fn apply_step(p: &[f64; 5], dp: &[f64; 5]) -> [f64; 5] {
    [p[0]+dp[0], p[1]+dp[1], p[2]+dp[2], p[3]+dp[3], p[4]+dp[4]]
}

fn params_to_vec(p: &HestonParams) -> [f64; 5] {
    [p.v0, p.kappa, p.theta, p.sigma, p.rho]
}

fn vec_to_params(v: &[f64; 5]) -> HestonParams {
    HestonParams { v0: v[0], kappa: v[1], theta: v[2], sigma: v[3], rho: v[4] }
}

// hard bounds — if a step lands outside these, reject it.
// loose enough to not interfere with calibration, tight enough to keep params sane.
fn bounds_ok(p: &HestonParams) -> bool {
    p.v0    > 1e-8 && p.v0    < 5.0  &&
    p.kappa > 1e-6 && p.kappa < 50.0 &&
    p.theta > 1e-8 && p.theta < 5.0  &&
    p.sigma > 1e-6 && p.sigma < 10.0 &&
    p.rho   > -0.9999 && p.rho < 0.9999
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionType;

    fn make_quotes(p: &HestonParams) -> Vec<(OptionContract, f64)> {
        let strikes  = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
        let expiries = [0.25, 0.5, 1.0];
        let mut out  = vec![];
        for &t in &expiries {
            for &k in &strikes {
                let opt_type = if k >= 100.0 { OptionType::Call } else { OptionType::Put };
                let px = heston_price(100.0, k, t, 0.03, 0.0, p, opt_type);
                let c  = OptionContract {
                    spot: 100.0, strike: k, expiry: t,
                    rate: 0.03, div_yield: 0.0, vol: 0.2, opt_type,
                };
                if let Some(iv) = implied_vol(&IvProblem { contract: c, market_price: px }) {
                    if iv > 0.01 && iv < 2.0 { out.push((c, iv)); }
                }
            }
        }
        out
    }

    #[test]
    fn recovers_params_from_synthetic_surface() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        // perturbed initial guess
        let p0 = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };
        let res = calibrate_heston(&quotes, p0);

        assert!(res.converged, "calibration did not converge after {} iters", res.iters);
        assert!(res.rmse < 0.005, "rmse={:.6} — surface fit too poor", res.rmse);
        assert!((res.params.v0    - true_p.v0   ).abs() < 0.005, "v0 off");
        assert!((res.params.rho   - true_p.rho  ).abs() < 0.05,  "rho off");
    }

    #[test]
    fn feller_always_satisfied_after_calibration() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();
        let p0  = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };
        let res = calibrate_heston(&quotes, p0);
        assert!(res.params.feller_ok(), "Feller violated: {:?}", res.params);
    }
}
