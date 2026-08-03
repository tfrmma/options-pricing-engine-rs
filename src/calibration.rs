// Heston/Bates calibration via Levenberg-Marquardt.
// fits model params to a slice of (contract, iv_market) pairs.
//
// why IVs and not prices? prices overweight ITM options by ~10x. fitting in
// vol space treats a 10d wing the same as an ATM, which is what you want
// when you care about the shape of the surface, not just the center.
//
// LM in a nutshell: Newton with a damping term that makes it behave like
// gradient descent when the Jacobian is badly conditioned. lambda up = more
// cautious, lambda down = more aggressive. standard Marquardt update rule.
//
// Jacobian: FD central differences on each param. analytic would be faster
// but this isn't on the hot path, calibration runs offline or on surface updates.
//
// the engine (residuals/jacobian/lm_step/calibrate) is generic over
// CalibModel so Heston (5 params) and Bates (8 params, jump term folded in)
// share one implementation instead of two ~150-line copies that'd drift
// apart the first time someone tweaks the damping schedule for one and
// forgets the other.

use crate::types::{HestonParams, BatesParams, OptionContract, IvProblem};
use crate::bates::bates_price;
use crate::heston::heston_price;
use crate::iv::implied_vol;
use crate::mc::splitmix64;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;

const MAX_ITER:   usize = 200;
const TOL_GRAD:   f64   = 1e-8;   // stop when gradient norm < this
const TOL_PARAMS: f64   = 1e-10;  // stop when param step < this
const LM_INIT:    f64   = 1e-3;   // initial damping factor
const LM_UP:      f64   = 10.0;   // damping multiplier on bad step
const LM_DOWN:    f64   = 0.1;    // damping multiplier on good step
const LM_MAX:     f64   = 1e8;    // bail if damping gets this large

pub struct CalibInput<'a> {
    pub contract:  &'a OptionContract,
    pub iv_market: f64,
    pub weight:    f64,  // typically 1/vega or uniform. set to 1.0 if you don't care.
}

pub struct CalibResult<P = HestonParams> {
    pub params:    P,
    pub rmse:      f64,   // weighted RMSE in vol points
    pub max_err:   f64,   // worst single option error
    pub iters:     usize,
    pub converged: bool,
}

pub struct MultistartResult<P = HestonParams> {
    pub best:        CalibResult<P>,
    pub n_restarts:  usize,
    pub n_converged: usize,  // how many of the restarts actually converged
}

// anything LM can calibrate: a fixed-size param vector, a pricer, bounds,
// FD bump sizes, and a way to sample a random starting point for multistart.
pub trait CalibModel: Copy + Send + Sync {
    fn dim() -> usize;
    fn to_vec(&self) -> Vec<f64>;
    fn from_vec(v: &[f64]) -> Self;
    fn price(&self, c: &OptionContract) -> f64;
    fn bounds_ok(&self) -> bool;
    fn bump_sizes() -> Vec<f64>;
    fn param_bounds() -> Vec<(f64, f64)>; // used to clamp FD bump points near a boundary
    fn random_guess(rng: &mut SmallRng) -> Self;
}

impl CalibModel for HestonParams {
    fn dim() -> usize { 5 }
    fn to_vec(&self) -> Vec<f64> { vec![self.v0, self.kappa, self.theta, self.sigma, self.rho] }
    fn from_vec(v: &[f64]) -> Self {
        HestonParams { v0: v[0], kappa: v[1], theta: v[2], sigma: v[3], rho: v[4] }
    }
    fn price(&self, c: &OptionContract) -> f64 {
        heston_price(c.spot, c.strike, c.expiry, c.rate, c.div_yield, self, c.opt_type)
    }
    // hard bounds, if a step lands outside these, reject it. loose enough to
    // not interfere with calibration, tight enough to keep params sane.
    fn bounds_ok(&self) -> bool {
        self.feller_ok() &&
        self.v0    > 1e-8 && self.v0    < 5.0  &&
        self.kappa > 1e-6 && self.kappa < 50.0 &&
        self.theta > 1e-8 && self.theta < 5.0  &&
        self.sigma > 1e-6 && self.sigma < 10.0 &&
        self.rho   > -0.9999 && self.rho < 0.9999
    }
    fn bump_sizes()   -> Vec<f64>         { vec![1e-4, 1e-3, 1e-4, 1e-3, 1e-4] }
    fn param_bounds() -> Vec<(f64, f64)>  {
        vec![(1e-8, 5.0), (1e-6, 50.0), (1e-8, 5.0), (1e-6, 10.0), (-0.9999, 0.9999)]
    }
    fn random_guess(rng: &mut SmallRng) -> Self {
        let v0:    f64 = rng.gen_range(0.005..0.5);
        let kappa: f64 = rng.gen_range(0.3..8.0);
        let theta: f64 = rng.gen_range(0.005..0.5);
        // sigma capped so Feller holds by construction, no rejection loop needed
        let sigma_cap = ((2.0 * kappa * theta).sqrt() * 0.9).max(0.05);
        let sigma = rng.gen_range(0.02..sigma_cap);
        let rho   = rng.gen_range(-0.9..0.9);
        HestonParams { v0, kappa, theta, sigma, rho }
    }
}

impl CalibModel for BatesParams {
    fn dim() -> usize { 8 }
    fn to_vec(&self) -> Vec<f64> {
        vec![self.heston.v0, self.heston.kappa, self.heston.theta, self.heston.sigma, self.heston.rho,
             self.lambda, self.mu_j, self.sigma_j]
    }
    fn from_vec(v: &[f64]) -> Self {
        BatesParams {
            heston:  HestonParams { v0: v[0], kappa: v[1], theta: v[2], sigma: v[3], rho: v[4] },
            lambda:  v[5], mu_j: v[6], sigma_j: v[7],
        }
    }
    fn price(&self, c: &OptionContract) -> f64 {
        bates_price(c.spot, c.strike, c.expiry, c.rate, c.div_yield, self, c.opt_type)
    }
    fn bounds_ok(&self) -> bool {
        self.heston.bounds_ok() &&
        self.lambda  >= 0.0 && self.lambda  < 10.0 &&
        self.mu_j.abs() < 2.0 &&
        self.sigma_j > 1e-6 && self.sigma_j < 2.0
    }
    fn bump_sizes() -> Vec<f64> {
        let mut b = HestonParams::bump_sizes();
        b.extend([1e-4, 1e-4, 1e-4]); // lambda, mu_j, sigma_j
        b
    }
    fn param_bounds() -> Vec<(f64, f64)> {
        let mut b = HestonParams::param_bounds();
        b.extend([(0.0, 10.0), (-2.0, 2.0), (1e-6, 2.0)]);
        b
    }
    fn random_guess(rng: &mut SmallRng) -> Self {
        let heston  = HestonParams::random_guess(rng);
        let lambda  = rng.gen_range(0.0..2.0);
        let mu_j    = rng.gen_range(-0.3..0.1);
        let sigma_j = rng.gen_range(0.02..0.4);
        BatesParams { heston, lambda, mu_j, sigma_j }
    }
}

// main entry points. pass a reasonable initial guess, ATM vol^2 for v0/theta,
// kappa=1-3, sigma=0.3-0.5, rho=-0.5 to -0.7 is usually fine. for Bates,
// lambda=0.3-1.0, mu_j=-0.15 to -0.05, sigma_j=0.1-0.3 is a sane starting point.
pub fn calibrate_heston(quotes: &[CalibInput], p0: HestonParams) -> CalibResult {
    calibrate(quotes, p0)
}

pub fn calibrate_bates(quotes: &[CalibInput], p0: BatesParams) -> CalibResult<BatesParams> {
    calibrate(quotes, p0)
}

// single-start LM is a local optimizer, a bad p0 converges to a bad local
// minimum without complaining. this runs n_restarts LM fits in parallel
// (rayon, one thread per restart) from the caller's p0 plus n_restarts-1
// randomized starting points, and keeps the lowest-RMSE result. not a real
// global optimizer (no CMA-ES, no simulated annealing), just enough restarts
// that a single bad p0 doesn't quietly wreck the fit.
pub fn calibrate_heston_multistart(
    quotes: &[CalibInput], p0: HestonParams, n_restarts: usize, seed: u64,
) -> MultistartResult {
    calibrate_multistart(quotes, p0, n_restarts, seed)
}

pub fn calibrate_bates_multistart(
    quotes: &[CalibInput], p0: BatesParams, n_restarts: usize, seed: u64,
) -> MultistartResult<BatesParams> {
    calibrate_multistart(quotes, p0, n_restarts, seed)
}

fn calibrate_multistart<M: CalibModel>(
    quotes: &[CalibInput], p0: M, n_restarts: usize, seed: u64,
) -> MultistartResult<M> {
    let n_restarts = n_restarts.max(1);
    let results: Vec<CalibResult<M>> = (0..n_restarts)
        .into_par_iter()
        .map(|i| {
            // restart 0 always honors the caller's own guess, the rest are
            // randomized so a single bad p0 isn't the whole story.
            let start = if i == 0 { p0 } else {
                let mut rng = SmallRng::seed_from_u64(splitmix64(seed ^ i as u64));
                M::random_guess(&mut rng)
            };
            calibrate(quotes, start)
        })
        .collect();

    let n_converged = results.iter().filter(|r| r.converged).count();
    let best = results.into_iter()
        .min_by(|a, b| a.rmse.total_cmp(&b.rmse))
        .expect("n_restarts >= 1, there's always a best");

    MultistartResult { best, n_restarts, n_converged }
}

fn calibrate<M: CalibModel>(quotes: &[CalibInput], p0: M) -> CalibResult<M> {
    assert!(!quotes.is_empty(), "nothing to calibrate");
    let n = M::dim();

    let mut p   = p0.to_vec();
    let mut lam = LM_INIT;
    let mut res = residuals(quotes, &M::from_vec(&p));
    let mut sse = weighted_sse(&res, quotes);

    let mut iters     = 0;
    let mut converged = false;

    for iter in 0..MAX_ITER {
        iters = iter + 1;

        let j = jacobian::<M>(quotes, &M::from_vec(&p));
        let (jtj, grad) = jtj_and_grad(&j, &res, quotes, n);

        if grad.iter().map(|g| g*g).sum::<f64>().sqrt() < TOL_GRAD {
            converged = true;
            break;
        }

        // LM update: (J'J + lambda*diag(J'J)) * dp = -J'r
        let dp = lm_step(&jtj, &grad, lam, n);
        let p_new: Vec<f64> = p.iter().zip(dp.iter()).map(|(a, b)| a + b).collect();
        let m_new = M::from_vec(&p_new);

        // reject any step that violates Feller (Heston side) or param bounds
        if !m_new.bounds_ok() {
            lam = (lam * LM_UP).min(LM_MAX);
            continue;
        }

        let res_new = residuals(quotes, &m_new);
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
            // bad step, increase damping and retry with same params
            lam = (lam * LM_UP).min(LM_MAX);
            if lam >= LM_MAX { break; }
        }
    }

    let m       = M::from_vec(&p);
    let wmse    = sse / quotes.iter().map(|q| q.weight * q.weight).sum::<f64>();
    let rmse    = wmse.sqrt();
    let max_err = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

    CalibResult { params: m, rmse, max_err, iters, converged }
}

// residual for one option: iv_model(p) - iv_market.
// returns 0.0 if the pricer or iv solver bails, don't let one bad quote blow up the fit.
fn single_residual<M: CalibModel>(contract: &OptionContract, iv_mkt: f64, p: &M) -> f64 {
    let px = p.price(contract);
    // need a contract with a vol field to run the iv solver, use iv_mkt as placeholder
    let c_for_iv = OptionContract { vol: iv_mkt, ..*contract };
    match implied_vol(&IvProblem { contract: c_for_iv, market_price: px }) {
        Some(iv) => iv - iv_mkt,
        None     => 0.0,
    }
}

fn residuals<M: CalibModel>(quotes: &[CalibInput], p: &M) -> Vec<f64> {
    quotes.iter()
        .map(|q| single_residual(q.contract, q.iv_market, p))
        .collect()
}

fn weighted_sse(res: &[f64], quotes: &[CalibInput]) -> f64 {
    res.iter().zip(quotes.iter())
        .map(|(r, q)| (q.weight * r).powi(2))
        .sum()
}

// central FD Jacobian. one column per param, one row per option. bump points
// are clamped to each param's valid range before pricing (no negative v0
// going into the CF), and the denominator uses the actual clamped distance,
// not the nominal 2*h, so a param sitting near its boundary doesn't get a
// quietly biased derivative.
fn jacobian<M: CalibModel>(quotes: &[CalibInput], p: &M) -> Vec<Vec<f64>> {
    let base   = p.to_vec();
    let bumps  = M::bump_sizes();
    let bounds = M::param_bounds();
    let n      = M::dim();

    let mut j = vec![vec![0.0; n]; quotes.len()];
    for col in 0..n {
        let (lo, hi) = bounds[col];
        let mut vu = base.clone(); vu[col] = (base[col] + bumps[col]).min(hi);
        let mut vd = base.clone(); vd[col] = (base[col] - bumps[col]).max(lo);
        let h = vu[col] - vd[col];
        if h <= 0.0 { continue; } // pinned against both bounds, leave column zero

        let ru = residuals(quotes, &M::from_vec(&vu));
        let rd = residuals(quotes, &M::from_vec(&vd));
        for row in 0..quotes.len() {
            j[row][col] = (ru[row] - rd[row]) / h;
        }
    }
    j
}

// J'J and J'r in one pass
fn jtj_and_grad(j: &[Vec<f64>], res: &[f64], quotes: &[CalibInput], n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut jtj  = vec![vec![0.0_f64; n]; n];
    let mut grad = vec![0.0_f64; n];
    for (row, (r, q)) in res.iter().zip(quotes.iter()).enumerate() {
        let w2 = q.weight * q.weight;
        for c1 in 0..n {
            grad[c1] += w2 * j[row][c1] * r;
            for c2 in 0..n {
                jtj[c1][c2] += w2 * j[row][c1] * j[row][c2];
            }
        }
    }
    (jtj, grad)
}

// solve (J'J + lam*diag(J'J)) * dp = -grad via Gaussian elimination with
// partial pivoting. NxN now instead of hardcoded 5x5, still not worth
// pulling in a linear algebra crate for, N tops out at 8.
fn lm_step(jtj: &[Vec<f64>], grad: &[f64], lam: f64, n: usize) -> Vec<f64> {
    let mut a: Vec<Vec<f64>> = jtj.to_vec();
    let mut b: Vec<f64>      = grad.to_vec();

    for i in 0..n {
        a[i][i] *= 1.0 + lam;
        b[i] = -b[i];
    }

    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &k| a[i][col].abs().partial_cmp(&a[k][col].abs()).unwrap()).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);

        if a[col][col].abs() < 1e-15 { continue; }
        let inv = 1.0 / a[col][col];
        for row in (col+1)..n {
            let f = a[row][col] * inv;
            for k in col..n { a[row][k] -= f * a[col][k]; }
            b[row] -= f * b[col];
        }
    }

    let mut dp = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i+1)..n { s -= a[i][k] * dp[k]; }
        dp[i] = if a[i][i].abs() > 1e-15 { s / a[i][i] } else { 0.0 };
    }
    dp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionType;

    fn make_quotes_heston(p: &HestonParams) -> Vec<(OptionContract, f64)> {
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

    fn make_quotes_bates(p: &BatesParams) -> Vec<(OptionContract, f64)> {
        let strikes  = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
        let expiries = [0.25, 0.5, 1.0];
        let mut out  = vec![];
        for &t in &expiries {
            for &k in &strikes {
                let opt_type = if k >= 100.0 { OptionType::Call } else { OptionType::Put };
                let px = bates_price(100.0, k, t, 0.03, 0.0, p, opt_type);
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
        let raw    = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        // perturbed initial guess
        let p0 = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };
        let res = calibrate_heston(&quotes, p0);

        assert!(res.converged, "calibration did not converge after {} iters", res.iters);
        assert!(res.rmse < 0.005, "rmse={:.6}, surface fit too poor", res.rmse);
        assert!((res.params.v0    - true_p.v0   ).abs() < 0.005, "v0 off");
        assert!((res.params.rho   - true_p.rho  ).abs() < 0.05,  "rho off");
    }

    #[test]
    fn feller_always_satisfied_after_calibration() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();
        let p0  = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };
        let res = calibrate_heston(&quotes, p0);
        assert!(res.params.feller_ok(), "Feller violated: {:?}", res.params);
    }

    #[test]
    fn recovers_bates_params_from_synthetic_surface() {
        let true_p = BatesParams {
            heston: HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 },
            lambda: 0.5, mu_j: -0.1, sigma_j: 0.15,
        };
        let raw    = make_quotes_bates(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let p0 = BatesParams {
            heston: HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 },
            lambda: 0.3, mu_j: -0.05, sigma_j: 0.1,
        };
        let res = calibrate_bates(&quotes, p0);

        assert!(res.converged, "bates calibration did not converge after {} iters", res.iters);
        assert!(res.rmse < 0.01, "rmse={:.6}, bates surface fit too poor", res.rmse);
        assert!((res.params.heston.v0 - true_p.heston.v0).abs() < 0.01, "v0 off");
        assert!(res.params.heston.feller_ok(), "Feller violated: {:?}", res.params.heston);
    }

    // deliberately terrible p0 (miles from truth). single-start LM from here
    // frequently lands in a bad local minimum, multistart with a handful of
    // randomized restarts should reliably find something much better.
    #[test]
    fn multistart_beats_bad_single_start() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let bad_p0 = HestonParams { v0: 0.25, kappa: 6.0, theta: 0.3, sigma: 0.15, rho: 0.6 };
        let single = calibrate_heston(&quotes, bad_p0);
        let multi  = calibrate_heston_multistart(&quotes, bad_p0, 8, 12345);

        assert!(multi.n_converged >= 1, "no restart converged at all");
        assert!(multi.best.rmse <= single.rmse + 1e-9,
            "multistart={:.6} single={:.6}, multistart should never lose", multi.best.rmse, single.rmse);
        assert!(multi.best.rmse < 0.01, "even the best restart fit poorly: rmse={:.6}", multi.best.rmse);
    }
}
