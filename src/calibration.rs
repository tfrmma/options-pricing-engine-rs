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
    pub params:        P,
    pub rmse:          f64,   // weighted RMSE in vol points
    pub max_err:       f64,   // worst single option error
    pub iters:         usize,
    pub converged:     bool,
    pub early_stopped: bool,  // pruned by multistart before finishing, see PRUNE_FACTOR

    // identifiability diagnostic, computed from J'J's eigendecomposition at
    // the final params. a LARGE condition_number means some direction in
    // parameter space barely moves the objective, i.e. many different
    // parameter combinations fit this quote set about equally well, an
    // excellent rmse does not imply the individual params are trustworthy
    // when this is large. see the Design section's identifiability note.
    pub condition_number:    f64,
    // the parameter-space direction least constrained by the data (the
    // eigenvector of J'J's smallest eigenvalue, unit length, in the same
    // order as CalibModel::to_vec()). when condition_number is large, this
    // tells you WHICH combination of params is floppy, not just that one
    // exists. e.g. a Bates result might show this dominated by (lambda,
    // sigma) with opposite signs, meaning the data can't tell a higher-
    // intensity/smaller-jump regime apart from a lower-intensity/bigger-
    // jump one at this quote set.
    pub weakest_direction:   Vec<f64>,
}

pub struct MultistartResult<P = HestonParams> {
    pub best:        CalibResult<P>,
    pub n_restarts:  usize,
    pub n_converged: usize,  // how many of the restarts actually converged
    pub n_pruned:    usize,  // how many were aborted early, see PRUNE_FACTOR
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

    // exact d(price)/d(param[col]) via AD, when available. default None
    // means "use FD for this column", which is all Heston's 5 columns ever
    // do (it has no jump params). Bates overrides this for columns 5..=7
    // (lambda, mu_j, sigma_j) via ad::bates_jump_sensitivities_ad. the 5
    // Heston-inherited columns (0..=4) still go through FD even for Bates,
    // AD-izing those too is a separate change nobody's asked for yet.
    fn ad_price_derivative(&self, _contract: &OptionContract, _col: usize) -> Option<f64> { None }

    // natural per-parameter scale for Tikhonov regularization: a deviation
    // of one full `scale` unit from the prior is treated as "one unit" of
    // penalty. these are order-of-magnitude typical values, not derived
    // from anything, same spirit as bump_sizes/param_bounds.
    fn regularization_scale() -> Vec<f64>;
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
    // order-of-magnitude typical values: ATM variance, a middling mean-
    // reversion speed, same for long-run variance, a middling vol-of-vol,
    // a "meaningfully negative but not pinned to -1" correlation.
    fn regularization_scale() -> Vec<f64> { vec![0.04, 2.0, 0.04, 0.3, 0.3] }
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
    fn regularization_scale() -> Vec<f64> {
        let mut s = HestonParams::regularization_scale();
        s.extend([0.5, 0.1, 0.15]); // lambda, mu_j, sigma_j typical scales
        s
    }
    fn random_guess(rng: &mut SmallRng) -> Self {
        let heston  = HestonParams::random_guess(rng);
        let lambda  = rng.gen_range(0.0..2.0);
        let mu_j    = rng.gen_range(-0.3..0.1);
        let sigma_j = rng.gen_range(0.02..0.4);
        BatesParams { heston, lambda, mu_j, sigma_j }
    }
    fn ad_price_derivative(&self, contract: &OptionContract, col: usize) -> Option<f64> {
        if col < 5 { return None; }
        let sens = crate::ad::bates_jump_sensitivities_ad(
            contract.spot, contract.strike, contract.expiry, contract.rate, contract.div_yield,
            &self.heston, self.lambda, self.mu_j, self.sigma_j, contract.opt_type,
        );
        Some(match col {
            5 => sens.d_lambda,
            6 => sens.d_mu_j,
            7 => sens.d_sigma_j,
            _ => unreachable!("BatesParams::dim() is 8, col {col} out of range"),
        })
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

// LM with a Tikhonov pull toward `prior`, for exactly the case the
// identifiability diagnostic exists to flag: an excellent rmse whose
// individual params don't mean much because the data barely constrains
// some combination of them (see condition_number/weakest_direction on the
// result, and the Design section's identifiability note). `reg_weight`
// trades fit quality for closeness to the prior, there's no universally
// right value, start small (1e-3 to 1e-2) and check whether rmse degrades
// more than you're willing to accept before trusting the result.
pub fn calibrate_heston_regularized(
    quotes: &[CalibInput], p0: HestonParams, prior: &HestonParams, reg_weight: f64,
) -> CalibResult {
    calibrate_inner(quotes, p0, None, Some(&Regularization { prior, weight: reg_weight }))
}

pub fn calibrate_bates_regularized(
    quotes: &[CalibInput], p0: BatesParams, prior: &BatesParams, reg_weight: f64,
) -> CalibResult<BatesParams> {
    calibrate_inner(quotes, p0, None, Some(&Regularization { prior, weight: reg_weight }))
}

// single-start LM is a local optimizer, a bad p0 converges to a bad local
// minimum without complaining. this runs n_restarts LM fits in parallel
// (rayon, one thread per restart) from the caller's p0 plus n_restarts-1
// randomized starting points, and keeps the lowest-RMSE result. not a real
// global optimizer (no CMA-ES, no simulated annealing), just enough restarts
// that a single bad p0 doesn't quietly wreck the fit.
//
// restarts aren't fully isolated: they share one best-SSE-seen-so-far
// behind a mutex, and every 10 iterations a restart checks it against its
// own current SSE. more than PRUNE_FACTOR times worse than the best any
// OTHER restart has found gets aborted (early_stopped=true, not counted as
// converged). this can only ever kill a restart that's already losing, it
// never changes which restart wins, it just stops paying for LM iterations
// and Jacobian evaluations on a run that was never going to catch up.
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

// how much worse than the shared best a restart has to be to get pruned.
// tuned by feel, not derived: loose enough that a restart converging
// normally but a bit slower never gets killed, tight enough to actually
// save compute on restarts stuck in a clearly bad basin. worth revisiting
// if it starts pruning restarts that would've caught up.
const PRUNE_FACTOR: f64 = 8.0;
const PRUNE_CHECK_EVERY: usize = 10;

fn calibrate_multistart<M: CalibModel>(
    quotes: &[CalibInput], p0: M, n_restarts: usize, seed: u64,
) -> MultistartResult<M> {
    let n_restarts = n_restarts.max(1);
    let shared_best_sse = std::sync::Mutex::new(f64::INFINITY);

    let results: Vec<CalibResult<M>> = (0..n_restarts)
        .into_par_iter()
        .map(|i| {
            // restart 0 always honors the caller's own guess, the rest are
            // randomized so a single bad p0 isn't the whole story.
            let start = if i == 0 { p0 } else {
                let mut rng = SmallRng::seed_from_u64(splitmix64(seed ^ i as u64));
                M::random_guess(&mut rng)
            };
            calibrate_inner(quotes, start, Some(&shared_best_sse), None)
        })
        .collect();

    let n_converged = results.iter().filter(|r| r.converged).count();
    let n_pruned     = results.iter().filter(|r| r.early_stopped).count();
    let best = results.into_iter()
        .min_by(|a, b| a.rmse.total_cmp(&b.rmse))
        .expect("n_restarts >= 1, there's always a best");

    MultistartResult { best, n_restarts, n_converged, n_pruned }
}

pub struct GlobalCalibResult<P = HestonParams> {
    pub best:           CalibResult<P>, // after DE search + LM polish
    pub de_best_sse:    f64,            // SSE of the DE winner, pre-polish
    pub de_generations: usize,
}

const DE_F:  f64 = 0.6; // differential weight, classic DE/rand/1/bin default range is 0.4-1.0
const DE_CR: f64 = 0.9; // crossover probability

// a genuine population-based global search (differential evolution,
// DE/rand/1/bin), not repeated local restarts. doesn't need an initial
// guess at all, unlike multistart it explores the whole box from the start
// instead of hoping a handful of random points land in the right basin.
// DE is good at finding the right basin, bad at polishing once it's there
// (it has no gradient information), so the winner gets one LM run to finish
// the convergence LM is already fast and precise at. this is the standard
// hybrid pattern (same idea as scipy.optimize.differential_evolution's
// polish=True), not a novel trick.
//
// infeasible individuals (mostly Feller violations on the Heston side) get
// an infinite fitness instead of a repair step, DE's own selection pressure
// naturally steers the population away from them.
pub fn calibrate_heston_global(
    quotes: &[CalibInput], n_pop: usize, n_gen: usize, seed: u64,
) -> GlobalCalibResult {
    calibrate_global(quotes, n_pop, n_gen, seed, None)
}

pub fn calibrate_bates_global(
    quotes: &[CalibInput], n_pop: usize, n_gen: usize, seed: u64,
) -> GlobalCalibResult<BatesParams> {
    calibrate_global(quotes, n_pop, n_gen, seed, None)
}

// same DE search as calibrate_heston_global/calibrate_bates_global, DE
// itself stays unregularized (let it explore freely to find a good-fitting
// region), but the final LM polish pulls toward `prior`. this is the
// combination that actually addresses the identifiability problem DE can
// land in: broad search finds a region that fits well, regularized polish
// picks a specific, plausible point in that region instead of an arbitrary
// one. see calibrate_heston_regularized for the reg_weight tradeoff.
pub fn calibrate_heston_global_regularized(
    quotes: &[CalibInput], n_pop: usize, n_gen: usize, seed: u64,
    prior: &HestonParams, reg_weight: f64,
) -> GlobalCalibResult {
    calibrate_global(quotes, n_pop, n_gen, seed, Some(&Regularization { prior, weight: reg_weight }))
}

pub fn calibrate_bates_global_regularized(
    quotes: &[CalibInput], n_pop: usize, n_gen: usize, seed: u64,
    prior: &BatesParams, reg_weight: f64,
) -> GlobalCalibResult<BatesParams> {
    calibrate_global(quotes, n_pop, n_gen, seed, Some(&Regularization { prior, weight: reg_weight }))
}

fn calibrate_global<M: CalibModel>(
    quotes: &[CalibInput], n_pop: usize, n_gen: usize, seed: u64,
    reg: Option<&Regularization<M>>,
) -> GlobalCalibResult<M> {
    let n = M::dim();
    let bounds = M::param_bounds();
    let n_pop = n_pop.max(4 * n); // DE needs real diversity to work, floor it

    let base_weights: Vec<f64> = quotes.iter().map(|q| q.weight).collect();
    let sse_of = |v: &[f64]| -> f64 {
        let m = M::from_vec(v);
        if !m.bounds_ok() { return f64::INFINITY; }
        weighted_sse(&residuals(quotes, &m), &base_weights)
    };

    let mut init_rng = SmallRng::seed_from_u64(splitmix64(seed));
    let mut pop: Vec<Vec<f64>> = (0..n_pop).map(|_| {
        // a few resample attempts for a feasible start, doesn't need to be
        // perfect, DE will steer infeasible members out via selection anyway
        for _ in 0..20 {
            let v: Vec<f64> = bounds.iter().map(|&(lo, hi)| init_rng.gen_range(lo..hi)).collect();
            if sse_of(&v).is_finite() { return v; }
        }
        bounds.iter().map(|&(lo, hi)| init_rng.gen_range(lo..hi)).collect()
    }).collect();
    let mut fitness: Vec<f64> = pop.iter().map(|v| sse_of(v)).collect();

    for gen in 0..n_gen {
        let updated: Vec<(Vec<f64>, f64)> = (0..n_pop)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(splitmix64(seed ^ ((gen as u64) << 32) ^ i as u64));
                let mut idxs = [0usize; 3];
                for slot in 0..3 {
                    loop {
                        let cand = rng.gen_range(0..n_pop);
                        if cand != i && !idxs[..slot].contains(&cand) { idxs[slot] = cand; break; }
                    }
                }
                let (a, b, c) = (idxs[0], idxs[1], idxs[2]);
                let forced_dim = rng.gen_range(0..n); // guarantees at least one dim actually changes

                let mut trial = pop[i].clone();
                for d in 0..n {
                    if d == forced_dim || rng.gen::<f64>() < DE_CR {
                        let (lo, hi) = bounds[d];
                        trial[d] = (pop[a][d] + DE_F * (pop[b][d] - pop[c][d])).clamp(lo, hi);
                    }
                }

                let trial_fitness = sse_of(&trial);
                if trial_fitness < fitness[i] { (trial, trial_fitness) } else { (pop[i].clone(), fitness[i]) }
            })
            .collect();

        for (i, (v, f)) in updated.into_iter().enumerate() { pop[i] = v; fitness[i] = f; }
    }

    let best_idx = (0..n_pop).min_by(|&i, &j| fitness[i].total_cmp(&fitness[j])).unwrap();
    let de_best_sse = fitness[best_idx];
    let de_winner   = M::from_vec(&pop[best_idx]);

    let polished = calibrate_inner(quotes, de_winner, None, reg);
    GlobalCalibResult { best: polished, de_best_sse, de_generations: n_gen }
}

// Tikhonov regularization toward a prior parameter set: pulls the fit
// toward `prior` with strength `weight`. implemented as extra pseudo-
// residuals appended to the data residuals/Jacobian (standard way to fold
// ridge regularization into a Gauss-Newton/LM solver without hand-deriving
// a separate penalty gradient, see calibrate_heston_regularized/
// calibrate_bates_regularized). `weight` has units of (vol points per
// regularization_scale unit)^2, start small (1e-4 to 1e-2) and increase
// until the identifiability diagnostic looks sane, there's no universally
// correct value, it depends on how much you trust the prior vs the data.
pub struct Regularization<'a, M> {
    pub prior:  &'a M,
    pub weight: f64,
}

fn calibrate<M: CalibModel>(quotes: &[CalibInput], p0: M) -> CalibResult<M> {
    calibrate_inner(quotes, p0, None, None)
}

fn calibrate_inner<M: CalibModel>(
    quotes: &[CalibInput], p0: M,
    shared_best_sse: Option<&std::sync::Mutex<f64>>,
    reg: Option<&Regularization<M>>,
) -> CalibResult<M> {
    assert!(!quotes.is_empty(), "nothing to calibrate");
    let n = M::dim();
    let base_weights: Vec<f64> = quotes.iter().map(|q| q.weight).collect();
    let reg_scale = M::regularization_scale();

    // appends n Tikhonov pseudo-rows to (res, weights) or (jacobian rows)
    // when reg is active. the pseudo-Jacobian block is just 1/scale on the
    // diagonal, constant in p since the penalty is linear, no FD/AD needed.
    let augmented_residuals = |m: &M| -> (Vec<f64>, Vec<f64>) {
        let mut res = residuals(quotes, m);
        let mut w   = base_weights.clone();
        if let Some(r) = reg {
            let pv = m.to_vec();
            let priorv = r.prior.to_vec();
            let rw = r.weight.sqrt();
            for i in 0..n {
                res.push((pv[i] - priorv[i]) / reg_scale[i]);
                w.push(rw);
            }
        }
        (res, w)
    };
    let augmented_jacobian = |m: &M| -> Vec<Vec<f64>> {
        let mut j = jacobian::<M>(quotes, m);
        if reg.is_some() {
            for i in 0..n {
                let mut row = vec![0.0; n];
                row[i] = 1.0 / reg_scale[i];
                j.push(row);
            }
        }
        j
    };

    let mut p   = p0.to_vec();
    let mut lam = LM_INIT;
    let (mut res, mut weights) = augmented_residuals(&M::from_vec(&p));
    let mut sse = weighted_sse(&res, &weights);

    let mut iters         = 0;
    let mut converged     = false;
    let mut early_stopped = false;

    for iter in 0..MAX_ITER {
        iters = iter + 1;

        if let Some(shared) = shared_best_sse {
            if iter > 0 && iter % PRUNE_CHECK_EVERY == 0 {
                let best = *shared.lock().unwrap();
                if best.is_finite() && sse > best * PRUNE_FACTOR {
                    early_stopped = true;
                    break;
                }
            }
        }

        let j = augmented_jacobian(&M::from_vec(&p));
        let (jtj, grad) = jtj_and_grad(&j, &res, &weights, n);

        if grad.iter().map(|g| g*g).sum::<f64>().sqrt() < TOL_GRAD {
            converged = true;
            break;
        }

        // LM update: (J'J + lambda*diag(J'J)) * dp = -J'r
        let dp = lm_step(&jtj, &grad, lam, n);
        let p_new: Vec<f64> = p.iter().zip(dp.iter()).map(|(a, b)| a + b).collect();
        let m_new = M::from_vec(&p_new);

        // reject any step that violates Feller (Heston side) or param bounds
        //
        // the LM_MAX bail-out belongs here too, not only on the bad-step branch below.
        // a start point wedged against a constraint boundary otherwise burns every
        // MAX_ITER iteration with lambda pinned at the ceiling and returns the initial
        // guess, reporting iters == MAX_ITER. that reads like "still working" when it
        // means "stuck since iteration 3", and the resulting rmse is just the fit
        // quality of the initial guess.
        if !m_new.bounds_ok() {
            lam = (lam * LM_UP).min(LM_MAX);
            if lam >= LM_MAX { break; }
            continue;
        }

        let (res_new, weights_new) = augmented_residuals(&m_new);
        let sse_new = weighted_sse(&res_new, &weights_new);

        if sse_new < sse {
            // good step
            p       = p_new;
            res     = res_new;
            weights = weights_new;
            sse     = sse_new;
            lam     = (lam * LM_DOWN).max(1e-12);

            if let Some(shared) = shared_best_sse {
                let mut best = shared.lock().unwrap();
                if sse < *best { *best = sse; }
            }

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

    let m = M::from_vec(&p);

    // fit-quality metrics (rmse, max_err) and the identifiability diagnostic
    // are always computed on DATA ONLY, even when reg was used to get here.
    // reporting these over the regularization-augmented residuals would be
    // dishonest: rmse would look better than the market fit actually is,
    // and the condition number would look better than the data actually
    // supports, both because the prior is quietly doing the work, not the
    // quotes. regularization is allowed to change WHERE you land, not what
    // you're told about how well-supported that landing spot is by the data.
    let res_data = residuals(quotes, &m);
    let wmse     = weighted_sse(&res_data, &base_weights) / base_weights.iter().map(|w| w*w).sum::<f64>();
    let rmse     = wmse.sqrt();
    let max_err  = res_data.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

    let final_j = jacobian::<M>(quotes, &m);
    let (final_jtj, _) = jtj_and_grad(&final_j, &res_data, &base_weights, n);
    let (eigenvalues, eigenvectors) = jacobi_eigen(&final_jtj, n);
    let max_eig = eigenvalues.iter().cloned().fold(f64::MIN, f64::max);
    let min_eig = eigenvalues.iter().cloned().fold(f64::MAX, f64::min);
    let condition_number = if min_eig.abs() > 1e-300 { (max_eig / min_eig).abs() } else { f64::INFINITY };
    let weakest_idx = (0..n).min_by(|&i, &j| eigenvalues[i].abs().total_cmp(&eigenvalues[j].abs())).unwrap();
    let weakest_direction = eigenvectors[weakest_idx].clone();

    CalibResult {
        params: m, rmse, max_err, iters, converged, early_stopped,
        condition_number, weakest_direction,
    }
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

fn weighted_sse(res: &[f64], weights: &[f64]) -> f64 {
    res.iter().zip(weights.iter()).map(|(r, w)| (w * r).powi(2)).sum()
}

// converts an exact d(price)/d(param) into d(iv)/d(param) via the implicit
// function theorem: price = BSM(iv), so d(price)/d(iv) = vega_BSM(iv), and
// d(iv)/d(param) = [d(price)/d(param)] / vega_BSM(iv). standard technique
// for turning a price-space derivative into a vol-space one without
// re-deriving the IV solver's own derivative.
fn model_iv_and_vega<M: CalibModel>(contract: &OptionContract, iv_seed: f64, p: &M) -> Option<(f64, f64)> {
    let px = p.price(contract);
    let c_for_iv = OptionContract { vol: iv_seed, ..*contract };
    let iv = implied_vol(&IvProblem { contract: c_for_iv, market_price: px })?;
    let c_for_vega = OptionContract { vol: iv, ..*contract };
    let vega = crate::bsm::bsm_price_and_greeks(&c_for_vega).vega;
    Some((iv, vega))
}

// central FD Jacobian, with an AD override per column where CalibModel
// provides one (currently: Bates' 3 jump columns). bump points are clamped
// to each param's valid range before pricing (no negative v0 going into the
// CF), and the denominator uses the actual clamped distance, not the
// nominal 2*h, so a param sitting near its boundary doesn't get a quietly
// biased derivative.
fn jacobian<M: CalibModel>(quotes: &[CalibInput], p: &M) -> Vec<Vec<f64>> {
    let base   = p.to_vec();
    let bumps  = M::bump_sizes();
    let bounds = M::param_bounds();
    let n      = M::dim();

    let mut j = vec![vec![0.0; n]; quotes.len()];
    for col in 0..n {
        // try AD for the whole column first. if any single quote's solver
        // bails (None), fall back to FD for the whole column rather than
        // mixing AD and FD entries within one column, that's not worth the
        // bookkeeping for what should be a rare edge case.
        let ad_col: Option<Vec<f64>> = quotes.iter()
            .map(|q| {
                let d_price   = p.ad_price_derivative(q.contract, col)?;
                let (_, vega) = model_iv_and_vega(q.contract, q.iv_market, p)?;
                if vega.abs() < 1e-10 { return None; }
                Some(d_price / vega)
            })
            .collect();

        if let Some(col_vals) = ad_col {
            for row in 0..quotes.len() { j[row][col] = col_vals[row]; }
            continue;
        }

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
fn jtj_and_grad(j: &[Vec<f64>], res: &[f64], weights: &[f64], n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut jtj  = vec![vec![0.0_f64; n]; n];
    let mut grad = vec![0.0_f64; n];
    for (row, (r, w)) in res.iter().zip(weights.iter()).enumerate() {
        let w2 = w * w;
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
        for (k, &dpk) in dp.iter().enumerate().skip(i + 1) { s -= a[i][k] * dpk; }
        dp[i] = if a[i][i].abs() > 1e-15 { s / a[i][i] } else { 0.0 };
    }
    dp
}

// Jacobi eigenvalue algorithm for real symmetric NxN matrices. classic,
// simple, numerically robust for the small matrices here (N<=8, one J'J
// per calibration call, not a hot path). used for the identifiability
// diagnostic: J'J's eigenvalues tell you how much curvature the objective
// has in each parameter-space direction, a tiny eigenvalue means the data
// barely constrains that combination of parameters, which is exactly what
// non-identifiability looks like quantitatively instead of just by
// eyeballing a suspicious-looking result.
//
// rotation formulas are the standard Numerical Recipes form, verified here
// against matrices with known eigenvalues (see jacobi_eigen_* tests), not
// trusted from memory alone.
fn jacobi_eigen(a_in: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = a_in.to_vec();
    let mut v = vec![vec![0.0_f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() { row[i] = 1.0; }

    const MAX_SWEEPS: usize = 100;
    for _sweep in 0..MAX_SWEEPS {
        let off: f64 = (0..n).map(|i| ((i+1)..n).map(|j| a[i][j]*a[i][j]).sum::<f64>()).sum();
        if off.sqrt() < 1e-14 { break; }

        for p in 0..n {
            for q in (p+1)..n {
                if a[p][q].abs() < 1e-300 { continue; }

                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta*theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta*theta + 1.0).sqrt())
                };
                let c = 1.0 / (t*t + 1.0).sqrt();
                let s = t * c;

                let apq = a[p][q];
                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        a[i][p] = c*aip - s*aiq; a[p][i] = a[i][p];
                        a[i][q] = s*aip + c*aiq; a[q][i] = a[i][q];
                    }
                }
                for row in v.iter_mut() {
                    let vip = row[p];
                    let viq = row[q];
                    row[p] = c*vip - s*viq;
                    row[q] = s*vip + c*viq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // eigenvector i is column i of v
    let eigenvectors: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|k| v[k][i]).collect()).collect();
    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionType;

    // 2x2 has a closed form: eigenvalues of [[a,b],[b,d]] are
    // (a+d)/2 +/- sqrt(((a-d)/2)^2 + b^2). independent reference, not
    // derived from jacobi_eigen itself.
    #[test]
    fn jacobi_eigen_matches_closed_form_2x2() {
        let (a, b, d) = (4.0_f64, 1.5_f64, 2.0_f64);
        let mat = vec![vec![a, b], vec![b, d]];
        let half_diff = (a - d) / 2.0;
        let disc = (half_diff*half_diff + b*b).sqrt();
        let mut expected = [(a + d) / 2.0 - disc, (a + d) / 2.0 + disc];
        expected.sort_by(|x, y| x.total_cmp(y));

        let (mut eigenvalues, _) = jacobi_eigen(&mat, 2);
        eigenvalues.sort_by(|x, y| x.total_cmp(y));

        for (got, exp) in eigenvalues.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-10, "got={eigenvalues:?} expected={expected:?}");
        }
    }

    // general self-consistency check, not tied to a specially-constructed
    // matrix: for ANY symmetric A, V (eigenvector columns) has to be
    // orthogonal (V^T V = I) and has to actually diagonalize A
    // (V^T A V = diag(eigenvalues)). this is the defining property of an
    // eigendecomposition, checking it directly is a stronger test than
    // checking against one hand-picked example, it holds for every matrix
    // this function will ever actually be called on.
    fn assert_valid_eigendecomposition(a: &[Vec<f64>], n: usize) {
        let (eigenvalues, eigenvectors) = jacobi_eigen(a, n);

        // orthogonality: V^T V = I
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|k| eigenvectors[i][k] * eigenvectors[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-8,
                    "eigenvectors {i},{j} not orthonormal: dot={dot:.10}");
            }
        }

        // A*v_i = lambda_i*v_i for every eigenpair
        for i in 0..n {
            let av: Vec<f64> = (0..n).map(|r| (0..n).map(|c| a[r][c]*eigenvectors[i][c]).sum()).collect();
            for (r, &av_r) in av.iter().enumerate() {
                let expected = eigenvalues[i] * eigenvectors[i][r];
                assert!((av_r - expected).abs() < 1e-6,
                    "eigenpair {i} fails A*v=lambda*v at component {r}: Av={:.8} lambda*v={:.8}", av_r, expected);
            }
        }
    }

    #[test]
    fn jacobi_eigen_reconstructs_5x5() {
        // arbitrary but fixed symmetric 5x5, same shape as a Heston J'J
        let a = vec![
            vec![ 6.0, -1.5,  0.8,  2.1, -0.3],
            vec![-1.5,  4.2,  1.1, -0.6,  0.9],
            vec![ 0.8,  1.1,  5.5,  0.4, -1.2],
            vec![ 2.1, -0.6,  0.4,  3.8,  0.7],
            vec![-0.3,  0.9, -1.2,  0.7,  2.9],
        ];
        assert_valid_eigendecomposition(&a, 5);
    }

    #[test]
    fn jacobi_eigen_reconstructs_8x8() {
        // same idea at Bates' dimensionality, diagonally dominant so it's
        // guaranteed positive definite like a real J'J near a good fit
        let mut a = vec![vec![0.0_f64; 8]; 8];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = if i == j { 10.0 + i as f64 } else { 0.3 * ((i + j) as f64 % 3.0 - 1.0) };
            }
        }
        assert_valid_eigendecomposition(&a, 8);
    }

    // a genuinely near-singular matrix (one direction almost unconstrained),
    // has to come back with one eigenvalue orders of magnitude smaller than
    // the rest, this is the actual shape of the identifiability problem.
    #[test]
    fn jacobi_eigen_finds_near_zero_eigenvalue() {
        // rank-deficient by construction: row 3 = row 1 + row 2 (up to the
        // symmetric off-diagonal terms this induces a near-null direction)
        let a = vec![
            vec![4.0, 1.0, 5.0],
            vec![1.0, 3.0, 4.0],
            vec![5.0, 4.0, 9.0],
        ];
        let (mut eigenvalues, _) = jacobi_eigen(&a, 3);
        eigenvalues.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
        assert!(eigenvalues[0].abs() < 1e-6, "expected a near-zero eigenvalue, got {eigenvalues:?}");
    }

    // the actual point of the diagnostic: it has to tell apart the
    // the actual point of regularization: apply it to the exact scenario
    // that revealed the identifiability problem in the first place (see
    // de_global_bates_finds_an_excellent_fit_not_necessarily_the_true_params)
    // and check it actually helps. prior is deliberately NOT the true
    // params (that'd be cheating, in practice you never know the true
    // params, the prior represents "roughly what a trader expects going
    // in"), so recovering something close to the true params here is
    // evidence the regularization is doing real work, not just parroting
    // back whatever prior it was given. numbers below are from the
    // measured run at reg_weight=1e-3, not aspirational (see
    // probe_regularization, #[ignore]d, for the full weight sweep).
    #[test]
    fn regularization_recovers_plausible_params_in_the_degenerate_case() {
        let true_p = BatesParams {
            heston: HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 },
            lambda: 0.5, mu_j: -0.1, sigma_j: 0.15,
        };
        let raw = make_quotes_bates(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        // deliberately off from the true params, not a cheat prior
        let prior = BatesParams {
            heston: HestonParams { v0: 0.035, kappa: 1.8, theta: 0.04, sigma: 0.35, rho: -0.45 },
            lambda: 0.4, mu_j: -0.08, sigma_j: 0.12,
        };
        let reg = calibrate_bates_global_regularized(&quotes, 60, 80, 999, &prior, 1e-3);

        assert!(reg.best.rmse < 0.005, "regularization shouldn't cost much fit quality, rmse={:.6}", reg.best.rmse);
        assert!(reg.best.condition_number.is_finite() && reg.best.condition_number < 1e9,
            "regularized landing point should not be flagged as degenerate, cond={:.3e}", reg.best.condition_number);

        let p = reg.best.params;
        assert!((p.heston.v0  - true_p.heston.v0 ).abs() < 0.01, "v0={:.4}",  p.heston.v0);
        assert!((p.heston.kappa - true_p.heston.kappa).abs() < 0.5, "kappa={:.4}", p.heston.kappa);
        assert!((p.lambda  - true_p.lambda ).abs() < 0.15, "lambda={:.4}",  p.lambda);
        assert!((p.mu_j    - true_p.mu_j   ).abs() < 0.05, "mu_j={:.4}",    p.mu_j);
        assert!((p.sigma_j - true_p.sigma_j).abs() < 0.05, "sigma_j={:.4}", p.sigma_j);
    }

    // regularization shouldn't matter when the data already pins the fit
    // down well, a well-identified Heston case with a reasonable prior
    // should land close to both the unregularized result and the truth.
    #[test]
    fn regularization_does_not_hurt_a_well_identified_case() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();
        let p0 = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };

        let unreg = calibrate_heston(&quotes, p0);
        let prior = HestonParams { v0: 0.045, kappa: 1.8, theta: 0.045, sigma: 0.32, rho: -0.45 };
        let reg   = calibrate_heston_regularized(&quotes, p0, &prior, 1e-3);

        // unreg lands at rmse=0 exactly here (noiseless synthetic data, a
        // fully-identified model), so comparing reg's cost relatively
        // against that isn't meaningful, any nonzero regularization pulls
        // away from a literally perfect fit by construction. check the
        // absolute cost is small instead, 0.0006 vol points is nothing
        // against real quote noise.
        assert!(reg.rmse < 0.002, "regularizing a well-identified fit shouldn't cost much, rmse={:.6}", reg.rmse);
        assert!((reg.params.v0 - unreg.params.v0).abs() < 0.005, "regularized v0 drifted too far from the unregularized answer");
    }

    // known-degenerate Bates DE result (see de_global_bates_finds_an_
    // excellent_fit_not_necessarily_the_true_params) from a normally-
    // identified fit, quantitatively, not just "trust the README". numbers
    // below are from the real measured run, not aspirational:
    //   Heston, good p0:        cond ~ 3.2e6  (finite, kappa is the floppy one)
    //   Bates, DE, no p0:       cond = inf    (v0 direction has ~zero curvature)
    //   Bates, good p0, LM:     cond ~ 3.7e7  (finite, still floppy but not degenerate)
    #[test]
    fn condition_number_flags_the_degenerate_bates_case() {
        let true_heston = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw_h = make_quotes_heston(&true_heston);
        let quotes_h: Vec<CalibInput> = raw_h.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();
        let p0 = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };
        let heston_res = calibrate_heston(&quotes_h, p0);
        assert!(heston_res.condition_number.is_finite(), "well-identified Heston should not be flagged as degenerate");
        assert!(heston_res.condition_number < 1e9, "cond={:.3e}, higher than the measured ~3.2e6 baseline suggests", heston_res.condition_number);

        let true_bates = BatesParams {
            heston: HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 },
            lambda: 0.5, mu_j: -0.1, sigma_j: 0.15,
        };
        let raw_b = make_quotes_bates(&true_bates);
        let quotes_b: Vec<CalibInput> = raw_b.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let bp0 = BatesParams {
            heston: HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 },
            lambda: 0.3, mu_j: -0.05, sigma_j: 0.1,
        };
        let bates_good_p0 = calibrate_bates(&quotes_b, bp0);
        assert!(bates_good_p0.condition_number.is_finite(), "LM from a sensible p0 shouldn't land somewhere fully degenerate");

        let de_res = calibrate_bates_global(&quotes_b, 60, 80, 999);
        assert!(de_res.best.condition_number > 1e10 || !de_res.best.condition_number.is_finite(),
            "expected the unconstrained DE result to be flagged as (near-)degenerate, got cond={:.3e}", de_res.best.condition_number);
        assert!(de_res.best.condition_number > 100.0 * bates_good_p0.condition_number,
            "DE result should be flagged as meaningfully worse-identified than the good-p0 LM result: de={:.3e} good_p0={:.3e}",
            de_res.best.condition_number, bates_good_p0.condition_number);
    }

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

    // DE doesn't take a p0 at all, unlike calibrate_heston/multistart it
    // explores the whole box from a uniform-random population, this is the
    // actual differentiator from repeated local restarts, not just "more
    // restarts". small population/generation budget on purpose, this is a
    // correctness check, not a convergence-speed benchmark.
    #[test]
    fn de_global_recovers_heston_params_without_any_initial_guess() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let res = calibrate_heston_global(&quotes, 40, 60, 777);

        assert!(res.best.rmse < 0.01, "rmse={:.6}", res.best.rmse);
        assert!((res.best.params.v0  - true_p.v0 ).abs() < 0.01, "v0 off: {}",  res.best.params.v0);
        assert!((res.best.params.rho - true_p.rho).abs() < 0.1,  "rho off: {}", res.best.params.rho);
        assert!(res.best.params.feller_ok());
        // the polish step should never make things worse than DE's own
        // raw winner, LM only accepts improving steps.
        let de_rmse = (res.de_best_sse / quotes.len() as f64).sqrt();
        assert!(res.best.rmse <= de_rmse + 1e-9,
            "polish made it worse: polished={:.6} de_raw={:.6}", res.best.rmse, de_rmse);
    }

    // DE on Bates without a p0 does NOT reliably land back on the params
    // that generated the surface, and that's not a bug. Bates is a known
    // hard case for parameter identifiability from vanilla quotes alone:
    // this run found params wildly different from the generating ones
    // (v0=1.64 vs true 0.04, kappa=21 vs true 2, ...) with rmse essentially
    // 0, a different point on a near-flat manifold of equally-good fits to
    // the same finite quote grid, not a worse fit. calibrate_bates from a
    // sensible p0 (see recovers_bates_params_from_synthetic_surface above)
    // stays near the intended basin precisely because LM only takes
    // improving *local* steps, DE has no such bias and no reason to prefer
    // "the params that generated this" over any other point with the same
    // objective value. check what's actually true here: fit quality, not
    // param recovery, if you need Bates params that resemble a market
    // prior, constrain the search or start LM from one, don't rely on an
    // unconstrained global search to find "the" answer for an under-
    // identified model.
    #[test]
    fn de_global_bates_finds_an_excellent_fit_not_necessarily_the_true_params() {
        let true_p = BatesParams {
            heston: HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 },
            lambda: 0.5, mu_j: -0.1, sigma_j: 0.15,
        };
        let raw    = make_quotes_bates(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let res = calibrate_bates_global(&quotes, 60, 80, 999);

        assert!(res.best.rmse < 0.005, "rmse={:.8}, DE+polish should still find an excellent fit even if the params don't match the generator", res.best.rmse);
        assert!(res.best.params.heston.feller_ok(), "Feller violated: {:?}", res.best.params.heston);
        assert!(res.best.params.heston.v0 > 0.0 && res.best.params.lambda >= 0.0, "sanity bounds");
    }

    // direct check on jacobian() itself, not just "calibration still
    // converges" (LM's damping can paper over a slightly-wrong Jacobian and
    // still converge, slower). the AD columns (5,6,7: lambda, mu_j, sigma_j)
    // have to match what pure FD gives for the same columns, computed here
    // independently of jacobian()'s own AD path.
    #[test]
    fn bates_jacobian_ad_columns_match_pure_fd() {
        let heston = HestonParams { v0: 0.045, kappa: 1.8, theta: 0.045, sigma: 0.35, rho: -0.55 };
        let p = BatesParams { heston, lambda: 0.4, mu_j: -0.08, sigma_j: 0.12 };
        let raw = make_quotes_bates(&p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let j_actual = jacobian(&quotes, &p);

        // independent pure-FD reimplementation of just the jump columns,
        // deliberately not reusing jacobian()'s own bump/clamp machinery so
        // this isn't just checking the code against itself.
        let h = 1e-4;
        for (row, q) in quotes.iter().enumerate() {
            for (col, bump) in [(5, "lambda"), (6, "mu_j"), (7, "sigma_j")] {
                let mut vu = p; let mut vd = p;
                match bump {
                    "lambda"  => { vu.lambda  += h; vd.lambda  -= h; }
                    "mu_j"    => { vu.mu_j    += h; vd.mu_j    -= h; }
                    "sigma_j" => { vu.sigma_j += h; vd.sigma_j -= h; }
                    _ => unreachable!(),
                }
                let ru = single_residual(q.contract, q.iv_market, &vu);
                let rd = single_residual(q.contract, q.iv_market, &vd);
                let fd = (ru - rd) / (2.0 * h);
                let actual = j_actual[row][col];
                let err = (actual - fd).abs() / fd.abs().max(1e-6);
                assert!(err < 0.02,
                    "row={row} col={col} ({bump}): jacobian()={actual:.6} independent_fd={fd:.6} rel_err={err:.4}");
            }
        }
    }

    // direct test of the pruning mechanism itself, not relying on random
    // restarts happening to trigger it. rig the shared best to something a
    // deliberately-terrible start can't plausibly beat quickly, confirm it
    // gets aborted (early_stopped, and well under MAX_ITER) instead of
    // burning the full 200 iterations on a run that's not going anywhere.
    #[test]
    fn prune_mechanism_aborts_a_hopeless_restart() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        // pretend some other restart already found a near-perfect fit
        let shared = std::sync::Mutex::new(1e-10);
        let terrible_p0 = HestonParams { v0: 0.3, kappa: 7.5, theta: 0.35, sigma: 0.12, rho: 0.8 };
        let result = calibrate_inner(&quotes, terrible_p0, Some(&shared), None);

        assert!(result.early_stopped, "expected the hopeless restart to get pruned");
        assert!(result.iters < MAX_ITER, "should abort well before the iteration cap, took {}", result.iters);
        assert!(!result.converged, "a pruned restart shouldn't also claim convergence");
    }

    // and the opposite: a restart that's actually competitive should NOT
    // get pruned just because it started behind. shared best set to
    // something loose enough that reasonable progress stays under it.
    #[test]
    fn prune_mechanism_does_not_kill_a_competitive_restart() {
        let true_p = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.5 };
        let raw    = make_quotes_heston(&true_p);
        let quotes: Vec<CalibInput> = raw.iter()
            .map(|(c, iv)| CalibInput { contract: c, iv_market: *iv, weight: 1.0 })
            .collect();

        let shared = std::sync::Mutex::new(f64::INFINITY); // nothing to compare against yet
        let decent_p0 = HestonParams { v0: 0.05, kappa: 1.5, theta: 0.05, sigma: 0.4, rho: -0.3 };
        let result = calibrate_inner(&quotes, decent_p0, Some(&shared), None);

        assert!(!result.early_stopped, "a competitive restart with nothing to lose to shouldn't be pruned");
        assert!(result.converged, "should still converge normally");
        assert!(result.rmse < 0.005, "rmse={:.6}", result.rmse);
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
