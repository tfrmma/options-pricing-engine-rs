// Monte Carlo pricer for Heston/Bates path-dependent payoffs.
//
// for anything vanilla, use heston_price/bates_price instead, they're exact
// (up to quadrature tolerance) and orders of magnitude cheaper than
// simulating paths. this exists for the stuff those can't touch: Asian
// averages, barriers, anything where the payoff depends on the path and
// not just S_T.
//
// variance process: full truncation Euler (Lord, Koekkoek, van Dijk 2010) by
// default, biased at large steps for low-vol-of-vol or near-zero-variance
// paths. VarianceScheme::QuadraticExponential (Andersen 2008) is available
// via McConfig::scheme when that bias matters, samples v_{t+dt} from a
// moment-matched distribution instead of discretizing and truncating the
// SDE. costs a bit more per step, only worth it if you've actually checked
// the bias matters for your case, see qe_reduces_bias_in_feller_violating_regime.
//
// jumps: exact Poisson draw per step (Knuth's algorithm), not the "coin
// flip with probability lambda*dt" shortcut. that shortcut silently drops
// the (small but nonzero) probability of 2+ jumps in one step. a sum of n
// i.i.d. N(mu_j, sigma_j^2) draws is itself N(n*mu_j, n*sigma_j^2), so
// handling n>1 costs one extra branch, not a nested loop.

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use crate::types::{HestonParams, OptionType, RoughBergomiParams, ForwardVarianceCurve};

#[derive(Debug, Clone, Copy)]
pub struct McResult {
    pub price: f64,
    pub std_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarianceScheme {
    // Lord, Koekkoek, van Dijk (2010). simple, correct, biased at large
    // steps for low-vol-of-vol or near-zero-variance paths (see module note).
    FullTruncationEuler,
    // Andersen (2008). samples v_{t+dt} from a moment-matched distribution
    // (quadratic for high psi, exponential-mixture for low psi) instead of
    // discretizing the SDE, eliminates the truncation bias above. costs an
    // ncdf() call in the high-psi branch and is a bit more arithmetic per
    // step either way, use it when the bias actually matters for your case
    // (short-dated, low vol-of-vol, or anywhere close to a Feller violation).
    QuadraticExponential,
}

#[derive(Debug, Clone, Copy)]
pub struct McConfig {
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: u64,
    pub antithetic: bool,
    pub scheme: VarianceScheme,
}

impl Default for McConfig {
    fn default() -> Self {
        Self {
            n_paths: 200_000, n_steps: 252, seed: 0xC0FFEE, antithetic: true,
            scheme: VarianceScheme::FullTruncationEuler, // unchanged default, see module note
        }
    }
}

#[derive(Clone, Copy)]
pub enum Payoff {
    European { strike: f64, opt_type: OptionType },
    AsianArithmetic { strike: f64, opt_type: OptionType },
    // rebate paid immediately on breach, most desks use 0.0
    UpAndOut { strike: f64, barrier: f64, rebate: f64, opt_type: OptionType },
}

#[inline]
fn intrinsic(s: f64, k: f64, opt_type: OptionType) -> f64 {
    match opt_type {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put  => (k - s).max(0.0),
    }
}

impl Payoff {
    #[inline]
    fn eval(&self, terminal: f64, running_sum: f64, running_max: f64, n_steps: usize) -> f64 {
        match *self {
            Payoff::European { strike, opt_type } => intrinsic(terminal, strike, opt_type),
            Payoff::AsianArithmetic { strike, opt_type } => {
                intrinsic(running_sum / n_steps as f64, strike, opt_type)
            }
            Payoff::UpAndOut { strike, barrier, rebate, opt_type } => {
                if running_max >= barrier { rebate } else { intrinsic(terminal, strike, opt_type) }
            }
        }
    }
}

// no jumps: lambda=0 collapses jump_sum to 0 every step, same code path as Bates
pub fn mc_heston(
    spot: f64, expiry: f64, rate: f64, div_yield: f64,
    params: &HestonParams, payoff: Payoff, cfg: &McConfig,
) -> McResult {
    mc_price(spot, expiry, rate, div_yield, params, None, payoff, cfg)
}

// standard pricer-call shape plus jump params, config, and payoff spec.
#[allow(clippy::too_many_arguments)]
pub fn mc_bates(
    spot: f64, expiry: f64, rate: f64, div_yield: f64,
    params: &HestonParams, lambda: f64, mu_j: f64, sigma_j: f64,
    payoff: Payoff, cfg: &McConfig,
) -> McResult {
    mc_price(spot, expiry, rate, div_yield, params, Some((lambda, mu_j, sigma_j)), payoff, cfg)
}

// one random draw per step: two INDEPENDENT standard normals, plus
// whatever jump landed this step (0.0 if none). Euler mixes them into a
// rho-correlated pair itself; QE needs one independent normal to drive the
// variance step and a SEPARATE independent one for the price residual (eq
// 33 of Andersen 2008 bakes the S-V correlation into K1/K2 analytically,
// it does NOT use a shared rho-correlated shock the way Euler does), so
// mixing happens per-scheme inside run_path, not here.
//
// storing this per step lets the antithetic sibling replay the exact same
// jumps with only the Brownian part negated, jump signs aren't symmetric
// so mirroring them would bias, not reduce, variance.
#[derive(Clone, Copy)]
struct StepDraw { za: f64, zb: f64, jump_sum: f64 }

fn draw_step<R: Rng>(rng: &mut R, jumps: Option<(f64, f64, f64)>, dt: f64) -> StepDraw {
    let (za, zb) = box_muller(rng);

    let jump_sum = match jumps {
        None => 0.0,
        Some((lambda, mu_j, sigma_j)) => {
            let n = sample_poisson(rng, lambda * dt);
            if n == 0 {
                0.0
            } else {
                let (u, _) = box_muller(rng);
                mu_j * n as f64 + sigma_j * (n as f64).sqrt() * u
            }
        }
    };
    StepDraw { za, zb, jump_sum }
}

#[allow(clippy::too_many_arguments)]
fn mc_price(
    spot: f64, expiry: f64, rate: f64, div_yield: f64,
    hp: &HestonParams, jumps: Option<(f64, f64, f64)>,
    payoff: Payoff, cfg: &McConfig,
) -> McResult {
    let dt      = expiry / cfg.n_steps as f64;
    let sqdt    = dt.sqrt();
    let k_bar   = jumps.map_or(0.0, |(_, mu_j, sigma_j)| (mu_j + 0.5*sigma_j*sigma_j).exp() - 1.0);
    let lambda  = jumps.map_or(0.0, |(l, _, _)| l);
    let drift   = rate - div_yield - lambda * k_bar;

    let n_threads = rayon::current_num_threads().max(1);
    let paths_per_pair = if cfg.antithetic { 2 } else { 1 };
    let n_pairs_total = cfg.n_paths / paths_per_pair;
    let chunk_size = (n_pairs_total / n_threads).max(1);

    // (sum of payoffs, sum of payoffs^2, count) per chunk, reduced at the end
    let (sum, sum_sq, count): (f64, f64, usize) = (0..n_threads)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end   = if chunk_idx == n_threads - 1 { n_pairs_total } else { (start + chunk_size).min(n_pairs_total) };
            if start >= end { return (0.0, 0.0, 0usize); }

            let mut rng = SmallRng::seed_from_u64(splitmix64(cfg.seed ^ chunk_idx as u64));
            let mut local_sum = 0.0;
            let mut local_sum_sq = 0.0;
            let mut local_count = 0usize;

            for _ in start..end {
                let steps: Vec<StepDraw> = (0..cfg.n_steps)
                    .map(|_| draw_step(&mut rng, jumps, dt))
                    .collect();

                let payoff_main = run_path(spot, drift, hp, &steps, dt, sqdt, false, &payoff, rate, expiry, cfg.scheme);
                local_sum += payoff_main;
                local_sum_sq += payoff_main * payoff_main;
                local_count += 1;

                if cfg.antithetic {
                    let payoff_anti = run_path(spot, drift, hp, &steps, dt, sqdt, true, &payoff, rate, expiry, cfg.scheme);
                    local_sum += payoff_anti;
                    local_sum_sq += payoff_anti * payoff_anti;
                    local_count += 1;
                }
            }
            (local_sum, local_sum_sq, local_count)
        })
        .reduce(|| (0.0, 0.0, 0usize), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    let n = count as f64;
    let mean = sum / n;
    // sample variance of the (discounted) payoffs, std error = sd / sqrt(n).
    // antithetic pairs are negatively correlated by construction so this
    // slightly overstates the true error, conservative in the right direction.
    let var = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (var / n).sqrt();

    McResult { price: mean, std_error }
}

#[allow(clippy::too_many_arguments)]
fn run_path(
    spot: f64, drift: f64, hp: &HestonParams, steps: &[StepDraw],
    dt: f64, sqdt: f64, antithetic: bool, payoff: &Payoff, rate: f64, expiry: f64,
    scheme: VarianceScheme,
) -> f64 {
    let mut ln_s = spot.ln();
    let mut v    = hp.v0;
    let mut running_sum = 0.0;
    let mut running_max = spot;

    // Andersen (2008) eq (33) coefficients for the QE price update, central
    // discretization (gamma1=gamma2=0.5, his own recommendation, used in
    // his own numerical tests in section 5). computed once, dt is fixed
    // across steps. verified against the primary source PDF, not a
    // secondary writeup, K1/K2 in particular are easy to transcribe wrong.
    //   K0 = -rho*kappa*theta*dt/sigma
    //   K1 = gamma1*dt*(kappa*rho/sigma - 0.5) - rho/sigma
    //   K2 = gamma2*dt*(kappa*rho/sigma - 0.5) + rho/sigma
    //   K3 = gamma1*dt*(1-rho^2)
    //   K4 = gamma2*dt*(1-rho^2)
    let (k0, k1, k2, k3, k4) = if scheme == VarianceScheme::QuadraticExponential {
        let rho = hp.rho;
        let (g1, g2) = (0.5, 0.5);
        let k0 = -rho * hp.kappa * hp.theta * dt / hp.sigma;
        let k1 = g1*dt*(hp.kappa*rho/hp.sigma - 0.5) - rho/hp.sigma;
        let k2 = g2*dt*(hp.kappa*rho/hp.sigma - 0.5) + rho/hp.sigma;
        let k3 = g1*dt*(1.0 - rho*rho);
        let k4 = g2*dt*(1.0 - rho*rho);
        (k0, k1, k2, k3, k4)
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

    for step in steps {
        let (za, zb) = if antithetic { (-step.za, -step.zb) } else { (step.za, step.zb) };
        let v_pos = v.max(0.0);

        match scheme {
            VarianceScheme::FullTruncationEuler => {
                // unchanged since before QE existed: rho-correlated pair,
                // pre-step variance drives both drift and diffusion terms.
                let z1 = za;
                let z2 = hp.rho*za + (1.0 - hp.rho*hp.rho).sqrt()*zb;
                ln_s += (drift - 0.5*v_pos) * dt + v_pos.sqrt() * sqdt * z1 + step.jump_sum;
                v = (v + hp.kappa*(hp.theta - v_pos)*dt + hp.sigma*v_pos.sqrt()*sqdt*z2).max(0.0);
            }
            VarianceScheme::QuadraticExponential => {
                // za drives the variance step (mapped through ncdf when the
                // high-psi branch fires). zb is the INDEPENDENT residual eq
                // (33) needs for the price innovation, correlation with V
                // is already baked into K1/K2 analytically, reusing a
                // rho-mixed shock here (the Euler approach) would be wrong,
                // that's the bug the first version of this had.
                let v_new = qe_variance_step(v_pos, hp.kappa, hp.theta, hp.sigma, dt, za);
                let var_term = (k3*v_pos + k4*v_new).max(0.0);
                ln_s += drift*dt + k0 + k1*v_pos + k2*v_new + var_term.sqrt()*zb + step.jump_sum;
                v = v_new;
            }
        }

        let s = ln_s.exp();
        running_sum += s;
        running_max = running_max.max(s);
    }

    let terminal = ln_s.exp();
    let disc = (-rate * expiry).exp();
    disc * payoff.eval(terminal, running_sum, running_max, steps.len())
}

// --- rough Bergomi -----------------------------------------------------
//
// structurally different from mc_price/run_path above: the far-kernel term
// isn't a per-step recurrence, it's one FFT convolution over the whole path
// (rbergomi.rs::simulate_variance_and_dz), so this can't reuse draw_step /
// run_path as-is. RNG generation stays here (mc.rs already owns box_muller),
// everything downstream of the raw normals is a pure function in rbergomi.rs.

// 4 iid N(0,1) per step: 3 feed the Cholesky-correlated (W^n_i, W^n_i1,
// W^n_i2) block, 1 is the independent W-perp shock for Z. same
// precompute-then-replay-negated pattern as StepDraw, needed for antithetic
// to actually pair against the same underlying draws instead of fresh ones.
struct RbergomiDraws { z0: Vec<f64>, z1: Vec<f64>, z2: Vec<f64>, zperp: Vec<f64> }

fn draw_rbergomi_path<R: Rng>(rng: &mut R, n_steps: usize) -> RbergomiDraws {
    let mut z0 = vec![0.0; n_steps];
    let mut z1 = vec![0.0; n_steps];
    let mut z2 = vec![0.0; n_steps];
    let mut zperp = vec![0.0; n_steps];
    for i in 0..n_steps {
        let (a, b) = box_muller(rng);
        z0[i] = a; z1[i] = b;
        let (c, d) = box_muller(rng);
        z2[i] = c; zperp[i] = d;
    }
    RbergomiDraws { z0, z1, z2, zperp }
}

#[allow(clippy::too_many_arguments)]
fn run_rbergomi_path(
    spot: f64, drift0: f64, v: &[f64], dz: &[f64], dt: f64,
    payoff: &Payoff, rate: f64, expiry: f64,
) -> f64 {
    let mut ln_s = spot.ln();
    let mut running_sum = 0.0;
    let mut running_max = spot;

    for i in 0..v.len() {
        let vi = v[i].max(0.0); // v is exp(...) > 0 by construction, max(0.0) is just a guard against a garbage curve/param combo upstream
        ln_s += (drift0 - 0.5 * vi) * dt + vi.sqrt() * dz[i];
        let s = ln_s.exp();
        running_sum += s;
        running_max = running_max.max(s);
    }

    let terminal = ln_s.exp();
    let disc = (-rate * expiry).exp();
    disc * payoff.eval(terminal, running_sum, running_max, v.len())
}

// no jump term, rBergomi doesn't have one in the paper's formulation and
// nothing here asked for one, add a Bates-style jump-diffusion overlay as
// its own commit if that's ever needed rather than bolting it on here.
#[allow(clippy::too_many_arguments)]
pub fn mc_rough_bergomi(
    spot: f64, expiry: f64, rate: f64, div_yield: f64,
    params: &RoughBergomiParams, curve: &ForwardVarianceCurve,
    payoff: Payoff, cfg: &McConfig,
) -> McResult {
    let dt = expiry / cfg.n_steps as f64;
    let n  = 1.0 / dt;
    let alpha = params.alpha();
    let drift0 = rate - div_yield;

    // built once, shared read-only across every path below, not per-path,
    // see rbergomi.rs's module note on why that matters here specifically.
    let sigma = crate::rbergomi::hybrid_scheme_covariance(alpha, 2, n);
    let l = crate::rbergomi::cholesky_lower(&sigma, 3).unwrap_or_else(|| {
        panic!("hybrid scheme Sigma not positive definite for alpha={alpha}, hurst should be in (0, 0.5)")
    });
    let plan  = crate::rbergomi::build_conv_plan(cfg.n_steps);
    let gamma = crate::rbergomi::build_gamma(alpha, cfg.n_steps, n);

    let n_threads = rayon::current_num_threads().max(1);
    let paths_per_pair = if cfg.antithetic { 2 } else { 1 };
    let n_pairs_total = cfg.n_paths / paths_per_pair;
    let chunk_size = (n_pairs_total / n_threads).max(1);

    let (sum, sum_sq, count): (f64, f64, usize) = (0..n_threads)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end   = if chunk_idx == n_threads - 1 { n_pairs_total } else { (start + chunk_size).min(n_pairs_total) };
            if start >= end { return (0.0, 0.0, 0usize); }

            let mut rng = SmallRng::seed_from_u64(splitmix64(cfg.seed ^ chunk_idx as u64));
            let mut local_sum = 0.0;
            let mut local_sum_sq = 0.0;
            let mut local_count = 0usize;

            for _ in start..end {
                let draws = draw_rbergomi_path(&mut rng, cfg.n_steps);

                let (v, dz) = crate::rbergomi::simulate_variance_and_dz(
                    params, curve, dt, &l, &plan, &gamma,
                    &draws.z0, &draws.z1, &draws.z2, &draws.zperp, 1.0);
                let payoff_main = run_rbergomi_path(spot, drift0, &v, &dz, dt, &payoff, rate, expiry);
                local_sum += payoff_main;
                local_sum_sq += payoff_main * payoff_main;
                local_count += 1;

                if cfg.antithetic {
                    let (v2, dz2) = crate::rbergomi::simulate_variance_and_dz(
                        params, curve, dt, &l, &plan, &gamma,
                        &draws.z0, &draws.z1, &draws.z2, &draws.zperp, -1.0);
                    let payoff_anti = run_rbergomi_path(spot, drift0, &v2, &dz2, dt, &payoff, rate, expiry);
                    local_sum += payoff_anti;
                    local_sum_sq += payoff_anti * payoff_anti;
                    local_count += 1;
                }
            }
            (local_sum, local_sum_sq, local_count)
        })
        .reduce(|| (0.0, 0.0, 0usize), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    let n_f = count as f64;
    let mean = sum / n_f;
    let var = (sum_sq / n_f - mean * mean).max(0.0);
    let std_error = (var / n_f).sqrt();

    McResult { price: mean, std_error }
}

// Andersen (2008) quadratic-exponential step for the CIR variance process.
// samples v_{t+dt} from a distribution moment-matched to the true
// conditional distribution of the CIR process (quadratic-in-normal for
// psi <= psi_c, an exponential/point-mass mixture above), instead of
// discretizing the SDE and truncating negative excursions. z is the same
// correlated normal Euler would have used for this step's variance
// innovation, reused here (mapped through ncdf for the high-psi branch) so
// a single (z1,z2) draw per step still drives both schemes identically.
fn qe_variance_step(v_t: f64, kappa: f64, theta: f64, sigma: f64, dt: f64, z: f64) -> f64 {
    const PSI_C: f64 = 1.5; // Andersen recommends 1.0-2.0, 1.5 is the commonly used midpoint

    let ekt = (-kappa * dt).exp();
    let m   = theta + (v_t - theta) * ekt;
    let s2  = (v_t * sigma*sigma * ekt / kappa) * (1.0 - ekt)
            + (theta * sigma*sigma / (2.0*kappa)) * (1.0 - ekt).powi(2);
    let psi = (s2 / (m*m).max(1e-300)).max(1e-12);

    if psi <= PSI_C {
        let inv_psi = 1.0 / psi;
        let b2 = 2.0*inv_psi - 1.0 + (2.0*inv_psi * (2.0*inv_psi - 1.0)).max(0.0).sqrt();
        let a  = m / (1.0 + b2);
        let b  = b2.sqrt();
        a * (b + z).powi(2)
    } else {
        let p    = (psi - 1.0) / (psi + 1.0);
        let beta = (1.0 - p) / m.max(1e-300);
        let u    = crate::math::ncdf(z); // same normal, mapped to keep correlation with z1
        if u <= p { 0.0 } else { (1.0 / beta) * ((1.0 - p) / (1.0 - u)).ln() }
    }
}

// Box-Muller, one call gives two independent standard normals.
fn box_muller<R: Rng>(rng: &mut R) -> (f64, f64) {
    let u1: f64 = rng.gen_range(1e-12..1.0); // avoid ln(0)
    let u2: f64 = rng.gen();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

// Knuth's algorithm. exact, fine for the small means (lambda*dt) we call it with.
fn sample_poisson<R: Rng>(rng: &mut R, mean: f64) -> u32 {
    if mean <= 0.0 { return 0; }
    let l = (-mean).exp();
    let mut k = 0u32;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.gen::<f64>();
        if p <= l { return k - 1; }
    }
}

// seed decorrelation for per-chunk RNGs. sequential seeds into Xoshiro-family
// generators can share low-order state; hashing through splitmix64 first
// (the standard recommended seeding technique for that generator family)
// avoids it for the price of one extra function call at setup.
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heston::heston_price;
    use crate::bates::bates_price;

    fn params() -> HestonParams {
        HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 }
    }

    fn small_cfg() -> McConfig {
        McConfig { n_paths: 60_000, n_steps: 100, seed: 7, antithetic: true, scheme: VarianceScheme::FullTruncationEuler }
    }

    // European MC has to agree with the analytic Heston price within a few
    // standard errors. this is the right test methodology for MC, not a
    // fixed absolute tolerance, MC output is a random variable.
    // the actual point of porting QE: in a badly Feller-violating regime
    // (2*kappa*theta=0.4 << sigma^2=1.44) with a coarse step (8 steps/year,
    // where discretization bias is large enough to see clearly), QE's error
    // against the analytic price has to be meaningfully smaller than full
    // truncation Euler's, not just "different". thresholds below come from
    // the actual measured run (euler err ~1.78, qe err ~0.09, a ~20x
    // reduction), not aspirational numbers, verify with:
    //   cargo test --release -- --ignored --nocapture mc::tests::probe_qe_vs_euler_bias
    #[test]
    fn qe_reduces_bias_in_feller_violating_regime() {
        use crate::heston::heston_price;
        let p = HestonParams { v0: 0.04, kappa: 5.0, theta: 0.04, sigma: 1.2, rho: -0.7 };
        assert!(!p.feller_ok(), "test needs a genuinely Feller-violating case");
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let analytic = heston_price(s, k, t, r, q, &p, OptionType::Call);

        let cfg_euler = McConfig { n_paths: 200_000, n_steps: 8, seed: 42, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };
        let cfg_qe    = McConfig { n_paths: 200_000, n_steps: 8, seed: 42, antithetic: true, scheme: VarianceScheme::QuadraticExponential };

        let euler = mc_heston(s, t, r, q, &p, Payoff::European { strike: k, opt_type: OptionType::Call }, &cfg_euler);
        let qe    = mc_heston(s, t, r, q, &p, Payoff::European { strike: k, opt_type: OptionType::Call }, &cfg_qe);

        let euler_err = (euler.price - analytic).abs();
        let qe_err    = (qe.price - analytic).abs();

        assert!(qe_err < euler_err / 5.0,
            "expected QE to cut the bias by at least 5x here: analytic={analytic:.4} euler_err={euler_err:.4} qe_err={qe_err:.4}");
        assert!(qe_err < 10.0 * qe.std_error.max(0.05),
            "QE error should be within noise of the analytic price, err={qe_err:.4} se={:.4}", qe.std_error);
    }

    #[test]
    fn mc_heston_matches_analytic_european() {
        let p = params();
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let analytic = heston_price(s, k, t, r, q, &p, OptionType::Call);
        let mc = mc_heston(s, t, r, q, &p, Payoff::European { strike: k, opt_type: OptionType::Call }, &small_cfg());
        let z = (mc.price - analytic).abs() / mc.std_error;
        assert!(z < 4.0, "analytic={analytic:.4} mc={:.4} se={:.4} z={z:.2}", mc.price, mc.std_error);
    }

    #[test]
    fn mc_heston_qe_matches_analytic_european() {
        let p = params();
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let analytic = heston_price(s, k, t, r, q, &p, OptionType::Call);
        let mut cfg = small_cfg();
        cfg.scheme = VarianceScheme::QuadraticExponential;
        let mc = mc_heston(s, t, r, q, &p, Payoff::European { strike: k, opt_type: OptionType::Call }, &cfg);
        let z = (mc.price - analytic).abs() / mc.std_error;
        assert!(z < 4.0, "analytic={analytic:.4} mc={:.4} se={:.4} z={z:.2}", mc.price, mc.std_error);
    }

    #[test]
    fn mc_bates_matches_analytic_european() {
        let p = params();
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let (lambda, mu_j, sigma_j) = (0.5, -0.1, 0.15);
        let analytic = bates_price(
            s, k, t, r, q,
            &crate::types::BatesParams { heston: p, lambda, mu_j, sigma_j },
            OptionType::Call,
        );
        let mc = mc_bates(s, t, r, q, &p, lambda, mu_j, sigma_j,
            Payoff::European { strike: k, opt_type: OptionType::Call }, &small_cfg());
        let z = (mc.price - analytic).abs() / mc.std_error;
        assert!(z < 4.0, "analytic={analytic:.4} mc={:.4} se={:.4} z={z:.2}", mc.price, mc.std_error);
    }

    // arithmetic averaging can't exceed the terminal-value payoff's convexity,
    // Asian call <= European call, same strike, always. real inequality, not
    // a vibe check.
    #[test]
    fn asian_call_cheaper_than_european() {
        let p = params();
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let euro  = mc_heston(s, t, r, q, &p, Payoff::European { strike: k, opt_type: OptionType::Call }, &small_cfg());
        let asian = mc_heston(s, t, r, q, &p, Payoff::AsianArithmetic { strike: k, opt_type: OptionType::Call }, &small_cfg());
        assert!(asian.price < euro.price, "asian={:.4} european={:.4}", asian.price, euro.price);
    }

    // with rebate=0, up-and-out payoff <= vanilla payoff pathwise (0 if
    // breached, vanilla payoff otherwise), so the price ordering has to hold
    // too. barrier comfortably above spot so we get a mix of breached/not.
    #[test]
    fn up_and_out_cheaper_than_vanilla() {
        let p = params();
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let vanilla = mc_heston(s, t, r, q, &p, Payoff::European { strike: k, opt_type: OptionType::Call }, &small_cfg());
        let uao = mc_heston(s, t, r, q, &p,
            Payoff::UpAndOut { strike: k, barrier: 130.0, rebate: 0.0, opt_type: OptionType::Call }, &small_cfg());
        assert!(uao.price < vanilla.price, "uao={:.4} vanilla={:.4}", uao.price, vanilla.price);
        assert!(uao.price >= 0.0);
    }

    #[test]
    fn poisson_sampler_matches_mean() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mean = 0.3;
        let n = 200_000;
        let total: u64 = (0..n).map(|_| sample_poisson(&mut rng, mean) as u64).sum();
        let empirical_mean = total as f64 / n as f64;
        assert!((empirical_mean - mean).abs() < 0.01, "empirical={empirical_mean:.4} target={mean}");
    }

    fn rbergomi_small_cfg() -> McConfig {
        McConfig { n_paths: 100_000, n_steps: 64, seed: 3, antithetic: true, scheme: VarianceScheme::FullTruncationEuler }
    }

    // sequential (no rayon, this is a correctness benchmark not a hot path),
    // builds the exact joint covariance once, prices via the same
    // run_rbergomi_path Euler update the fast pricer uses, so the ONLY
    // thing under test when comparing against mc_rough_bergomi is the
    // variance-path construction (hybrid+FFT vs exact), not two different
    // price-integration schemes.
    #[allow(clippy::too_many_arguments)]
    fn mc_rough_bergomi_exact_reference(
        spot: f64, expiry: f64, rate: f64, div_yield: f64,
        params: &RoughBergomiParams, curve: &ForwardVarianceCurve,
        strike: f64, opt_type: OptionType, n_steps: usize, n_paths: usize, seed: u64,
    ) -> (f64, f64) {
        let dt = expiry / n_steps as f64;
        let times: Vec<f64> = (0..n_steps).map(|i| i as f64 * dt).collect();
        let alpha = params.alpha();
        let drift0 = rate - div_yield;
        let payoff = Payoff::European { strike, opt_type };

        let joint_cov = crate::rbergomi::exact_joint_covariance(alpha, &times, dt);
        let joint_dim = 2 * n_steps - 1;
        let l_joint = crate::rbergomi::cholesky_lower(&joint_cov, joint_dim)
            .expect("exact joint covariance should be positive definite");

        let mut rng = SmallRng::seed_from_u64(seed);
        let (mut sum, mut sum_sq) = (0.0, 0.0);

        for _ in 0..n_paths {
            let mut z = vec![0.0; joint_dim];
            for i in (0..joint_dim).step_by(2) {
                let (a, b) = box_muller(&mut rng);
                z[i] = a;
                if i + 1 < joint_dim { z[i + 1] = b; }
            }
            let mut zperp = vec![0.0; n_steps];
            for i in (0..n_steps).step_by(2) {
                let (a, b) = box_muller(&mut rng);
                zperp[i] = a;
                if i + 1 < n_steps { zperp[i + 1] = b; }
            }

            let (v, dz) = crate::rbergomi::simulate_price_exact(params, curve, &times, dt, &l_joint, &z, &zperp);
            let payoff_val = run_rbergomi_path(spot, drift0, &v, &dz, dt, &payoff, rate, expiry);
            sum += payoff_val;
            sum_sq += payoff_val * payoff_val;
        }

        let n_f = n_paths as f64;
        let mean = sum / n_f;
        let var = (sum_sq / n_f - mean * mean).max(0.0);
        (mean, (var / n_f).sqrt())
    }

    // benchmark against the TRUE continuous-time process's own covariance
    // (Lemma 4.3, same closed form as commit 1, no discretization at all
    // beyond the Cholesky factorization's own float error), not against
    // another instance of the same discretized scheme, that would be
    // circular. same spirit as the paper's own Figure 4, done here via
    // prices at a few strikes rather than inverted IVs, price agreement at
    // multiple strikes carries the same information as smile agreement
    // without pulling iv.rs into this specific check.
    #[test]
    fn rbergomi_fast_scheme_matches_exact_covariance_reference() {
        let params = RoughBergomiParams { eta: 1.9, rho: -0.9, hurst: 0.07 };
        let curve = ForwardVarianceCurve::new(vec![1.0], vec![0.09]);
        let (spot, expiry, rate, div_yield) = (100.0, 0.25, 0.02, 0.0);
        let n_steps = 40;
        let n_paths = 200_000;

        let fast_cfg = McConfig { n_paths, n_steps, seed: 7, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };

        for strike in [90.0, 100.0, 110.0] {
            let fast = mc_rough_bergomi(spot, expiry, rate, div_yield, &params, &curve,
                Payoff::European { strike, opt_type: OptionType::Call }, &fast_cfg);
            let (exact_price, exact_se) = mc_rough_bergomi_exact_reference(
                spot, expiry, rate, div_yield, &params, &curve,
                strike, OptionType::Call, n_steps, n_paths, 9);

            let combined_se = (fast.std_error.powi(2) + exact_se.powi(2)).sqrt();
            let z = (fast.price - exact_price).abs() / combined_se;
            assert!(z < 4.0, "strike={strike} fast={:.4} exact={exact_price:.4} combined_se={combined_se:.4} z={z:.2}",
                fast.price);
        }
    }

    // rough vol's whole reason to exist: ATM skew explodes as T->0 like
    // T^(H-1/2), unlike classical stochastic vol (Heston/Bates) where it
    // flattens out. checks the skew is bigger at the shorter maturity by
    // roughly the predicted ratio, not a tight regression fit, a finite-h
    // secant at finite T against a T->0 asymptotic result has real slack in
    // it on its own, a generous multiplicative band is the honest way to
    // state what this actually checks: the right sign, the right order of
    // magnitude, not three decimal places of an exponent.
    #[test]
    fn rbergomi_short_maturity_skew_explodes_like_the_theory_predicts() {
        let params = RoughBergomiParams { eta: 1.9, rho: -0.9, hurst: 0.07 };
        let alpha = params.alpha();
        let curve = ForwardVarianceCurve::new(vec![1.0], vec![0.09]);
        let (spot, rate, div_yield) = (100.0, 0.0, 0.0);
        let h = 0.03_f64; // log-moneyness half-width for the finite-difference skew

        let skew_at = |expiry: f64| -> f64 {
            let cfg = McConfig { n_paths: 300_000, n_steps: 32, seed: 13, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };
            let k_lo = spot * (-h).exp();
            let k_hi = spot * h.exp();

            let price_lo = mc_rough_bergomi(spot, expiry, rate, div_yield, &params, &curve,
                Payoff::European { strike: k_lo, opt_type: OptionType::Call }, &cfg).price;
            let price_hi = mc_rough_bergomi(spot, expiry, rate, div_yield, &params, &curve,
                Payoff::European { strike: k_hi, opt_type: OptionType::Call }, &cfg).price;

            let iv_lo = crate::iv::implied_vol(&crate::types::IvProblem {
                contract: crate::types::OptionContract { spot, strike: k_lo, expiry, rate, div_yield, vol: 0.3, opt_type: OptionType::Call },
                market_price: price_lo,
            }).expect("price_lo should invert");
            let iv_hi = crate::iv::implied_vol(&crate::types::IvProblem {
                contract: crate::types::OptionContract { spot, strike: k_hi, expiry, rate, div_yield, vol: 0.3, opt_type: OptionType::Call },
                market_price: price_hi,
            }).expect("price_hi should invert");

            (iv_hi - iv_lo) / (2.0 * h)
        };

        let (t1, t2) = (0.02, 0.16); // 8x apart
        let skew1 = skew_at(t1).abs();
        let skew2 = skew_at(t2).abs();

        let predicted_ratio = (t1 / t2).powf(alpha); // alpha = H - 1/2 < 0, shorter T => bigger skew
        let actual_ratio = skew1 / skew2;

        assert!(actual_ratio > 1.3 && actual_ratio < predicted_ratio * 1.8,
            "skew({t1})={skew1:.4} skew({t2})={skew2:.4} actual_ratio={actual_ratio:.2} predicted_ratio={predicted_ratio:.2}");
    }

    // (S_T - 0)^+ = S_T always (S_T > 0), so a strike=0 call's discounted
    // price is exactly E[disc * S_T], which has to equal S(0) by
    // construction of the stochastic exponential (the -1/2 v ds compensator
    // in the paper's S(t) formula), for any eta/rho/hurst/curve. real
    // martingale check on the full near+far+FFT pipeline together, not a
    // vibe check on any one piece in isolation.
    #[test]
    fn rbergomi_forward_is_a_martingale() {
        let params = RoughBergomiParams { eta: 1.9, rho: -0.9, hurst: 0.07 };
        let curve = ForwardVarianceCurve::new(vec![0.5, 1.0, 2.0], vec![0.2, 0.25, 0.22]);
        let (s, t, r, q) = (100.0, 0.5, 0.0, 0.0);

        let mc = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::European { strike: 0.0, opt_type: OptionType::Call }, &rbergomi_small_cfg());

        let z = (mc.price - s).abs() / mc.std_error;
        assert!(z < 4.0, "price={:.4} spot={s} se={:.4} z={z:.2}", mc.price, mc.std_error);
    }

    // eta -> 0 collapses v(t) toward the deterministic curve xi0(t) (the
    // stochastic term in the exponential vanishes), so the price should
    // converge to plain Black-Scholes with vol = sqrt(xi0). cross-check
    // against a completely different, already-tested pricer (bsm.rs), not
    // against rBergomi's own machinery.
    #[test]
    fn rbergomi_matches_black_scholes_when_vol_of_vol_is_tiny() {
        let params = RoughBergomiParams { eta: 0.001, rho: -0.9, hurst: 0.07 };
        let xi0 = 0.04;
        let curve = ForwardVarianceCurve::new(vec![2.0], vec![xi0]);
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);

        let bs = crate::bsm::bsm_price(&crate::types::OptionContract {
            spot: s, strike: k, expiry: t, rate: r, div_yield: q,
            vol: xi0.sqrt(), opt_type: OptionType::Call,
        });

        let mc = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::European { strike: k, opt_type: OptionType::Call }, &rbergomi_small_cfg());

        let z = (mc.price - bs).abs() / mc.std_error;
        assert!(z < 4.0, "bs={bs:.4} mc={:.4} se={:.4} z={z:.2}", mc.price, mc.std_error);
    }

    // Asian/UpAndOut share run_rbergomi_path's running_sum/running_max
    // accumulation with European (which ignores both), but that sharing
    // was never actually checked, a bug in either one would sit invisible
    // behind passing European tests forever. same pathwise inequality as
    // the Heston/Bates versions above (real inequality, not a vibe check),
    // ported to rBergomi specifically.
    #[test]
    fn rbergomi_asian_cheaper_than_european() {
        let params = RoughBergomiParams { eta: 1.9, rho: -0.9, hurst: 0.07 };
        let curve = ForwardVarianceCurve::new(vec![1.0], vec![0.09]);
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);

        let euro = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::European { strike: k, opt_type: OptionType::Call }, &rbergomi_small_cfg());
        let asian = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::AsianArithmetic { strike: k, opt_type: OptionType::Call }, &rbergomi_small_cfg());

        assert!(asian.price < euro.price, "asian={:.4} european={:.4}", asian.price, euro.price);
    }

    #[test]
    fn rbergomi_up_and_out_cheaper_than_vanilla() {
        let params = RoughBergomiParams { eta: 1.9, rho: -0.9, hurst: 0.07 };
        let curve = ForwardVarianceCurve::new(vec![1.0], vec![0.09]);
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);

        let vanilla = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::European { strike: k, opt_type: OptionType::Call }, &rbergomi_small_cfg());
        let uao = mc_rough_bergomi(s, t, r, q, &params, &curve,
            Payoff::UpAndOut { strike: k, barrier: 130.0, rebate: 0.0, opt_type: OptionType::Call }, &rbergomi_small_cfg());

        assert!(uao.price < vanilla.price, "uao={:.4} vanilla={:.4}", uao.price, vanilla.price);
        assert!(uao.price >= 0.0);
    }

    // stronger than the pathwise inequalities above: two INDEPENDENTLY
    // implemented path accumulators (Heston's simple SDE loop, rBergomi's
    // hybrid-scheme+FFT variance path feeding the same Euler update) have
    // to land on the same VALUE, not just the same ordering, when their
    // stochastic-vol content is switched off. this is what actually
    // exercises run_rbergomi_path's running_sum/running_max, not just
    // whether Asian < European.
    #[test]
    fn rbergomi_asian_matches_heston_when_vol_of_vol_is_tiny() {
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let cfg = McConfig { n_paths: 100_000, n_steps: 64, seed: 11, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };

        let rb_params = RoughBergomiParams { eta: 0.001, rho: -0.9, hurst: 0.07 };
        let curve = ForwardVarianceCurve::new(vec![2.0], vec![0.04]);
        let rb = mc_rough_bergomi(s, t, r, q, &rb_params, &curve,
            Payoff::AsianArithmetic { strike: k, opt_type: OptionType::Call }, &cfg);

        let heston_params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.001, rho: -0.5 };
        let heston = mc_heston(s, t, r, q, &heston_params,
            Payoff::AsianArithmetic { strike: k, opt_type: OptionType::Call }, &cfg);

        let combined_se = (rb.std_error.powi(2) + heston.std_error.powi(2)).sqrt();
        let z = (rb.price - heston.price).abs() / combined_se;
        assert!(z < 4.0, "rbergomi={:.4} heston={:.4} se={combined_se:.4} z={z:.2}", rb.price, heston.price);
    }

    #[test]
    fn rbergomi_up_and_out_matches_heston_when_vol_of_vol_is_tiny() {
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let cfg = McConfig { n_paths: 100_000, n_steps: 64, seed: 13, antithetic: true, scheme: VarianceScheme::FullTruncationEuler };

        let rb_params = RoughBergomiParams { eta: 0.001, rho: -0.9, hurst: 0.07 };
        let curve = ForwardVarianceCurve::new(vec![2.0], vec![0.04]);
        let rb = mc_rough_bergomi(s, t, r, q, &rb_params, &curve,
            Payoff::UpAndOut { strike: k, barrier: 130.0, rebate: 0.0, opt_type: OptionType::Call }, &cfg);

        let heston_params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.001, rho: -0.5 };
        let heston = mc_heston(s, t, r, q, &heston_params,
            Payoff::UpAndOut { strike: k, barrier: 130.0, rebate: 0.0, opt_type: OptionType::Call }, &cfg);

        let combined_se = (rb.std_error.powi(2) + heston.std_error.powi(2)).sqrt();
        let z = (rb.price - heston.price).abs() / combined_se;
        assert!(z < 4.0, "rbergomi={:.4} heston={:.4} se={combined_se:.4} z={z:.2}", rb.price, heston.price);
    }
}
