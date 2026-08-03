// Monte Carlo pricer for Heston/Bates path-dependent payoffs.
//
// for anything vanilla, use heston_price/bates_price instead, they're exact
// (up to quadrature tolerance) and orders of magnitude cheaper than
// simulating paths. this exists for the stuff those can't touch: Asian
// averages, barriers, anything where the payoff depends on the path and
// not just S_T.
//
// variance process: full truncation Euler (Lord, Koekkoek, van Dijk 2010).
// simpler than the Andersen QE scheme and correct, which matters more than
// being clever here. if bias at large steps ever becomes a real problem,
// switch to QE, don't hand-tune this one into something QE-shaped.
//
// jumps: exact Poisson draw per step (Knuth's algorithm), not the "coin
// flip with probability lambda*dt" shortcut. that shortcut silently drops
// the (small but nonzero) probability of 2+ jumps in one step. a sum of n
// i.i.d. N(mu_j, sigma_j^2) draws is itself N(n*mu_j, n*sigma_j^2), so
// handling n>1 costs one extra branch, not a nested loop.

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use crate::types::{HestonParams, OptionType};

#[derive(Debug, Clone, Copy)]
pub struct McResult {
    pub price: f64,
    pub std_error: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct McConfig {
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: u64,
    pub antithetic: bool,
}

impl Default for McConfig {
    fn default() -> Self {
        Self { n_paths: 200_000, n_steps: 252, seed: 0xC0FFEE, antithetic: true }
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

pub fn mc_bates(
    spot: f64, expiry: f64, rate: f64, div_yield: f64,
    params: &HestonParams, lambda: f64, mu_j: f64, sigma_j: f64,
    payoff: Payoff, cfg: &McConfig,
) -> McResult {
    mc_price(spot, expiry, rate, div_yield, params, Some((lambda, mu_j, sigma_j)), payoff, cfg)
}

// one random draw per step: two correlated normals for (dS, dv), plus
// whatever jump landed this step (0.0 if none). storing this per step lets
// the antithetic sibling replay the exact same jumps with only the
// Brownian part negated, jump signs aren't symmetric so mirroring them
// would bias, not reduce, variance.
#[derive(Clone, Copy)]
struct StepDraw { z1: f64, z2: f64, jump_sum: f64 }

fn draw_step<R: Rng>(rng: &mut R, rho: f64, jumps: Option<(f64, f64, f64)>, dt: f64) -> StepDraw {
    let (za, zb) = box_muller(rng);
    let z1 = za;
    let z2 = rho * za + (1.0 - rho * rho).sqrt() * zb;

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
    StepDraw { z1, z2, jump_sum }
}

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
                    .map(|_| draw_step(&mut rng, hp.rho, jumps, dt))
                    .collect();

                let payoff_main = run_path(spot, drift, hp, &steps, dt, sqdt, false, &payoff, rate, expiry);
                local_sum += payoff_main;
                local_sum_sq += payoff_main * payoff_main;
                local_count += 1;

                if cfg.antithetic {
                    let payoff_anti = run_path(spot, drift, hp, &steps, dt, sqdt, true, &payoff, rate, expiry);
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
) -> f64 {
    let mut ln_s = spot.ln();
    let mut v    = hp.v0;
    let mut running_sum = 0.0;
    let mut running_max = spot;

    for step in steps {
        let (z1, z2) = if antithetic { (-step.z1, -step.z2) } else { (step.z1, step.z2) };
        let v_pos = v.max(0.0);

        ln_s += (drift - 0.5*v_pos) * dt + v_pos.sqrt() * sqdt * z1 + step.jump_sum;
        v    += hp.kappa * (hp.theta - v_pos) * dt + hp.sigma * v_pos.sqrt() * sqdt * z2;
        v     = v.max(0.0); // full truncation

        let s = ln_s.exp();
        running_sum += s;
        running_max = running_max.max(s);
    }

    let terminal = ln_s.exp();
    let disc = (-rate * expiry).exp();
    disc * payoff.eval(terminal, running_sum, running_max, steps.len())
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
        McConfig { n_paths: 60_000, n_steps: 100, seed: 7, antithetic: true }
    }

    // European MC has to agree with the analytic Heston price within a few
    // standard errors. this is the right test methodology for MC, not a
    // fixed absolute tolerance, MC output is a random variable.
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
}
