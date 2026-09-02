// synthetic (theta -> IV grid) dataset generator for the future rough
// Bergomi deep-calibration surrogate (Bayer, Horvath, Muguruza, Stemper,
// Tomas 2019, "image-based" approach: fixed strike/maturity grid, not
// per-sample strikes).
//
// scope for this first version: flat xi0 (a single scalar), not the full
// ForwardVarianceCurve term structure. the real forward variance curve is
// bootstrapped from market ATM quotes (rbergomi::bootstrap_forward_variance_curve),
// not calibrated, so the surrogate only needs to learn the (eta, rho,
// hurst) -> IV map GIVEN a curve level, not the curve shape itself. a
// production surrogate that has to generalize across very different curve
// shapes (steep backwardation vs flat) would need the curve as extra
// input dimensions, that's future work, not this commit, flagged in the
// README, not silently assumed away.
//
// output: CSV, self-describing header (column names encode the grid
// points directly, no separate metadata file to keep in sync). failed IV
// inversions are written as NaN, not dropped or guessed at, so a training
// script can filter them explicitly instead of silently training on rows
// that were never real.
//
// usage: cargo run --release --bin gen_rbergomi_dataset -- [n_samples] [output_path] [seed]
// defaults: 20 samples, rbergomi_dataset.csv, seed 42. 20 is a
// verification-scale run, not a training-scale one, see the README note
// on how far this was actually run before committing.

use options_pricing_engine::*;
use options_pricing_engine::mc::{mc_rough_bergomi, McConfig, Payoff, VarianceScheme};
use options_pricing_engine::iv::implied_vol;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use std::io::Write;

// crypto-short-dated on purpose, not the equity-index tenors the paper
// itself uses, this whole module exists for BTC/ETH pre-event surfaces.
const MATURITIES_DAYS: [f64; 5] = [3.0, 7.0, 14.0, 30.0, 90.0];
// standard-deviation multiples, not a fixed percentage moneyness. first
// version used a fixed +-0.40 log-moneyness at every maturity and 30% of
// grid points failed to invert, concentrated exactly at short T + wide
// moneyness: a 3-day option with ~20% vol has an expected move of about
// 2%, a strike 40% away is priced at noise-level, not a real number worth
// training on. scaling by sqrt(xi0*T) keeps every maturity's grid at a
// comparable, actually-priceable distance from ATM.
const Z_SCORES: [f64; 7] = [-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5];

const XI0_RANGE:   (f64, f64) = (0.01, 0.25);  // vol 10%-50%
const ETA_RANGE:   (f64, f64) = (0.5, 4.0);
const RHO_RANGE:   (f64, f64) = (-0.95, -0.05);
const HURST_RANGE: (f64, f64) = (0.02, 0.45);

fn sample_uniform<R: Rng>(rng: &mut R, range: (f64, f64)) -> f64 {
    range.0 + rng.gen::<f64>() * (range.1 - range.0)
}

// path resolution scaled to maturity, roughly one step per day, floored
// so the shortest (3-day) maturity doesn't get discretized too coarsely.
fn steps_for_maturity(expiry_years: f64) -> usize {
    (expiry_years * 365.0).round().max(16.0) as usize
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_samples: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let output_path = args.get(2).cloned().unwrap_or_else(|| "rbergomi_dataset.csv".to_string());
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut file = std::fs::File::create(&output_path)
        .unwrap_or_else(|e| panic!("could not create {output_path}: {e}"));

    write!(file, "xi0,eta,rho,hurst").unwrap();
    for &d in &MATURITIES_DAYS {
        for &z in &Z_SCORES {
            write!(file, ",iv_T{d:.0}d_Z{z:+.2}").unwrap();
        }
    }
    writeln!(file).unwrap();

    let mut rng = SmallRng::seed_from_u64(seed);
    let spot = 1.0; // normalized, rate=div_yield=0 throughout, log-moneyness is ln(K/spot) directly
    let mut n_failed_inversions = 0usize;
    let mut n_total_points = 0usize;

    for i in 0..n_samples {
        let xi0   = sample_uniform(&mut rng, XI0_RANGE);
        let eta   = sample_uniform(&mut rng, ETA_RANGE);
        let rho   = sample_uniform(&mut rng, RHO_RANGE);
        let hurst = sample_uniform(&mut rng, HURST_RANGE);

        let params = RoughBergomiParams { eta, rho, hurst };
        let curve = ForwardVarianceCurve::new(vec![1.0], vec![xi0]); // flat, one pillar covers the whole grid (max maturity 90d < 1y)

        write!(file, "{xi0:.6},{eta:.6},{rho:.6},{hurst:.6}").unwrap();

        for &d in &MATURITIES_DAYS {
            let expiry = d / 365.0;
            let cfg = McConfig {
                n_paths: 20_000, n_steps: steps_for_maturity(expiry),
                seed: seed ^ (i as u64) ^ (d as u64).wrapping_mul(0x9E3779B97F4A7C15),
                antithetic: true, scheme: VarianceScheme::FullTruncationEuler,
            };
            let move_scale = xi0.sqrt() * expiry.sqrt(); // approx expected log-move for this maturity

            for &z in &Z_SCORES {
                let strike = spot * (z * move_scale).exp();
                let mc = mc_rough_bergomi(spot, expiry, 0.0, 0.0, &params, &curve,
                    Payoff::European { strike, opt_type: OptionType::Call }, &cfg);

                let iv = implied_vol(&IvProblem {
                    contract: OptionContract { spot, strike, expiry, rate: 0.0, div_yield: 0.0, vol: 0.5, opt_type: OptionType::Call },
                    market_price: mc.price,
                });

                n_total_points += 1;
                match iv {
                    Some(v) => write!(file, ",{v:.6}").unwrap(),
                    None => { write!(file, ",NaN").unwrap(); n_failed_inversions += 1; }
                }
            }
        }
        writeln!(file).unwrap();

        if (i + 1) % 5 == 0 || i + 1 == n_samples {
            eprintln!("{}/{n_samples} samples done ({} failed inversions so far)", i + 1, n_failed_inversions);
        }
    }

    eprintln!("wrote {output_path}: {n_samples} samples, {n_total_points} grid points, {n_failed_inversions} failed inversions ({:.2}%)",
        100.0 * n_failed_inversions as f64 / n_total_points as f64);
}
