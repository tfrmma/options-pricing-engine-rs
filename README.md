# options-pricing-engine-rs

Low-latency options pricing engine in Rust. BSM, Black-76, Heston, Bates (stochastic vol + jumps), Local Vol (Dupire). Full analytic Greeks. Fast IV solver.

Built to run in production on a vol surface update cycle, not a toy.

## Models

| Model | Method | Greeks |
|---|---|---|
| Black-Scholes-Merton | Analytic | Full (Δ, Γ, ν, Θ, ρ, vanna, volga) |
| Black-76 | Analytic | Full |
| Heston (1993) | Albrecher stable CF + GK-15 quadrature | Bump-and-reprice |
| Bates (1996) | Heston + Merton jumps CF | Bump-and-reprice |
| Local Vol (Dupire) | Fritsch-Butland spline + finite diff | Numerical |

## Performance (release, single core)

```
BSM    chain of 500:  ~0.4ms   (~1300 opts/ms)
Heston chain of 500:  ~3.7ms   (~135 opts/ms)
```

Parallel batch via `rayon` set `RAYON_NUM_THREADS` or let it use all cores.

## Build

```bash
# dev
cargo build

# prod use this
RUSTFLAGS="-C target-cpu=native" cargo build --release

# run smoke test + timing
cargo run --release

# tests
cargo test
```

## Quick usage

```rust
use options_pricing_engine::*;

// BSM price + full Greeks
let contract = OptionContract {
    spot: 100.0, strike: 100.0, expiry: 1.0,
    rate: 0.05, div_yield: 0.02, vol: 0.20,
    opt_type: OptionType::Call,
};
let result = bsm_price_and_greeks(&contract);
println!("price={:.4} delta={:.4}", result.price, result.delta);

// Implied vol
let iv = implied_vol(&IvProblem { contract, market_price: 9.5 });

// Heston price + Greeks
let params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
let px = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);
let gr = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);

// Bates (Heston + Merton jumps)
let bparams = BatesParams {
    heston: params,
    lambda: 0.5, mu_j: -0.10, sigma_j: 0.15,
};
let px = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &bparams, OptionType::Call);
let gr = bates_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &bparams, OptionType::Call);

// Heston calibration to market IVs
let quotes: Vec<CalibInput> = /* (contract, iv_market, weight) triples */;
let p0 = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.4, rho: -0.5 };
let res = calibrate_heston(&quotes, p0);
println!("rmse={:.4} converged={}", res.rmse, res.converged);

// Arbitrage check + repair on local vol surface
let mut surf = LocalVolSurface::new(strikes, expiries, ivs);
let audit = check_and_repair_surface(&mut surf);
println!("{} violations, {} repaired", audit.violations.len(), audit.repaired);
let lv = dupire_local_vol(&surf, 100.0, 0.03, 0.0, 2, 1);

// Batch pricing (parallel)
let chain: Vec<OptionContract> = /* ... */;
let prices = batch_bsm_price(&chain);
let ivs    = batch_implied_vol(&chain, &market_prices);
```

## Design notes

- No `Box<dyn Model>` in hot paths. Static dispatch, monomorphized.
- Flat `Vec<f64>` for surfaces indexed by `i_strike * n_expiry + j_expiry`.
- `HestonParams::feller_ok()` checks the `2κθ > σ²` condition before you calibrate into nonsense.
- IV solver: Brenner-Subrahmanyam initial guess → Halley iterations (3rd-order convergence). Falls back to bisection for extreme cases.
- Heston/Bates CF: Albrecher (2007) stable formulation, no branch-cut discontinuities. `stable_cf` and `gk_integrate` live in `heston.rs` and are shared with `bates.rs` no duplication.
- Local vol: Fritsch-Butland monotone cubic splines on the IV surface to keep derivatives well-behaved. Raw FD on noisy data will give you negative local vols. Don't do that.
- Local vol FD: `d²w/dK²` uses the correct non-uniform grid denominator `(dp²+dm²)/2`. The symmetric form `((dp+dm)/2)²` is wrong for non-uniform strike spacing.
- Heston/Bates Greeks: bump-and-reprice. 14 pricing calls per option. AD would be faster; add it when it matters.
- Calibration: LM over implied vols, not prices. Fitting in vol space weights wings and ATM equally — fitting in price space overweights ITM by ~10x.
- `ncdf`: delegates to `libm::erfc`. Full double precision in the tails (~1e-15 vs ~1.5e-7 for the old A&S approximation).

## Known limitations / TODO

- Monte Carlo pricer not included — add `rayon` MC loops for exotics when needed.
- Heston/Bates Greeks are bump-and-reprice. Complex-step or AD would be ~10x faster for full surfaces.
- Calibration has no global optimizer LM finds a local minimum. If your initial guess is far off, perturb and retry.
- Local vol arbitrage repair is single-pass and conservative. Badly broken surfaces may need multiple passes.

## Crate dependencies

```
num-complex  — complex arithmetic for CF inversion
rayon        — parallel batch pricing
libm         — erfc for full-precision ncdf
```

That's it. No ndarray, no nalgebra, no heavy deps unless you need them.
