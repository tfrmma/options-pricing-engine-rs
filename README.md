# options-pricing-engine-rs

Low-latency options pricing engine in Rust. BSM, Black-76, Heston, Bates (stochastic vol + jumps), Local Vol (Dupire). Full analytic Greeks. Fast IV solver.

Built to run in production on a vol surface update cycle — not a toy.

## Models

| Model | Method | Greeks |
|---|---|---|
| Black-Scholes-Merton | Analytic | Full (Δ, Γ, ν, Θ, ρ, vanna, volga) |
| Black-76 | Analytic | Full |
| Heston (1993) | Albrecher stable CF + GK-15 quadrature | Via AD or FD |
| Bates (1996) | Heston + Merton jumps CF | Via AD or FD |
| Local Vol (Dupire) | Fritsch-Butland spline + finite diff | Numerical |

## Performance (release, single core)

```
BSM    chain of 500:  ~0.4ms   (~1300 opts/ms)
Heston chain of 500:  ~3.7ms   (~135 opts/ms)
```

Parallel batch via `rayon` — set `RAYON_NUM_THREADS` or let it use all cores.

## Build

```bash
# dev
cargo build

# prod — use this
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

// Heston
let params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
let px = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);

// Bates (Heston + Merton jumps)
let bparams = BatesParams {
    heston: params,
    lambda: 0.5, mu_j: -0.10, sigma_j: 0.15,
};
let px = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &bparams, OptionType::Call);

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
- Heston/Bates CF: Albrecher (2007) stable formulation — no branch-cut discontinuities.
- Local vol: Fritsch-Butland monotone cubic splines on the IV surface to keep derivatives well-behaved. Raw FD on noisy data will give you negative local vols. Don't do that.

## Known limitations / TODO

- Greeks for Heston/Bates are not yet implemented (need AD or bump-and-reprice).
- Calibration routines (Levenberg-Marquardt on Heston params) not included.
- Monte Carlo pricer not included — add `rayon` MC loops for exotics when needed.
- Local vol surface needs arbitrage check (calendar spread + butterfly conditions) on input.

## Crate dependencies

```
num-complex  — complex arithmetic for CF inversion
rayon        — parallel batch pricing
```

That's it. No ndarray, no nalgebra, no heavy deps unless you need them.
