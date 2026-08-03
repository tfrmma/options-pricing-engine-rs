# options-pricing-engine-rs

A Rust options pricing library covering Black-Scholes-Merton, Black-76, Heston (1993), Bates (1996), and Dupire local volatility, with full analytic Greeks where closed forms exist, a Halley-iteration implied vol solver, Levenberg-Marquardt calibration (single-start and multistart) for both Heston and Bates, no-arbitrage surface repair, and a Monte Carlo engine for path-dependent payoffs. Built for a vol surface update cycle, not a scripting exercise.

License: MIT. See [LICENSE](LICENSE).

## Contents

- [Models](#models)
- [Design](#design)
- [Build](#build)
- [Usage](#usage)
- [Testing](#testing)
- [Performance](#performance)
- [Known limitations and roadmap](#known-limitations-and-roadmap)
- [Dependencies](#dependencies)
- [References](#references)

## Models

| Model | Pricing method | Greeks |
|---|---|---|
| Black-Scholes-Merton | Closed form | Full analytic: Δ, Γ, ν, Θ, ρ, vanna, volga |
| Black-76 | Closed form | Full analytic |
| Heston (1993) | Albrecher et al. (2007) stable characteristic function, adaptive Gauss-Kronrod-15 quadrature | Bump-and-reprice (`heston_price_and_greeks`), or forward-mode automatic differentiation (`heston_greeks_ad`) |
| Bates (1996) | Heston CF × Merton (1976) log-normal jump CF | Bump-and-reprice (`bates_price_and_greeks`), or forward-mode AD (`bates_greeks_ad`) |
| Local Vol (Dupire 1994) | Fritsch-Butland monotone cubic spline, differentiated through the spline, not the raw grid | Numerical (local vol surface) |
| Monte Carlo (Heston/Bates) | Full truncation Euler, exact per-step Poisson jump counts, antithetic variates | N/A, path-dependent payoffs only (European, Asian, up-and-out barrier) |

All five analytic models share the same `OptionContract`/`PricingResult` conventions where applicable, so switching models in a caller doesn't mean rewriting the call site.

## Design

**Dispatch.** No `Box<dyn Model>` anywhere in a pricing path. Every model is a free function, statically dispatched and monomorphized. Model selection happens at the call site, not through a trait object indirection that shows up in a profiler.

**Memory layout.** `LocalVolSurface` stores `local_vols` as a flat `Vec<f64>` indexed `i_strike * n_expiry + j_expiry`, not `Vec<Vec<f64>>`. One allocation, contiguous, cache-friendly for the row/column sweeps the Dupire finite differences need.

**Characteristic function stability.** The Heston and Bates CF use the Albrecher et al. (2007) formulation, which removes the branch-cut discontinuities of the original 1993 formula. `stable_cf` and the adaptive Gauss-Kronrod integrator (`gk_integrate`) live in `heston.rs` and are shared, unmodified, by `bates.rs` and `ad.rs`. The quadrature is adaptive (substitution to a finite interval, panel subdivision until the Kronrod/Gauss error estimate is below tolerance), which matters most in the wings and at short expiries, where a fixed panel under-resolves the integrand and silently produces prices that violate static arbitrage bounds.

**Local vol.** `dvar_dk`, `dvar_dt`, and `d2var_dk2` differentiate the Fritsch-Butland monotone cubic spline (Fritsch & Butland, 1984), not the raw IV grid. Central differences on raw quotes amplify quote noise into negative local variances; sampling the spline at a symmetric offset around each node means the F-B overshoot limiter is actually load-bearing in the derivative, not just present for interpolation queries between nodes. Boundary nodes use a one-sided offset within the grid range, a symmetric step at the edge would sample outside `[strikes[0], strikes[n-1]]` and hit the spline's flat clamp, which silently halves the estimated slope.

**Surface repair.** `check_and_repair_surface` is multi-pass: it loops fixing calendar-spread and butterfly violations until the surface is clean or a pass cap is hit. Fixing one violation can create another next to it (bumping an IV to kill a calendar violation can turn a previously-fine butterfly into a violation), so a single pass is not sufficient on a surface with more than one problem.

**Automatic differentiation.** `heston_greeks_ad` and `bates_greeks_ad` propagate dual numbers (`Dual<f64>`, `Complex<Dual>`) through the characteristic function and integrate the derivative alongside the value via the Leibniz rule, giving exact vega/vanna without a finite-difference bump size to tune. The Bates version composes the Heston CF and the Merton jump CF *before* differentiating (same order `bates_call` uses, jumps bolted on after integration is mathematically wrong), so price and the five Heston-driven Greeks are exact through the jump-adjusted CF. Jump-parameter sensitivities (d/dλ, d/dμⱼ, d/dσⱼ) aren't computed by either path, out of scope for now, see [Known limitations](#known-limitations-and-roadmap). Both do fewer top-level integrations than bump-and-reprice, but this does not currently translate into a wall-clock win, see [Performance](#performance).

**Calibration.** `calibrate_heston`/`calibrate_bates` run Levenberg-Marquardt in implied-vol space, not price space. Fitting IV directly weights a 10-delta wing the same as an ATM quote; fitting price would overweight ITM options by roughly an order of magnitude relative to their information content about the smile. Both share one generic LM engine (`CalibModel` trait, Heston is a 5-parameter instance, Bates is 8) instead of two near-identical copies of the damping/Jacobian/Gauss-elimination machinery. `calibrate_heston_multistart`/`calibrate_bates_multistart` run several LM fits in parallel from randomized starting points (rayon, one thread per restart) and keep the best, a bad single initial guess converging to a bad local minimum no longer means a silently bad fit, it means one of several restarts landed there and got outvoted by RMSE. Still a local method under the hood, not CMA-ES or simulated annealing, more restarts help but don't guarantee the global optimum.

**Monte Carlo.** `mc_heston`/`mc_bates` simulate paths under full truncation Euler (Lord, Koekkoek, van Dijk 2010) for the variance process, correlated via the standard Cholesky-free two-normal construction. Bates jumps use an exact per-step Poisson draw (Knuth's algorithm), not the "coin flip with probability λdt" shortcut that silently drops the probability of two or more jumps landing in the same step. Parallelized over path chunks via rayon, with a `splitmix64`-hashed seed per chunk so parallel runs are reproducible and statistically independent. Use this for path-dependent payoffs the CF-inversion pricers can't touch (Asian averages, barriers), not for vanillas, `heston_price`/`bates_price` are exact and far cheaper for those.

**Numerics.** `ncdf` delegates to `libm::erfc`, full double precision through the tails (~1e-15), replacing the classical Abramowitz & Stegun 26.2.17 rational approximation the module used before, which has ~1.5e-7 error in the tails, enough to matter when solving implied vol on deep OTM quotes.

## Build

```bash
# dev build
cargo build

# release, with target-specific codegen (recommended for anything you're timing)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# smoke test + timing harness
cargo run --release

# full test suite
cargo test --release
```

## Usage

```rust
use options_pricing_engine::*;

// Black-Scholes-Merton
let contract = OptionContract {
    spot: 100.0, strike: 100.0, expiry: 1.0,
    rate: 0.05, div_yield: 0.02, vol: 0.20,
    opt_type: OptionType::Call,
};
let result = bsm_price_and_greeks(&contract);
println!("price={:.4} delta={:.4}", result.price, result.delta);

// Implied vol (Brenner-Subrahmanyam seed, Halley iteration, bisection fallback)
let iv = implied_vol(&IvProblem { contract, market_price: 9.5 });

// Heston: price, bump-and-reprice Greeks, or AD Greeks
let params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
let px    = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);
let gr    = heston_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);
let gr_ad = heston_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &params, OptionType::Call);

// Bates: Heston + Merton jumps
let bparams = BatesParams { heston: params, lambda: 0.5, mu_j: -0.10, sigma_j: 0.15 };
let px = bates_price(100.0, 100.0, 1.0, 0.05, 0.0, &bparams, OptionType::Call);
let gr = bates_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.0, &bparams, OptionType::Call);

// Heston calibration to a market IV surface, single-start or multistart
let quotes: Vec<CalibInput> = /* (contract, iv_market, weight) triples */;
let p0  = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.4, rho: -0.5 };
let res = calibrate_heston(&quotes, p0);
println!("rmse={:.4} converged={}", res.rmse, res.converged);

let multi = calibrate_heston_multistart(&quotes, p0, 8, 42);
println!("best rmse={:.4}, {}/{} restarts converged", multi.best.rmse, multi.n_converged, multi.n_restarts);

// Bates calibration, same engine, 8 params instead of 5
let bp0  = BatesParams { heston: p0, lambda: 0.5, mu_j: -0.1, sigma_j: 0.15 };
let bres = calibrate_bates(&quotes, bp0);

// AD Greeks for Bates, exact through the jump-adjusted CF
let gr_ad = bates_greeks_ad(100.0, 100.0, 1.0, 0.05, 0.0, &params, 0.5, -0.10, 0.15, OptionType::Call);

// Local vol: no-arbitrage repair, then Dupire
let mut surf = LocalVolSurface::new(strikes, expiries, ivs);
let audit = check_and_repair_surface(&mut surf);
println!("{} violations found, {} repaired", audit.violations.len(), audit.repaired);
let lv = dupire_local_vol(&surf, 100.0, 0.03, 0.0, 2, 1);

// Monte Carlo: path-dependent payoffs, European/Asian/up-and-out
let cfg = McConfig::default();
let asian = mc_heston(100.0, 1.0, 0.05, 0.0, &params,
    Payoff::AsianArithmetic { strike: 100.0, opt_type: OptionType::Call }, &cfg);
println!("asian price={:.4} +/- {:.4}", asian.price, asian.std_error);

let barrier = mc_bates(100.0, 1.0, 0.05, 0.0, &params, 0.5, -0.10, 0.15,
    Payoff::UpAndOut { strike: 100.0, barrier: 130.0, rebate: 0.0, opt_type: OptionType::Call }, &cfg);

// Batch pricing, parallel via rayon
let chain: Vec<OptionContract> = /* ... */;
let prices = batch_bsm_price(&chain);
let ivs    = batch_implied_vol(&chain, &market_prices);
```

## Testing

58 tests, `cargo test --release`, all synchronous and deterministic (no timing-dependent assertions, the Monte Carlo tests use a fixed seed and check convergence against the analytic price within a multiple of the MC's own reported standard error, not a fixed tolerance):

| Module | Tests | Covers |
|---|---:|---|
| `local_vol` | 13 | Flat-surface recovery, spline no-overshoot, calendar/butterfly detection and repair, multi-pass cascading repair, non-uniform grid curvature, spline-vs-raw-FD divergence on a kinked surface |
| `heston` | 8 | Put-call parity, sign checks, Feller condition, BSM limit (σ→0), no-static-arbitrage across a strike/expiry grid |
| `ad` | 9 | Heston and Bates price match their analytic pricers to 1e-6 across strikes and expiries including the short-dated wings that broke the old fixed-panel quadrature, vega within 1% of bump-and-reprice for both, Bates AD collapses to Heston AD when jumps are off, sign checks |
| `bates` | 6 | Recovers Heston when jump intensity is zero, put-call parity, sign checks, no-static-arbitrage |
| `calibration` | 4 | Recovers known Heston and Bates params from a synthetic surface, Feller condition always holds post-calibration, multistart never loses to single-start and reliably escapes a deliberately bad initial guess |
| `mc` | 5 | European MC matches analytic Heston/Bates within a z-score bound on the MC's own standard error, Asian call cheaper than European (real inequality), up-and-out cheaper than vanilla with zero rebate (real inequality), Poisson sampler mean check |
| `bsm` | 4 | Put-call parity, sign checks, Black-76 sanity vs BSM, Black-76 rho vs finite difference |
| `iv` | 4 | Round-trip recovery at ATM, OTM, and low vol, rejects a price outside no-arbitrage bounds |
| `batch` | 2 | Batch output matches scalar calls, batch IV round-trips |
| `math` | 3 | `ncdf` sanity and tail precision, `ncdf_inv` round-trip |

The arbitrage tests aren't incidental: `no_static_arbitrage` in `heston.rs` and the AD wings test exist specifically because a non-adaptive quadrature passed every other test in this suite while quietly producing arbitrage-violating prices at short expiries. A regression here is a real bug, not a tolerance nitpick.

## Performance

Throughput on batch pricing scales with core count through `rayon` (`RAYON_NUM_THREADS`, or default to all cores), so a single fixed "ops/ms" number is a property of whatever machine ran it, not of the code. Measure it on your own hardware:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

`main.rs::batch_bench` times a 500-option BSM and Heston chain and prints real numbers for whatever box it runs on. `main.rs::heston_ad_demo` does the same for bump-and-reprice vs AD Greeks on a single option, averaged over 2,000 reps. Treat both as a local baseline, not a spec.

Qualitatively: BSM is closed-form and embarrassingly parallel, Heston and Bates cost an adaptive double integral per price (more at short expiries and in the wings, where more panels are needed to hit tolerance), and `heston_greeks_ad` currently runs slower wall-clock than bump-and-reprice despite doing fewer integrations, `Complex<Dual>` arithmetic costs more per quadrature node than plain `Complex64`, see [Known limitations](#known-limitations-and-roadmap).

## Known limitations and roadmap

- `batch_heston`/`batch_bates` return price only, no `PricingResult`. `batch_bsm` returns full Greeks. Inconsistent API surface, loop it yourself with `rayon` if you need batch Heston/Bates Greeks today.
- `heston_greeks_ad`/`bates_greeks_ad` are exact but measurably slower than bump-and-reprice in this repo's own benchmark (see `main.rs::heston_ad_demo`), not a theoretical concern, a measured one. Worth profiling `Complex<Dual>` before recommending the AD path for anything latency-sensitive.
- Neither AD path computes jump-parameter sensitivities (d/dλ, d/dμⱼ, d/dσⱼ). Only price and the 5 Heston-driven Greeks are exact for Bates, if you need a jump-parameter gradient for a calibration Jacobian, that's a separate forward pass someone still has to write, `calibrate_bates`'s Jacobian uses FD for those columns, not AD.
- Multistart calibration (`calibrate_heston_multistart`/`calibrate_bates_multistart`) is repeated local optimization, not a real global optimizer. More restarts help but don't guarantee the global minimum, and every restart still costs a full LM run, there's no early-exit sharing of information between restarts.
- Monte Carlo (`mc_heston`/`mc_bates`) uses full truncation Euler, not the Andersen (2008) QE scheme. Simpler and correct, but biased at large time steps for very low-vol-of-vol or near-zero-variance paths, if that bias matters for your use case, increase `n_steps` or port QE. No Greeks from the MC path either, bump-and-reprice it yourself against `mc_heston`/`mc_bates` if you need them, that's expensive (each bump is a full path simulation) and noisy (each price has its own MC standard error), so no wrapper is provided.

## Dependencies

```
num-complex   complex arithmetic for characteristic function inversion
num-traits    trait bounds for Complex<Dual> in the AD path
rayon         parallel batch pricing, Monte Carlo paths, and multistart calibration
rand          RNG for the Monte Carlo engine (SmallRng, seeded per path chunk)
libm          erfc for full-precision ncdf
```

No `ndarray`, no `nalgebra`, no linear algebra crate, the calibration Jacobian is a 5x5 (Heston) or 8x8 (Bates) system solved by hand-rolled Gaussian elimination, not worth pulling in a dependency for at this size.

## References

- Black, F., Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Black, F. (1976). *The Pricing of Commodity Contracts.* (Black-76)
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.*
- Bates, D. S. (1996). *Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options.*
- Merton, R. C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous.*
- Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). *The Little Heston Trap.*
- Dupire, B. (1994). *Pricing with a Smile.*
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* (total-variance parametrization used by the local vol module)
- Fritsch, F. N., Carlson, R. E. (1980). *Monotone Piecewise Cubic Interpolation.*
- Fritsch, F. N., Butland, J. (1984). *A Method for Constructing Local Monotone Piecewise Cubic Interpolants.*
- Piessens, R., de Doncker-Kapenga, E., Uberhuber, C., Kahaner, D. (1983). *QUADPACK: A Subroutine Package for Automatic Integration.* (Gauss-Kronrod 15-point rule)
- Brenner, M., Subrahmanyam, M. G. (1988). *A Simple Formula to Compute the Implied Standard Deviation.*
- Levenberg, K. (1944); Marquardt, D. (1963). (Levenberg-Marquardt nonlinear least squares)
- Lord, R., Koekkoek, R., van Dijk, D. (2010). *A Comparison of Biased Simulation Schemes for Stochastic Volatility Models.* (full truncation Euler scheme used by the Monte Carlo engine)
- Knuth, D. E. (1969). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms.* (exact Poisson sampling used for per-step jump counts)
