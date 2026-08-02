# options-pricing-engine-rs

A Rust options pricing library covering Black-Scholes-Merton, Black-76, Heston (1993), Bates (1996), and Dupire local volatility, with full analytic Greeks where closed forms exist, a Halley-iteration implied vol solver, Levenberg-Marquardt calibration, and no-arbitrage surface repair. Built for a vol surface update cycle, not a scripting exercise.

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
| Bates (1996) | Heston CF × Merton (1976) log-normal jump CF | Bump-and-reprice |
| Local Vol (Dupire 1994) | Fritsch-Butland monotone cubic spline, differentiated through the spline, not the raw grid | Numerical (local vol surface) |

All five models share the same `OptionContract`/`PricingResult` conventions where applicable, so switching models in a caller doesn't mean rewriting the call site.

## Design

**Dispatch.** No `Box<dyn Model>` anywhere in a pricing path. Every model is a free function, statically dispatched and monomorphized. Model selection happens at the call site, not through a trait object indirection that shows up in a profiler.

**Memory layout.** `LocalVolSurface` stores `local_vols` as a flat `Vec<f64>` indexed `i_strike * n_expiry + j_expiry`, not `Vec<Vec<f64>>`. One allocation, contiguous, cache-friendly for the row/column sweeps the Dupire finite differences need.

**Characteristic function stability.** The Heston and Bates CF use the Albrecher et al. (2007) formulation, which removes the branch-cut discontinuities of the original 1993 formula. `stable_cf` and the adaptive Gauss-Kronrod integrator (`gk_integrate`) live in `heston.rs` and are shared, unmodified, by `bates.rs` and `ad.rs`. The quadrature is adaptive (substitution to a finite interval, panel subdivision until the Kronrod/Gauss error estimate is below tolerance), which matters most in the wings and at short expiries, where a fixed panel under-resolves the integrand and silently produces prices that violate static arbitrage bounds.

**Local vol.** `dvar_dk`, `dvar_dt`, and `d2var_dk2` differentiate the Fritsch-Butland monotone cubic spline (Fritsch & Butland, 1984), not the raw IV grid. Central differences on raw quotes amplify quote noise into negative local variances; sampling the spline at a symmetric offset around each node means the F-B overshoot limiter is actually load-bearing in the derivative, not just present for interpolation queries between nodes. Boundary nodes use a one-sided offset within the grid range, a symmetric step at the edge would sample outside `[strikes[0], strikes[n-1]]` and hit the spline's flat clamp, which silently halves the estimated slope.

**Surface repair.** `check_and_repair_surface` is multi-pass: it loops fixing calendar-spread and butterfly violations until the surface is clean or a pass cap is hit. Fixing one violation can create another next to it (bumping an IV to kill a calendar violation can turn a previously-fine butterfly into a violation), so a single pass is not sufficient on a surface with more than one problem.

**Automatic differentiation.** `heston_greeks_ad` propagates dual numbers (`Dual<f64>`, `Complex<Dual>`) through `stable_cf` and integrates the derivative alongside the value via the Leibniz rule, giving exact vega/vanna without a finite-difference bump size to tune. It does fewer top-level integrations than bump-and-reprice (10 vs 28, one pair of adaptive integrals per active parameter instead of two per bumped price call), but this does not currently translate into a wall-clock win, see [Performance](#performance).

**Calibration.** `calibrate_heston` runs Levenberg-Marquardt in implied-vol space, not price space. Fitting IV directly weights a 10-delta wing the same as an ATM quote; fitting price would overweight ITM options by roughly an order of magnitude relative to their information content about the smile.

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

// Heston calibration to a market IV surface
let quotes: Vec<CalibInput> = /* (contract, iv_market, weight) triples */;
let p0  = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.4, rho: -0.5 };
let res = calibrate_heston(&quotes, p0);
println!("rmse={:.4} converged={}", res.rmse, res.converged);

// Local vol: no-arbitrage repair, then Dupire
let mut surf = LocalVolSurface::new(strikes, expiries, ivs);
let audit = check_and_repair_surface(&mut surf);
println!("{} violations found, {} repaired", audit.violations.len(), audit.repaired);
let lv = dupire_local_vol(&surf, 100.0, 0.03, 0.0, 2, 1);

// Batch pricing, parallel via rayon
let chain: Vec<OptionContract> = /* ... */;
let prices = batch_bsm_price(&chain);
let ivs    = batch_implied_vol(&chain, &market_prices);
```

## Testing

47 tests, `cargo test --release`, all synchronous and deterministic (no timing-dependent assertions):

| Module | Tests | Covers |
|---|---:|---|
| `heston` | 8 | Put-call parity, sign checks, Feller condition, BSM limit (σ→0), no-static-arbitrage across a strike/expiry grid |
| `local_vol` | 13 | Flat-surface recovery, spline no-overshoot, calendar/butterfly detection and repair, multi-pass cascading repair, non-uniform grid curvature, spline-vs-raw-FD divergence on a kinked surface |
| `bates` | 6 | Recovers Heston when jump intensity is zero, put-call parity, sign checks, no-static-arbitrage |
| `ad` | 5 | Price matches the standard pricer to 1e-6 across strikes and expiries including the short-dated wings that broke the old fixed-panel quadrature, vega within 1% of bump-and-reprice, sign checks |
| `bsm` | 4 | Put-call parity, sign checks, Black-76 sanity vs BSM, Black-76 rho vs finite difference |
| `iv` | 4 | Round-trip recovery at ATM, OTM, and low vol, rejects a price outside no-arbitrage bounds |
| `calibration` | 2 | Recovers known Heston params from a synthetic surface, Feller condition always holds post-calibration |
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

- No Monte Carlo pricer. Add `rayon`-parallel MC paths if you need exotics this can't price analytically or semi-analytically.
- Bates has no AD Greeks path, `ad.rs` covers Heston only. Extending it through the Merton jump CF term is unported work, not a design limitation.
- `calibrate_heston` is a local optimizer (Levenberg-Marquardt), no global search. A bad initial guess converges to a bad local minimum, not an error, perturb and retry if `res.converged` looks suspicious. Bates isn't calibrated at all yet, only Heston.
- `batch_heston`/`batch_bates` return price only, no `PricingResult`. `batch_bsm` returns full Greeks. Inconsistent API surface, loop it yourself with `rayon` if you need batch Heston/Bates Greeks today.
- `heston_greeks_ad` is exact but measurably slower than bump-and-reprice in this repo's own benchmark (see `main.rs::heston_ad_demo`), not a theoretical concern, a measured one. Worth profiling `Complex<Dual>` before recommending the AD path for anything latency-sensitive.

## Dependencies

```
num-complex   complex arithmetic for characteristic function inversion
num-traits    trait bounds for Complex<Dual> in the AD path
rayon         parallel batch pricing
libm          erfc for full-precision ncdf
```

No `ndarray`, no `nalgebra`, no linear algebra crate, the calibration Jacobian is a 5x5 system solved by hand-rolled Gaussian elimination, not worth pulling in a dependency for.

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
