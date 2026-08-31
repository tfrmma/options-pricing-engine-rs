# options-pricing-engine-rs

[![CI](https://github.com/tfrmma/options-pricing-engine-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tfrmma/options-pricing-engine-rs/actions/workflows/ci.yml)

A Rust options pricing library covering Black-Scholes-Merton, Black-76, Heston (1993), Bates (1996), and Dupire local volatility, with full analytic Greeks where closed forms exist, a Halley-iteration implied vol solver, Levenberg-Marquardt calibration (single-start, multistart, and differential-evolution global search) for both Heston and Bates, no-arbitrage surface repair, and a Monte Carlo engine (full truncation Euler or Andersen QE) for path-dependent payoffs. Built for a vol surface update cycle, not a scripting exercise.

License: MIT. See [LICENSE](LICENSE).

## Contents

- [Models](#models)
- [Rough Bergomi (work in progress)](#rough-bergomi-work-in-progress)
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
| Monte Carlo (Heston/Bates) | Full truncation Euler (default) or Andersen (2008) QE, exact per-step Poisson jump counts, antithetic variates | N/A, path-dependent payoffs only (European, Asian, up-and-out barrier) |
| Rough Bergomi (Bayer, Friz, Gatheral 2016) | Monte Carlo via the hybrid scheme (Bennedsen, Lunde, Pakkanen 2017), κ=2, FFT tail convolution | N/A. Only `Payoff::European` is validated so far, see [Rough Bergomi](#rough-bergomi-work-in-progress) |

All five analytic models share the same `OptionContract`/`PricingResult` conventions where applicable, so switching models in a caller doesn't mean rewriting the call site.

## Rough Bergomi (work in progress)

`mc_rough_bergomi` in `mc.rs` is a real, usable pricer now: hybrid scheme (κ=2) for the near-kernel Wiener integrals via the Σ/Cholesky from commit 1, FFT tail convolution (`realfft`) for the far Riemann sum, mapped through the rBergomi variance formula (Bayer, Friz, Gatheral 2016, section 3.3) against a `ForwardVarianceCurve`, Euler-integrated log-price sharing the same `McConfig`/`Payoff`/`McResult` types as the Heston/Bates Monte Carlo pricer. Only `Payoff::European` has been validated end to end so far, `AsianArithmetic`/`UpAndOut` go through the same code path (same `payoff.eval` call as `run_path`) but haven't been specifically tested against a rough-vol path, don't assume they're correct just because the enum accepts them.

`ForwardVarianceCurve` and `RoughBergomiParams` (`types.rs`) are the two public types. The forward variance curve is piecewise-constant between listed expiries, deliberately not flat: it's arb-free by construction, not by a repair pass, strictly positive variance times strictly ascending expiries makes cumulative total variance strictly increasing, no calendar-arb check needed the way `local_vol.rs` needs one for the full (K,T) surface, this is 1D and ATM-only, there's no butterfly dimension to fight. Bootstrapping it from market ATM quotes isn't implemented yet, callers construct it directly.

The FFT plan and the Γ tail weights (`b*_k`-derived) are built once per `mc_rough_bergomi` call, outside the `rayon` path loop, and shared read-only across every path, not rebuilt per path, that was the whole point of using an FFT here instead of the O(N²) direct sum. RNG generation stays in `mc.rs` (same `box_muller`/`splitmix64` as Heston/Bates), everything downstream in `rbergomi.rs` is a pure function of the raw normals, which makes `simulate_variance_and_dz` directly unit-testable without an RNG in the loop, see the exact-variance test below.

Still missing: the deep-calibration surrogate (offline-trained tiny MLP, hand-rolled Rust inference, no ONNX runtime for a ~5.6k-parameter network), bootstrapping `ForwardVarianceCurve` from market quotes, the skew power-law and Cholesky-benchmark-smile validation tests from the paper's own section 3.3 (what exists now checks the discretization scheme's own internal consistency, not yet how well it reproduces the specific smile in Table 1), and `AsianArithmetic`/`UpAndOut` validation against a rough-vol path specifically.

All formulas checked against arXiv:1507.03004v4 directly, not from memory and not from a secondary writeup, see [References](#references). Two things worth calling out. First: `hyp2f1_matches_numeric_quadrature` initially passed with a plain equally-spaced Simpson quadrature that silently carried ~0.7% error from the integrable singularity at the interval endpoint, caught by cross-checking against an independent arbitrary-precision implementation (mpmath), not by inspection, fixed with a substitution, now agrees to 3e-7. Second: `variance_of_simulated_y_matches_exact_scheme_variance` doesn't test against the paper's asymptotic MSE result (Theorem 2.5/Corollary 2.7), that convergence is genuinely slow for α near -0.5 (rate ~n^{-(α+1/2)+ε}, which at α=-0.43 is close to n^{-0.07}) and would need an impractically large `n_steps` to pin down tightly in a fast test. Instead it checks Var(Yn(t)) against a closed form derived from the discretization scheme's *own* definition (near and far terms touch disjoint step indices with nonzero weight, so they're independent, no cross-covariance term needed), which is exact at any `n_steps` and isolates implementation bugs from the scheme's own, already-proven-in-the-paper approximation error.

## Design

**Dispatch.** No `Box<dyn Model>` anywhere in a pricing path. Every model is a free function, statically dispatched and monomorphized. Model selection happens at the call site, not through a trait object indirection that shows up in a profiler.

**Memory layout.** `LocalVolSurface` stores `local_vols` as a flat `Vec<f64>` indexed `i_strike * n_expiry + j_expiry`, not `Vec<Vec<f64>>`. One allocation, contiguous, cache-friendly for the row/column sweeps the Dupire finite differences need.

**Characteristic function stability.** The Heston and Bates CF use the Albrecher et al. (2007) formulation, which removes the branch-cut discontinuities of the original 1993 formula. `stable_cf` and the adaptive Gauss-Kronrod integrator (`gk_integrate`) live in `heston.rs` and are shared, unmodified, by `bates.rs` and `ad.rs`. The quadrature is adaptive (substitution to a finite interval, panel subdivision until the Kronrod/Gauss error estimate is below tolerance), which matters most in the wings and at short expiries, where a fixed panel under-resolves the integrand and silently produces prices that violate static arbitrage bounds. `stable_cf`'s one `sqrt()` call per evaluation uses `fast_csqrt`, a closed-form algebraic complex square root, instead of `num_complex::Complex64::sqrt()`'s general branch, which goes through `to_polar()`/`from_polar()` (hypot + atan2 + sqrt + cos + sin, verified by reading the crate source). `fast_csqrt` needs hypot + 2 sqrt + a sign, no trig. In isolation this is ~5x faster (measured), but `stable_cf` also calls `exp()` and `ln()` and does several complex multiply/divides, so the end-to-end win on a full `stable_cf` call is a real but modest ~4%, not 5x, isolated micro-benchmarks of one operation don't linearly predict aggregate impact when a CPU can overlap independent work. Kept anyway: verified correct (680+ point sweep against the builtin, concentrated near the axes where a naive version of this formula catastrophically cancels, see [Testing](#testing)), strictly no downside, and it's the same fix `ad.rs`'s dual `csqrt` needed for the same reason.

**Local vol.** `dvar_dk`, `dvar_dt`, and `d2var_dk2` differentiate the Fritsch-Butland monotone cubic spline (Fritsch & Butland, 1984), not the raw IV grid. Central differences on raw quotes amplify quote noise into negative local variances; sampling the spline at a symmetric offset around each node means the F-B overshoot limiter is actually load-bearing in the derivative, not just present for interpolation queries between nodes. Boundary nodes use a one-sided offset within the grid range, a symmetric step at the edge would sample outside `[strikes[0], strikes[n-1]]` and hit the spline's flat clamp, which silently halves the estimated slope.

**Surface repair.** `check_and_repair_surface` is multi-pass: it loops fixing calendar-spread and butterfly violations until the surface is clean or a pass cap is hit. Fixing one violation can create another next to it (bumping an IV to kill a calendar violation can turn a previously-fine butterfly into a violation), so a single pass is not sufficient on a surface with more than one problem.

**Monte Carlo.** `mc_heston`/`mc_bates` default to full truncation Euler (Lord, Koekkoek, van Dijk 2010) for the variance process, correlated via the standard two-normal construction. `McConfig::scheme = VarianceScheme::QuadraticExponential` switches to Andersen's (2008) QE scheme: samples v(t+Δ) from a moment-matched distribution (quadratic-in-normal when ψ≤1.5, an exponential/point-mass mixture above) instead of discretizing and truncating the CIR SDE, and prices using his equation (33) for the log-price update (K0-K4 coefficients, central discretization γ1=γ2=0.5), verified against the primary source PDF, not a secondary writeup. The price innovation in (33) uses an independent normal, correlation with V is already analytic in K1/K2, reusing Euler's rho-correlated shock there was the bug in the first version of this, it gave a *worse* price than plain Euler (off by 6.2 vs analytic, Euler was off by 1.8) until fixed against the actual paper. Measured, not assumed: in a badly Feller-violating case (2κθ=0.4 ≪ σ²=1.44) with a coarse 8-step/year grid, QE cuts the bias against the analytic price roughly 20x versus Euler (`qe_reduces_bias_in_feller_violating_regime`). Bates jumps use an exact per-step Poisson draw (Knuth's algorithm), not the "coin flip with probability λdt" shortcut that silently drops the probability of two or more jumps landing in the same step. Parallelized over path chunks via rayon, with a `splitmix64`-hashed seed per chunk so parallel runs are reproducible and statistically independent. Use this for path-dependent payoffs the CF-inversion pricers can't touch (Asian averages, barriers), not for vanillas, `heston_price`/`bates_price` are exact and far cheaper for those.

**Automatic differentiation.** `heston_greeks_ad` and `bates_greeks_ad` propagate dual numbers (`Dual<f64>`, `Complex<Dual>`) through the characteristic function and integrate the derivative alongside the value via the Leibniz rule, giving exact vega/vanna without a finite-difference bump size to tune. The Bates version composes the Heston CF and the Merton jump CF *before* differentiating (same order `bates_call` uses), so price and the five Heston-driven Greeks are exact through the jump-adjusted CF. Jump-parameter sensitivities (d/dλ, d/dμⱼ, d/dσⱼ) are a separate function, `ad::bates_jump_sensitivities_ad`, same `forward_pass` machinery with the Heston side pinned constant and a jump parameter carrying the active derivative instead, it's what `calibrate_bates`'s Jacobian uses for its jump columns now instead of FD.

`heston_greeks_ad5` is a second, experimental AD path: `Dual5` carries all 5 Heston-parameter tangent directions at once (`dot: [f64; 5]` instead of `f64`) instead of running 5 separate scalar-`Dual` passes, so the CF's *value* gets computed once instead of five times redundantly. Measured (`profile_dual5_vs_five_scalar_passes`, `#[ignore]`d, same methodology as the other profile tests): the joint pass is a real **~3x faster** than 5 scalar passes at the integration level (211µs vs 623µs on this box). But `heston_greeks_ad`/`ad5` spend most of their wall-clock time in the FD-bumped delta/gamma/theta/rho/vanna/volga, identical between both versions, so the win at the full-function level is a much more modest ~6%, and `heston_greeks_ad5` is still slower than bump-and-reprice overall. The 3x is real and would matter if delta/gamma/rho/vanna/volga also moved onto AD (a further, larger `Dual5`-style tangent space covering spot and rate too), that's the natural next step this result points to, not implemented here.

Profiled properly (see `ad::tests::profile_*`, `#[ignore]`d, run with `cargo test --release -- --ignored --nocapture --test-threads=1 ad::tests::profile`), not guessed at: `Complex<Dual>` multiply is ~5.5x a plain `Complex64` multiply, divide is ~11x, both measured with varying inputs cycled through the benchmark so LLVM can't hoist a closure-captured constant out of the loop and report a fake sub-nanosecond number (the first version of this benchmark did exactly that for `sqrt`/`ln`, caught by the results being physically impossible, not by inspection). But `exp` is only ~1.3x and `ln` ~1.1x, dual bookkeeping for those reuses the already-computed value via the standard AD reuse trick instead of redoing the whole computation twice. A full CF evaluation is dominated by the cheap transcendentals, not by raw multiply/divide count, so the aggregate overhead lands around 1.4-1.5x per GK panel and per full pricing pass, nowhere near the 11x the division number alone would suggest. That 1.4-1.5x is why `heston_greeks_ad`/`bates_greeks_ad` are exact but not currently faster wall-clock than bump-and-reprice despite doing fewer integrations (10 vs 28), see [Performance](#performance).

**Calibration.** `calibrate_heston`/`calibrate_bates` run Levenberg-Marquardt in implied-vol space, not price space. Fitting IV directly weights a 10-delta wing the same as an ATM quote; fitting price would overweight ITM options by roughly an order of magnitude relative to their information content about the smile. Both share one generic LM engine (`CalibModel` trait, Heston is a 5-parameter instance, Bates is 8) instead of two near-identical copies of the damping/Jacobian/Gauss-elimination machinery. Bates' 3 jump columns (λ, μⱼ, σⱼ) use exact forward-mode AD (`ad::bates_jump_sensitivities_ad`) converted from price-space to vol-space via `d(iv)/d(param) = [d(price)/d(param)] / vega_BSM(iv)`, the standard implicit-function-theorem trick for turning a price derivative into a vol derivative without re-deriving the IV solver. The 5 Heston-inherited columns still use FD, extending those to AD too is a separate change, `CalibModel::ad_price_derivative` defaults to `None` (FD) and only `BatesParams` overrides it, for columns 5..=7.

`calibrate_heston_multistart`/`calibrate_bates_multistart` run several LM fits in parallel from randomized starting points (rayon, one thread per restart) and keep the best, a bad single initial guess converging to a bad local minimum no longer means a silently bad fit. Restarts aren't fully isolated: they share one best-SSE-seen-so-far behind a mutex, checked every 10 iterations, and a restart running more than 8x worse than the best any other restart has found gets aborted instead of burning its full iteration budget. This can only kill a restart that's already losing, never changes which restart wins.

`calibrate_heston_global`/`calibrate_bates_global` are an actual population-based global search (differential evolution, DE/rand/1/bin) instead of repeated local restarts, doesn't need an initial guess at all. Infeasible individuals (mostly Feller violations) get an infinite fitness rather than a repair step, DE's own selection pressure steers the population away from them. DE finds the right basin but doesn't polish well (no gradient), so the winner gets one LM run to finish. Real caveat found while testing this on Bates: DE can converge to an *excellent* fit (rmse ~0) with parameters wildly different from whatever generated the data, that's not a bug, it's Bates' well-known identifiability problem, several very different (v0, kappa, theta, sigma, rho, λ, μⱼ, σⱼ) combinations can price the same finite set of vanilla quotes essentially identically.

Two things exist specifically to deal with that instead of just documenting it as a footnote. First, every `CalibResult` carries a quantitative identifiability diagnostic: `condition_number` and `weakest_direction`, the ratio and eigenvector of J'J's largest and smallest eigenvalues at the converged params (`J'J` was already computed for the LM step, this is a Jacobi eigenvalue decomposition on a matrix that already exists, no extra pricing calls). Measured, not asserted: a well-identified Heston fit lands around condition_number ~3e6, Bates from a sensible p0 lands around ~3.7e7, the pathological unconstrained-DE result comes back literally infinite (a Jacobian column with ~zero curvature). The diagnostic is always computed from the data-only Jacobian, even when regularization (below) is active, adding a prior artificially shrinks the condition number by construction and reporting that would hide the exact thing this exists to catch. Second, `calibrate_heston_regularized`/`calibrate_bates_regularized` (and the `_global_regularized` DE variants) add a Tikhonov pull toward a prior parameter set, implemented as extra pseudo-residuals appended to the LM problem (the standard way to fold ridge regularization into Gauss-Newton without a separate hand-derived penalty gradient), so it reuses the exact same LM machinery. Measured on the same pathological case: even a light `reg_weight` (1e-3) moves the DE+polish result from wild (v0=1.64 vs true 0.04) to close (v0=0.041 vs true 0.04) for a rmse cost of about 0.0008 vol points, and the condition number drops from infinite to ~3e7, the same order as a normally-identified fit. Regularizing a case the data already pins down well costs almost nothing (`regularization_does_not_hurt_a_well_identified_case`).

`calibrate_bates` from a sensible p0 stays near the intended basin because LM only takes local steps; DE has no such bias and no reason to prefer "the generating params" over any other point on the same fitness plateau. Still a local method under the hood once LM polishes, more restarts/generations help but don't guarantee the global optimum.

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
println!("best rmse={:.4}, {}/{} converged, {} pruned early",
    multi.best.rmse, multi.n_converged, multi.n_restarts, multi.n_pruned);

// real global search, no p0 needed at all
let global = calibrate_heston_global(&quotes, 40, 60, 777);
println!("DE+polish rmse={:.4}", global.best.rmse);

// identifiability diagnostic: every CalibResult carries this, check it
// before trusting individual parameter values, not just the rmse
println!("condition number={:.2e}", global.best.condition_number);
if global.best.condition_number > 1e9 {
    println!("poorly identified, weakest direction: {:?}", global.best.weakest_direction);
}

// Tikhonov regularization toward a prior, for exactly that situation
let prior = BatesParams { heston: p0, lambda: 0.4, mu_j: -0.08, sigma_j: 0.12 };
let bp0   = BatesParams { heston: p0, lambda: 0.5, mu_j: -0.1,  sigma_j: 0.15 };
let reg   = calibrate_bates_global_regularized(&quotes, 40, 60, 777, &prior, 1e-3);
println!("regularized rmse={:.4} cond={:.2e}", reg.best.rmse, reg.best.condition_number);

// Bates calibration, same engine, 8 params instead of 5, jump columns via AD
let bres = calibrate_bates(&quotes, bp0);

// Jump-parameter sensitivities directly, if you need them outside a Jacobian
let sens = bates_jump_sensitivities_ad(100.0, 100.0, 1.0, 0.05, 0.0, &params, 0.5, -0.10, 0.15, OptionType::Call);
println!("d(price)/d(lambda)={:.4} d(price)/d(mu_j)={:.4}", sens.d_lambda, sens.d_mu_j);

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

// QE scheme instead of the default full truncation Euler, worth it when
// the bias in a Feller-violating / short-dated regime actually matters
let mut cfg_qe = McConfig::default();
cfg_qe.scheme = VarianceScheme::QuadraticExponential;
let qe_price = mc_heston(100.0, 1.0, 0.05, 0.0, &params,
    Payoff::European { strike: 100.0, opt_type: OptionType::Call }, &cfg_qe);

let barrier = mc_bates(100.0, 1.0, 0.05, 0.0, &params, 0.5, -0.10, 0.15,
    Payoff::UpAndOut { strike: 100.0, barrier: 130.0, rebate: 0.0, opt_type: OptionType::Call }, &cfg);

// Batch pricing, parallel via rayon
let chain: Vec<OptionContract> = /* ... */;
let prices = batch_bsm_price(&chain);
let ivs    = batch_implied_vol(&chain, &market_prices);
let hgreeks = batch_heston_greeks(&chain, &params);       // full PricingResult per option
let bgreeks = batch_bates_greeks(&chain, &bp0);
```

## Testing

93 tests, `cargo test --release`, all synchronous and deterministic (no timing-dependent assertions, the Monte Carlo tests use a fixed seed and check convergence against the analytic price within a multiple of the MC's own reported standard error, not a fixed tolerance). A further 6 profiling benchmarks are marked `#[ignore]` since they measure timing, not correctness, run them with `cargo test --release -- --ignored --nocapture --test-threads=1 ad::tests::profile`.

| Module | Tests | Covers |
|---|---:|---|
| `calibration` | 16 | Recovers known Heston and Bates params from a synthetic surface, Feller condition always holds post-calibration, multistart never loses to single-start and reliably escapes a deliberately bad initial guess, the early-stop prune mechanism kills a hopeless restart but leaves a competitive one alone, Bates' AD Jacobian columns match an independently-computed pure-FD reference, DE+polish recovers Heston params with no initial guess at all, DE+polish on Bates finds an excellent fit without necessarily finding the generating params (identifiability), the Jacobi eigenvalue decomposition reconstructs known matrices exactly (closed-form 2x2, self-consistency at 5x5 and 8x8, finds a planted near-zero eigenvalue), the condition-number diagnostic separates a well-identified fit from the known-degenerate Bates case (finite ~3e6-3.7e7 vs literally infinite), Tikhonov regularization recovers plausible params in that same degenerate case and costs almost nothing on an already-well-identified one |
| `local_vol` | 13 | Flat-surface recovery, spline no-overshoot, calendar/butterfly detection and repair, multi-pass cascading repair, non-uniform grid curvature, spline-vs-raw-FD divergence on a kinked surface |
| `ad` | 15 | Heston and Bates price match their analytic pricers to 1e-6 across strikes and expiries including the short-dated wings that broke the old fixed-panel quadrature, vega within 1% of bump-and-reprice for both, Bates AD collapses to Heston AD when jumps are off, dual `csqrt` matches the builtin value and a finite-difference derivative across a sweep concentrated near the branch cut, jump-parameter sensitivities match FD on the analytic Bates pricer, `Dual5` (the multi-directional experiment) matches the scalar `Dual` path on every Greek across the same wings grid, sign checks |
| `heston` | 9 | Put-call parity, sign checks, Feller condition, BSM limit (σ→0), no-static-arbitrage across a strike/expiry grid, `fast_csqrt` matches `Complex64::sqrt()` across 680+ swept points including near-axis angles down to 1e-12 radians |
| `rbergomi` | 9 | Hybrid scheme covariance Σ matches the Itô isometry at Σ₁,₁, symmetric, Cholesky reconstructs it exactly, stays positive definite down to α=-0.49 (H≈0.01, closer to the crypto short-dated regime than the paper's own H=0.07 test case), scales with n at the rate the paper predicts, optimal evaluation points b*_k land inside their cell, the 2F1 series matches an independent singularity-regularized quadrature, the FFT tail convolution matches an independent direct O(N²) sum, Var(Yn(t)) matches a closed form derived from the scheme's own definition (see [Rough Bergomi](#rough-bergomi-work-in-progress)) |
| `mc` | 9 | European MC matches analytic Heston/Bates within a z-score bound on the MC's own standard error (both schemes), Asian call cheaper than European (real inequality), up-and-out cheaper than vanilla with zero rebate (real inequality), QE cuts bias ~20x vs Euler in a Feller-violating coarse-step regime, Poisson sampler mean check, rBergomi forward is a martingale (strike=0 call = E[disc·S_T] = S_0 exactly, by construction) within a z-score bound, rBergomi converges to Black-Scholes as η→0 |
| `bates` | 6 | Recovers Heston when jump intensity is zero, put-call parity, sign checks, no-static-arbitrage |
| `batch` | 5 | Batch price matches scalar calls, batch IV round-trips, batch Heston/Bates Greeks match scalar `PricingResult` field-by-field, batch price-only output agrees with batch Greeks output |
| `bsm` | 4 | Put-call parity, sign checks, Black-76 sanity vs BSM, Black-76 rho vs finite difference |
| `iv` | 4 | Round-trip recovery at ATM, OTM, and low vol, rejects a price outside no-arbitrage bounds |
| `math` | 3 | `ncdf` sanity and tail precision, `ncdf_inv` round-trip |

Two of these are worth calling out specifically: `fast_csqrt_matches_builtin` exists because the first version of `fast_csqrt` passed a 37-angle, evenly-spread correctness sweep and then broke `zero_vol_of_vol_matches_bsm` in the full suite, a coarse angular sweep doesn't sample close enough to the axes to catch catastrophic cancellation that only bites within a fraction of a degree of them. Both `fast_csqrt` and the dual `csqrt` in `ad.rs` had this bug, independently, from the same textbook formula. Neither test is decorative.

The arbitrage tests aren't incidental: `no_static_arbitrage` in `heston.rs` and the AD wings test exist specifically because a non-adaptive quadrature passed every other test in this suite while quietly producing arbitrage-violating prices at short expiries. A regression here is a real bug, not a tolerance nitpick.

## Performance

Throughput on batch pricing scales with core count through `rayon` (`RAYON_NUM_THREADS`, or default to all cores), so a single fixed "ops/ms" number is a property of whatever machine ran it, not of the code. Measure it on your own hardware:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

`main.rs::batch_bench` times a 500-option BSM and Heston chain and prints real numbers for whatever box it runs on. `main.rs::heston_ad_demo` does the same for bump-and-reprice vs AD Greeks on a single option, averaged over 2,000 reps. Treat both as a local baseline, not a spec.

Qualitatively: BSM is closed-form and embarrassingly parallel, Heston and Bates cost an adaptive double integral per price (more at short expiries and in the wings, where more panels are needed to hit tolerance), and `heston_greeks_ad` currently runs slower wall-clock than bump-and-reprice despite doing fewer integrations, `Complex<Dual>` arithmetic costs more per quadrature node than plain `Complex64`, see [Known limitations](#known-limitations-and-roadmap).

## Known limitations and roadmap

- Bates' identifiability problem now has a diagnostic (`condition_number`/`weakest_direction` on every `CalibResult`) and a mitigation (`calibrate_*_regularized`), but neither makes the underlying issue go away. The diagnostic tells you a fit is untrustworthy, it doesn't tell you the *right* answer. Regularization pulls toward a prior you supply, if that prior is wrong the regularized result is just confidently wrong in a different direction, garbage in, garbage out still applies. `reg_weight` has no universally correct value, and there's no automatic way to pick one, that's still on the caller.
- QE (`VarianceScheme::QuadraticExponential`) implements Andersen's base scheme (his eq 33), not the martingale-corrected QE-M variant. Andersen's own paper treats QE (not QE-M) as the practical default, so this isn't a shortcut, but QE-M exists as a further refinement nobody's ported.
- `heston_greeks_ad5` proves multi-directional dual arithmetic is a real ~3x win at the integration level (measured, `profile_dual5_vs_five_scalar_passes`), but doesn't flip the headline number: `heston_greeks_ad`/`ad5`/`bates_greeks_ad` are still slower than bump-and-reprice overall (~1.3-1.5x) because delta/gamma/theta/rho/vanna/volga are still FD-bumped regardless of which path computes vega. Extending the `Dual5`-style joint pass to cover spot and rate too (not just the 5 Heston CF params) is the next step this result points to, not done here.
- Rough Bergomi (`rbergomi.rs`, `mc_rough_bergomi` in `mc.rs`) prices vanilla European options, no calibration surrogate yet and `AsianArithmetic`/`UpAndOut` are untested against it, see [Rough Bergomi (work in progress)](#rough-bergomi-work-in-progress) for exactly what exists and what's still missing.
- CI (`.github/workflows/ci.yml`) pins the toolchain to 1.75.0, the exact version everything here was verified clean against (`cargo build --release --all-targets`, full test suite, `cargo clippy --release --all-targets -- -D warnings`). Bumping it is fine, but re-run clippy locally against the new toolchain first, new Rust releases add new clippy lints and "stable" drifting out from under you is exactly how a previously-green CI starts failing on code nobody touched.

## Dependencies

```
num-complex   complex arithmetic for characteristic function inversion
num-traits    trait bounds for Complex<Dual> in the AD path
rayon         parallel batch pricing, Monte Carlo paths, and multistart calibration
rand          RNG for Monte Carlo paths, DE population init, and multistart restarts (SmallRng, seeded per unit of parallel work)
libm          erfc for full-precision ncdf
realfft       real-to-complex FFT for the rough Bergomi hybrid scheme's tail convolution (wraps rustfft, pulled in transitively)
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
- Storn, R., Price, K. (1997). *Differential Evolution, A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.* (DE/rand/1/bin scheme used by `calibrate_heston_global`/`calibrate_bates_global`)
- Jacobi, C. G. J. (1846). *Über ein leichtes Verfahren, die in der Theorie der Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen.* Classic eigenvalue algorithm for symmetric matrices, used for the calibration identifiability diagnostic (`condition_number`/`weakest_direction`). See also Golub, G. H., Van Loan, C. F. *Matrix Computations* for the modern presentation this implementation follows.
- Tikhonov, A. N. (1963). *Solution of Incorrectly Formulated Problems and the Regularization Method.* Regularization toward a prior, used by `calibrate_heston_regularized`/`calibrate_bates_regularized`.
- Lord, R., Koekkoek, R., van Dijk, D. (2010). *A Comparison of Biased Simulation Schemes for Stochastic Volatility Models.* (full truncation Euler scheme used by the Monte Carlo engine)
- Andersen, L. (2008). *Efficient Simulation of the Heston Stochastic Volatility Model.* (QE scheme, `VarianceScheme::QuadraticExponential`)
- Knuth, D. E. (1969). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms.* (exact Poisson sampling used for per-step jump counts)
- Bennedsen, M., Lunde, A., Pakkanen, M. S. (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics 21(4). (the hybrid scheme kernel implemented in `rbergomi.rs`)
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under rough volatility.* Quantitative Finance 16(6). (the rough Bergomi model itself, target of the hybrid scheme's option-pricing experiment)
