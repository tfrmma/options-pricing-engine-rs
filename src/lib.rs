// options-pricing-engine-rs
// build: RUSTFLAGS="-C target-cpu=native" cargo build --release

pub mod types;
pub mod math;
mod greeks;
pub mod bsm;
pub mod iv;
pub mod heston;
pub mod bates;
pub mod ad;
pub mod local_vol;
pub mod mc;
pub mod batch;
pub mod calibration;
pub mod rbergomi;

// flatten the hot path to crate root
pub use types::{OptionType, OptionContract, PricingResult, IvProblem,
                HestonParams, BatesParams, LocalVolSurface,
                RoughBergomiParams, ForwardVarianceCurve};
pub use bsm::{bsm_price, bsm_price_and_greeks, black76_price_and_greeks};
pub use iv::implied_vol;
pub use heston::{heston_price, heston_price_and_greeks};
pub use ad::{heston_greeks_ad, heston_greeks_ad5, bates_greeks_ad, bates_jump_sensitivities_ad, BatesJumpSensitivities};
pub use bates::{bates_price, bates_price_and_greeks};
pub use local_vol::{dupire_local_vol, monotone_cubic_interp, check_and_repair_surface,
                     repair_surface_to_clean, SurfaceAudit, SurfaceViolation, ViolationKind};
pub use mc::{mc_heston, mc_bates, McConfig, McResult, Payoff, VarianceScheme};
pub use calibration::{calibrate_heston, calibrate_bates,
                       calibrate_heston_multistart, calibrate_bates_multistart,
                       calibrate_heston_global, calibrate_bates_global,
                       calibrate_heston_regularized, calibrate_bates_regularized,
                       calibrate_heston_global_regularized, calibrate_bates_global_regularized,
                       CalibInput, CalibResult, MultistartResult, GlobalCalibResult};
pub use batch::{batch_bsm, batch_bsm_price, batch_heston, batch_heston_greeks,
                batch_bates, batch_bates_greeks, batch_implied_vol};
