// options-pricing-engine-rs
// build: RUSTFLAGS="-C target-cpu=native" cargo build --release

pub mod types;
pub mod math;
pub mod bsm;
pub mod iv;
pub mod heston;
pub mod bates;
pub mod local_vol;
pub mod batch;

// flatten the hot path to crate root
pub use types::{OptionType, OptionContract, PricingResult, IvProblem,
                HestonParams, BatesParams, LocalVolSurface};
pub use bsm::{bsm_price, bsm_price_and_greeks, black76_price_and_greeks};
pub use iv::implied_vol;
pub use heston::{heston_price, heston_price_and_greeks};
pub use bates::{bates_price, bates_price_and_greeks};
pub use local_vol::{dupire_local_vol, monotone_cubic_interp};
pub use batch::{batch_bsm, batch_bsm_price, batch_heston, batch_bates, batch_implied_vol};
