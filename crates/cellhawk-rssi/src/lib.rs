//! # cellhawk-rssi
//!
//! RF signal processing layer for CellHawk, implementing:
//!
//! - [`ldpl`]            — Log-Distance Path Loss model with adaptive exponent (§3.1)
//! - [`multilateration`] — Weighted Least Squares + TRF solver + RANSAC (§3.2)
//! - [`fading`]          — Rician fading channel model (§3.3)

pub mod fading;
pub mod ldpl;
pub mod multilateration;

pub use fading::RicianFadingModel;
pub use ldpl::LdplModel;
pub use multilateration::MultilaterationSolver;
