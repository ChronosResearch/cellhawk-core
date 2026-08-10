//! # cellhawk-ekf
//!
//! Extended Kalman Filter with JNR-based dynamic covariance scaling and
//! three-tier navigation arbitration, as specified in §3.4 of the paper.
//!
//! ## Modules
//! - [`filter`]     — 9-state EKF implementation
//! - [`covariance`] — JNR-driven dynamic covariance scaling (sigmoid ramp)
//! - [`tier`]       — Tier arbitration and smooth handover logic

pub mod covariance;
pub mod filter;
pub mod tier;

pub use filter::CellHawkEkf;
pub use tier::TierArbiter;
