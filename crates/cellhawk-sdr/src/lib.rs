//! # cellhawk-sdr
//!
//! Software-Defined Radio front-end for CellHawk.
//!
//! Bridges raw I/Q samples from hardware (RTL-SDR, HackRF, USRP) to the
//! navigation-layer types consumed by the EKF and tier arbiter.
//!
//! ## Pipeline
//!
//! ```text
//! Hardware I/Q → PowerEstimator → JnrEstimator → jnr_db  → EKF update_jnr
//!                               → RssiExtractor → rssi_dbm → MultilaterationSolver
//! ```
//!
//! ## Design
//!
//! All processing is backend-agnostic: the [`SdrBackend`] trait abstracts
//! hardware I/O.  [`SimulatedSdrBackend`] provides deterministic I/Q injection
//! for unit tests and HIL simulation.

pub mod jnr;
pub mod power;
pub mod rssi;

pub use jnr::{JnrEstimator, JnrEstimatorConfig};
pub use power::{IqSample, PowerEstimator, PowerEstimatorConfig};
pub use rssi::{RssiExtractor, RssiExtractorConfig};

#[allow(unused_imports)]
use cellhawk_types::CellHawkError as _;

// ─────────────────────────────────────────────────────────────────────────────
// SdrBackend trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over any SDR hardware or simulation source.
///
/// Implementors must be `Send` so the front-end can run on a dedicated thread.
pub trait SdrBackend: Send {
    /// Pull the next batch of I/Q samples.
    ///
    /// Returns `None` when the hardware buffer is empty (non-blocking).
    /// The batch size is implementation-defined; typical values are 512–4096.
    fn pull_samples(&mut self) -> Option<Vec<IqSample>>;

    /// Whether the hardware link is healthy (USB connected, gain locked, etc.).
    fn is_healthy(&self) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedSdrBackend
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic I/Q injection backend for tests and HIL simulation.
///
/// Callers push batches of I/Q samples; the backend returns them in FIFO order.
#[derive(Debug, Default, Clone)]
pub struct SimulatedSdrBackend {
    queue: std::collections::VecDeque<Vec<IqSample>>,
    healthy: bool,
}

impl SimulatedSdrBackend {
    pub fn new() -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
            healthy: true,
        }
    }

    /// Inject a batch of I/Q samples to be returned by the next `pull_samples`.
    pub fn inject(&mut self, samples: Vec<IqSample>) {
        self.queue.push_back(samples);
    }

    /// Inject a constant-power sinusoidal signal at normalised frequency `f`
    /// (cycles per sample) with amplitude `amplitude`.
    ///
    /// Useful for testing JNR estimation with a known SNR.
    pub fn inject_tone(&mut self, amplitude: f32, freq_norm: f32, n_samples: usize) {
        let samples: Vec<IqSample> = (0..n_samples)
            .map(|k| {
                let phase = 2.0 * std::f32::consts::PI * freq_norm * k as f32;
                IqSample {
                    i: amplitude * phase.cos(),
                    q: amplitude * phase.sin(),
                }
            })
            .collect();
        self.inject(samples);
    }

    /// Inject white Gaussian noise with standard deviation `sigma`.
    ///
    /// Uses a deterministic Box-Muller transform seeded from the sample index
    /// so tests are reproducible without an external RNG dependency.
    pub fn inject_noise(&mut self, sigma: f32, n_samples: usize) {
        let samples: Vec<IqSample> = (0..n_samples)
            .map(|k| {
                // Deterministic Box-Muller (no external RNG needed)
                let u1 = (k as f32 * 0.618_034 + 0.1).fract().max(1e-7);
                let u2 = (k as f32 * 0.381_966 + 0.7).fract();
                let mag = sigma * (-2.0 * u1.ln()).sqrt();
                let phi = 2.0 * std::f32::consts::PI * u2;
                IqSample {
                    i: mag * phi.cos(),
                    q: mag * phi.sin(),
                }
            })
            .collect();
        self.inject(samples);
    }

    pub fn set_healthy(&mut self, healthy: bool) {
        self.healthy = healthy;
    }
}

impl SdrBackend for SimulatedSdrBackend {
    fn pull_samples(&mut self) -> Option<Vec<IqSample>> {
        self.queue.pop_front()
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SdrFrontEnd — top-level processing pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the SDR front-end pipeline.
#[derive(Debug, Clone)]
pub struct SdrFrontEndConfig {
    pub power: PowerEstimatorConfig,
    pub jnr: JnrEstimatorConfig,
    pub rssi: RssiExtractorConfig,
}

impl SdrFrontEndConfig {
    pub fn new() -> Self {
        Self {
            power: PowerEstimatorConfig::default(),
            jnr: JnrEstimatorConfig::default(),
            rssi: RssiExtractorConfig::default(),
        }
    }
}

impl Default for SdrFrontEndConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Output of one SDR processing cycle.
#[derive(Debug, Clone)]
pub struct SdrOutput {
    /// Jammer-to-Noise Ratio estimate (dB). Drives EKF tier arbitration.
    pub jnr_db: f64,
    /// Received Signal Strength Indicator from pilot correlation (dBm).
    pub rssi_dbm: f64,
    /// Whether the SDR hardware link is healthy.
    pub hardware_healthy: bool,
}

/// Top-level SDR front-end: pulls I/Q samples and runs the full pipeline.
pub struct SdrFrontEnd<B: SdrBackend> {
    backend: B,
    power: PowerEstimator,
    jnr: JnrEstimator,
    rssi: RssiExtractor,
}

impl<B: SdrBackend> SdrFrontEnd<B> {
    pub fn new(backend: B, config: SdrFrontEndConfig) -> Self {
        Self {
            backend,
            power: PowerEstimator::new(config.power),
            jnr: JnrEstimator::new(config.jnr),
            rssi: RssiExtractor::new(config.rssi),
        }
    }

    /// Process one batch of I/Q samples and return navigation-layer outputs.
    ///
    /// Returns `None` if no samples are available this cycle (non-blocking).
    pub fn process(&mut self) -> Option<SdrOutput> {
        let samples = self.backend.pull_samples()?;
        if samples.is_empty() {
            return None;
        }

        let mean_power_dbm = self.power.update(&samples);
        let jnr_db = self.jnr.update(mean_power_dbm);
        let rssi_dbm = self.rssi.extract(&samples);

        Some(SdrOutput {
            jnr_db,
            rssi_dbm,
            hardware_healthy: self.backend.is_healthy(),
        })
    }

    /// Direct access to the backend (e.g. for injection in tests).
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Whether the hardware link is healthy.
    pub fn is_healthy(&self) -> bool {
        self.backend.is_healthy()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frontend() -> SdrFrontEnd<SimulatedSdrBackend> {
        SdrFrontEnd::new(SimulatedSdrBackend::new(), SdrFrontEndConfig::default())
    }

    #[test]
    fn process_returns_none_when_no_samples() {
        let mut fe = make_frontend();
        assert!(fe.process().is_none());
    }

    #[test]
    fn process_returns_some_after_injection() {
        let mut fe = make_frontend();
        fe.backend_mut().inject_tone(1.0, 0.1, 256);
        let out = fe.process();
        assert!(out.is_some());
    }

    #[test]
    fn hardware_healthy_reflects_backend_state() {
        let mut fe = make_frontend();
        assert!(fe.is_healthy());
        fe.backend_mut().set_healthy(false);
        assert!(!fe.is_healthy());
    }

    #[test]
    fn strong_tone_gives_positive_jnr() {
        let mut fe = make_frontend();
        // Inject noise first to establish noise floor
        fe.backend_mut().inject_noise(0.01, 512);
        fe.process();
        // Now inject a strong tone (amplitude 1.0 >> noise sigma 0.01)
        fe.backend_mut().inject_tone(1.0, 0.1, 512);
        let out = fe.process().unwrap();
        assert!(
            out.jnr_db > 0.0,
            "strong tone must give positive JNR, got {:.2}",
            out.jnr_db
        );
    }

    #[test]
    fn pure_noise_gives_near_zero_jnr() {
        let mut fe = make_frontend();
        // Inject identical noise batches — signal ≈ noise floor → JNR ≈ 0
        for _ in 0..5 {
            fe.backend_mut().inject_noise(0.1, 256);
            fe.process();
        }
        fe.backend_mut().inject_noise(0.1, 256);
        let out = fe.process().unwrap();
        // JNR should be near 0 (within ±6 dB for minimum-statistics estimator)
        assert!(
            out.jnr_db < 10.0,
            "pure noise JNR should be low, got {:.2}",
            out.jnr_db
        );
    }

    #[test]
    fn jnr_exceeds_tier1_threshold_under_strong_jamming() {
        use cellhawk_types::NavigationTier;
        let mut fe = make_frontend();
        // Establish noise floor
        fe.backend_mut().inject_noise(0.001, 512);
        fe.process();
        // Inject very strong jamming signal (amplitude 10.0 >> noise 0.001)
        fe.backend_mut().inject_tone(10.0, 0.05, 512);
        let out = fe.process().unwrap();
        assert!(
            out.jnr_db >= NavigationTier::TIER1_JNR_THRESHOLD_DB,
            "strong jamming must exceed Tier1 threshold, got {:.2} dB",
            out.jnr_db
        );
    }
}
