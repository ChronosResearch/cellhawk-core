//! Sliding-window I/Q power estimator.
//!
//! Computes the mean received power over a configurable window of I/Q samples.
//! Power is returned in dBm using the convention:
//!
//! ```text
//! P_linear = (1/N) · Σ (I_k² + Q_k²)
//! P_dBm    = 10 · log₁₀(P_linear)   [relative to unit impedance]
//! ```
//!
//! The linear scale is normalised so that a full-scale sinusoid (amplitude 1.0)
//! gives P_linear = 0.5, i.e. P_dBm ≈ −3 dBm.  This matches the convention
//! used by RTL-SDR and HackRF drivers after 8-bit → float normalisation.

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// I/Q sample
// ─────────────────────────────────────────────────────────────────────────────

/// A single complex I/Q sample from the SDR front-end.
///
/// Values are normalised to [−1.0, +1.0] after ADC conversion.
/// Full-scale amplitude = 1.0 corresponds to the ADC clipping level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IqSample {
    /// In-phase component.
    pub i: f32,
    /// Quadrature component.
    pub q: f32,
}

impl IqSample {
    /// Instantaneous power (linear, dimensionless).
    #[inline]
    pub fn power(&self) -> f32 {
        self.i * self.i + self.q * self.q
    }

    /// Instantaneous amplitude (envelope).
    #[inline]
    pub fn amplitude(&self) -> f32 {
        self.power().sqrt()
    }

    /// Zero sample (DC offset = 0).
    pub const ZERO: Self = Self { i: 0.0, q: 0.0 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the sliding-window power estimator.
#[derive(Debug, Clone)]
pub struct PowerEstimatorConfig {
    /// Number of I/Q samples in the averaging window.
    ///
    /// At 2.4 MHz sample rate, 2400 samples = 1 ms.
    /// Default: 2048 samples ≈ 0.85 ms.
    pub window_size: usize,
    /// Minimum representable power (dBm) — returned when signal is below noise floor.
    pub floor_dbm: f64,
}

impl Default for PowerEstimatorConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            floor_dbm: -120.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PowerEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Sliding-window mean power estimator over I/Q samples.
///
/// Maintains a running sum for O(1) updates.
#[derive(Debug)]
pub struct PowerEstimator {
    config: PowerEstimatorConfig,
    window: VecDeque<f32>,
    running_sum: f64,
}

impl PowerEstimator {
    pub fn new(config: PowerEstimatorConfig) -> Self {
        Self {
            window: VecDeque::with_capacity(config.window_size + 1),
            running_sum: 0.0,
            config,
        }
    }

    /// Push a batch of samples and return the updated mean power in dBm.
    ///
    /// Processes all samples in the batch; the window slides forward by
    /// `batch.len()` positions.
    pub fn update(&mut self, batch: &[IqSample]) -> f64 {
        for &s in batch {
            let p = s.power();
            self.window.push_back(p);
            self.running_sum += p as f64;
            if self.window.len() > self.config.window_size {
                let evicted = self.window.pop_front().unwrap();
                self.running_sum -= evicted as f64;
            }
        }
        self.mean_power_dbm()
    }

    /// Current mean power in dBm (without pushing new samples).
    pub fn mean_power_dbm(&self) -> f64 {
        if self.window.is_empty() {
            return self.config.floor_dbm;
        }
        let mean = self.running_sum / self.window.len() as f64;
        if mean < 1e-20 {
            return self.config.floor_dbm;
        }
        10.0 * mean.log10()
    }

    /// Current mean power in linear scale.
    pub fn mean_power_linear(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.running_sum / self.window.len() as f64
    }

    /// Number of samples currently in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_estimator() -> PowerEstimator {
        PowerEstimator::new(PowerEstimatorConfig::default())
    }

    #[test]
    fn zero_samples_returns_floor() {
        let est = make_estimator();
        assert_eq!(est.mean_power_dbm(), -120.0);
    }

    #[test]
    fn unit_amplitude_sinusoid_power_is_minus3_dbm() {
        // Full-scale sinusoid: I = cos(θ), Q = sin(θ) → power = 1.0 per sample
        // Mean power = 1.0 → P_dBm = 10·log10(1.0) = 0 dBm
        let mut est = PowerEstimator::new(PowerEstimatorConfig {
            window_size: 1024,
            ..Default::default()
        });
        let samples: Vec<IqSample> = (0..1024)
            .map(|k| {
                let phase = 2.0 * std::f32::consts::PI * 0.1 * k as f32;
                IqSample {
                    i: phase.cos(),
                    q: phase.sin(),
                }
            })
            .collect();
        let p_dbm = est.update(&samples);
        // I² + Q² = cos²+sin² = 1.0 per sample → mean = 1.0 → 0 dBm
        assert_abs_diff_eq!(p_dbm, 0.0, epsilon = 0.1);
    }

    #[test]
    fn half_amplitude_gives_minus6_dbm() {
        // Amplitude 0.5: power = 0.25 → P_dBm = 10·log10(0.25) ≈ −6.02 dBm
        let mut est = PowerEstimator::new(PowerEstimatorConfig {
            window_size: 512,
            ..Default::default()
        });
        let samples: Vec<IqSample> = (0..512)
            .map(|k| {
                let phase = 2.0 * std::f32::consts::PI * 0.1 * k as f32;
                IqSample {
                    i: 0.5 * phase.cos(),
                    q: 0.5 * phase.sin(),
                }
            })
            .collect();
        let p_dbm = est.update(&samples);
        assert_abs_diff_eq!(p_dbm, -6.02, epsilon = 0.1);
    }

    #[test]
    fn window_eviction_keeps_size_bounded() {
        let mut est = PowerEstimator::new(PowerEstimatorConfig {
            window_size: 100,
            ..Default::default()
        });
        let samples: Vec<IqSample> = vec![IqSample { i: 1.0, q: 0.0 }; 300];
        est.update(&samples);
        assert_eq!(est.window_len(), 100);
    }

    #[test]
    fn running_sum_is_consistent_with_window() {
        let mut est = PowerEstimator::new(PowerEstimatorConfig {
            window_size: 50,
            ..Default::default()
        });
        let samples: Vec<IqSample> = (0..200)
            .map(|k| IqSample {
                i: (k as f32) * 0.01,
                q: 0.0,
            })
            .collect();
        est.update(&samples);
        // Recompute from scratch
        let expected: f64 =
            est.window.iter().map(|&p| p as f64).sum::<f64>() / est.window.len() as f64;
        assert_abs_diff_eq!(est.mean_power_linear(), expected, epsilon = 1e-6);
    }
}
