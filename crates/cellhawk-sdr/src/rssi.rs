//! RSSI extraction via pilot sequence correlation (§3.1, §2.1).
//!
//! ## Method
//!
//! LTE Cell Reference Signals (CRS) are known BPSK sequences transmitted at
//! fixed subcarrier positions.  By correlating the received signal with the
//! known pilot, we extract the Reference Signal Received Power (RSRP), which
//! is the standard LTE RSSI metric used by the LDPL model.
//!
//! ```text
//! C(k) = Σ_{n=0}^{N-1}  r_n · conj(p_n)     [complex correlation]
//! RSRP  = |C(k)|² / N                         [normalised power]
//! RSSI  = 10 · log₁₀(RSRP)                   [dBm, relative to unit impedance]
//! ```
//!
//! ## Pilot generation
//!
//! The default pilot is a length-63 Gold code (standard LTE CRS seed),
//! mapped to BPSK: 0 → +1, 1 → −1.  In production this would be replaced
//! with the actual Cell-ID-specific CRS sequence.

use crate::power::IqSample;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the RSSI extractor.
#[derive(Debug, Clone)]
pub struct RssiExtractorConfig {
    /// Pilot sequence length.  Must be ≤ the minimum expected batch size.
    pub pilot_length: usize,
    /// Minimum RSSI output (dBm) — returned when correlation is below noise.
    pub floor_rssi_dbm: f64,
}

impl Default for RssiExtractorConfig {
    fn default() -> Self {
        Self {
            pilot_length: 63,
            floor_rssi_dbm: -120.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RssiExtractor
// ─────────────────────────────────────────────────────────────────────────────

/// Pilot-correlation RSSI extractor.
///
/// Maintains a pre-computed pilot sequence and performs complex correlation
/// against each incoming batch of I/Q samples.
#[derive(Debug)]
pub struct RssiExtractor {
    config: RssiExtractorConfig,
    /// Pre-computed pilot sequence (BPSK: I ∈ {+1, −1}, Q = 0).
    pilot: Vec<IqSample>,
}

impl RssiExtractor {
    /// Construct with the given configuration.
    ///
    /// Generates a deterministic Gold-code-derived pilot sequence of length
    /// `config.pilot_length`.
    pub fn new(config: RssiExtractorConfig) -> Self {
        let pilot = Self::generate_pilot(config.pilot_length);
        Self { config, pilot }
    }

    /// Construct with a custom pilot sequence (for testing or Cell-ID-specific CRS).
    pub fn with_pilot(config: RssiExtractorConfig, pilot: Vec<IqSample>) -> Self {
        Self { config, pilot }
    }

    /// Extract RSSI (dBm) from a batch of I/Q samples via pilot correlation.
    ///
    /// If the batch is shorter than the pilot, returns `floor_rssi_dbm`.
    pub fn extract(&self, received: &[IqSample]) -> f64 {
        let n = self.pilot.len();
        if received.len() < n {
            return self.config.floor_rssi_dbm;
        }

        // Complex correlation: C = Σ r_k · conj(p_k)
        // For BPSK pilot (Q=0): conj(p_k) = p_k, so C_i = Σ r_i·p_i + r_q·0
        let mut corr_i = 0.0_f64;
        let mut corr_q = 0.0_f64;
        for (r, p) in received[..n].iter().zip(self.pilot.iter()) {
            // r · conj(p) = (r_i + j·r_q)(p_i − j·p_q)
            //             = (r_i·p_i + r_q·p_q) + j(r_q·p_i − r_i·p_q)
            corr_i += (r.i * p.i + r.q * p.q) as f64;
            corr_q += (r.q * p.i - r.i * p.q) as f64;
        }

        // Normalised correlation power
        let corr_power = (corr_i * corr_i + corr_q * corr_q) / (n as f64);

        if corr_power < 1e-20 {
            return self.config.floor_rssi_dbm;
        }

        10.0 * corr_power.log10()
    }

    /// The pilot sequence used for correlation.
    pub fn pilot(&self) -> &[IqSample] {
        &self.pilot
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Generate a deterministic BPSK pilot sequence using a linear feedback
    /// shift register (LFSR) with polynomial x^7 + x^6 + 1 (Gold code basis).
    ///
    /// This is a simplified CRS-like sequence; production code would use the
    /// 3GPP TS 36.211 §6.10.1 Gold code with Cell-ID seeding.
    fn generate_pilot(length: usize) -> Vec<IqSample> {
        let mut lfsr: u32 = 0x5F; // non-zero seed
        (0..length)
            .map(|_| {
                // Fibonacci LFSR: taps at positions 7 and 6
                let bit = ((lfsr >> 6) ^ (lfsr >> 5)) & 1;
                lfsr = (lfsr >> 1) | (bit << 6);
                // BPSK mapping: 0 → +1, 1 → −1
                let symbol = if bit == 0 { 1.0_f32 } else { -1.0_f32 };
                IqSample { i: symbol, q: 0.0 }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_extractor() -> RssiExtractor {
        RssiExtractor::new(RssiExtractorConfig::default())
    }

    #[test]
    fn pilot_has_correct_length() {
        let ext = make_extractor();
        assert_eq!(ext.pilot().len(), 63);
    }

    #[test]
    fn pilot_is_bpsk_unit_amplitude() {
        let ext = make_extractor();
        for s in ext.pilot() {
            assert_abs_diff_eq!(s.amplitude(), 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(s.q, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn perfect_pilot_gives_maximum_rssi() {
        // Received signal = pilot → correlation = N → RSRP = N²/N = N
        let ext = make_extractor();
        let received: Vec<IqSample> = ext.pilot().to_vec();
        let rssi = ext.extract(&received);
        // RSRP = N = 63 → RSSI = 10·log10(63) ≈ 18.0 dBm
        assert!(
            rssi > 15.0,
            "perfect pilot RSSI={rssi:.2} dBm, expected > 15"
        );
    }

    #[test]
    fn inverted_pilot_gives_same_power() {
        // Inverted pilot: correlation magnitude is the same (phase shift only)
        let ext = make_extractor();
        let inverted: Vec<IqSample> = ext
            .pilot()
            .iter()
            .map(|s| IqSample { i: -s.i, q: -s.q })
            .collect();
        let rssi_normal = ext.extract(ext.pilot());
        let rssi_inverted = ext.extract(&inverted);
        assert_abs_diff_eq!(rssi_normal, rssi_inverted, epsilon = 0.1);
    }

    #[test]
    fn short_batch_returns_floor() {
        let ext = make_extractor();
        let short: Vec<IqSample> = vec![IqSample { i: 1.0, q: 0.0 }; 10];
        let rssi = ext.extract(&short);
        assert_eq!(rssi, -120.0);
    }

    #[test]
    fn zero_signal_returns_floor() {
        let ext = make_extractor();
        let zeros: Vec<IqSample> = vec![IqSample::ZERO; 128];
        let rssi = ext.extract(&zeros);
        assert_eq!(rssi, -120.0);
    }

    #[test]
    fn custom_pilot_round_trip() {
        // Custom pilot: all +1 BPSK
        let pilot: Vec<IqSample> = vec![IqSample { i: 1.0, q: 0.0 }; 32];
        let cfg = RssiExtractorConfig {
            pilot_length: 32,
            ..Default::default()
        };
        let ext = RssiExtractor::with_pilot(cfg, pilot.clone());
        let rssi = ext.extract(&pilot);
        // Correlation = 32, RSRP = 32²/32 = 32 → RSSI = 10·log10(32) ≈ 15.05 dBm
        assert_abs_diff_eq!(rssi, 10.0 * 32.0_f64.log10(), epsilon = 0.1);
    }
}
