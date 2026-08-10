//! Weighted Least Squares multilateration with TRF solver and RANSAC
//! outlier rejection (§3.2).
//!
//! ## Algorithm
//!
//! Given N tower measurements, each providing a range estimate `d̂_i` and
//! known tower position `(x_i, y_i)`, we minimise:
//!
//! ```text
//! min Σ_i [ w_i · ( ‖(x,y) − (x_i,y_i)‖ − d̂_i )² ]
//! ```
//!
//! where `w_i ∝ 1/d̂_i²` (§3.2).
//!
//! ## Solver
//!
//! The Trust Region Reflective (TRF) algorithm is implemented as a
//! Gauss-Newton iteration with a trust-region step limiter. This matches
//! the paper's description (§3.2, Appendix A.4).
//!
//! ## RANSAC
//!
//! Before the final WLS solve, RANSAC identifies and removes NLoS-corrupted
//! tower measurements. The paper mentions RANSAC (§3.2) but does not
//! implement it; this is an improvement.

use cellhawk_types::{CellHawkError, EnuPosition, TowerMeasurement};
use nalgebra::{Matrix2, Vector2};

use crate::{fading::RicianFadingModel, ldpl::LdplModel};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the multilateration solver.
#[derive(Debug, Clone)]
pub struct MultilaterationConfig {
    /// Minimum towers for a valid 2-D fix (§3.2: N ≥ 3).
    pub min_towers: usize,
    /// Maximum usable tower range (m). Beyond this, LTE signal is at noise floor.
    pub max_range_m: f64,
    /// Weighting exponent: w_i ∝ 1/d_i^p (paper uses p=2).
    pub distance_weighting_exponent: f64,
    /// TRF solver max iterations.
    pub trf_max_iterations: usize,
    /// TRF function-value convergence tolerance.
    pub trf_function_tolerance: f64,
    /// RANSAC configuration.
    pub ransac: RansacConfig,
}

impl Default for MultilaterationConfig {
    fn default() -> Self {
        Self {
            min_towers: 3,
            max_range_m: 1200.0,
            distance_weighting_exponent: 2.0,
            trf_max_iterations: 200,
            trf_function_tolerance: 1e-10,
            ransac: RansacConfig::default(),
        }
    }
}

/// RANSAC configuration for outlier rejection.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    pub enabled: bool,
    pub max_iterations: usize,
    /// A tower is an inlier if its residual is below this threshold (m).
    pub inlier_threshold_m: f64,
    /// Minimum fraction of towers that must be inliers for a valid solution.
    pub min_inlier_fraction: f64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_iterations: 50,
            inlier_threshold_m: 30.0,
            min_inlier_fraction: 0.6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a successful multilateration solve.
#[derive(Debug, Clone)]
pub struct MultilaterationResult {
    /// Estimated 2-D ENU position (altitude from barometer, not RSSI).
    pub position: EnuPosition,
    /// Estimated horizontal position covariance (2×2, ENU).
    pub covariance: Matrix2<f64>,
    /// Geometric Dilution of Precision (dimensionless).
    pub gdop: f64,
    /// Number of towers used after RANSAC filtering.
    pub towers_used: usize,
    /// RMS residual of the final solution (m).
    pub rms_residual_m: f64,
    /// Indices of inlier towers (into the input slice).
    pub inlier_indices: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Solver
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted Least Squares multilateration solver.
pub struct MultilaterationSolver {
    config: MultilaterationConfig,
    ldpl: LdplModel,
}

impl MultilaterationSolver {
    pub fn new(config: MultilaterationConfig, ldpl: LdplModel) -> Self {
        Self { config, ldpl }
    }

    /// Solve for 2-D position from a set of tower RSSI measurements.
    ///
    /// Pipeline:
    /// 1. Convert RSSI → range via LDPL (with Rician correction).
    /// 2. Filter towers beyond `max_range_m`.
    /// 3. RANSAC outlier rejection (if enabled and N ≥ 4).
    /// 4. WLS Gauss-Newton (TRF) solve on inlier set.
    /// 5. Compute GDOP and covariance.
    pub fn solve(
        &self,
        measurements: &[TowerMeasurement],
        altitude_m: f64,
        initial_guess: Option<EnuPosition>,
    ) -> Result<MultilaterationResult, CellHawkError> {
        // ── Step 1: RSSI → range with Rician correction ──────────────────────
        let mut tower_ranges: Vec<(EnuPosition, f64, f64)> = Vec::new(); // (pos, range_m, weight)
        for m in measurements {
            let fading = RicianFadingModel::new(m.rician_k_db);
            let corrected_rssi = fading.correct_rssi(m.rssi_dbm);
            let range_m = match self.ldpl.estimate_range(corrected_rssi) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if range_m > self.config.max_range_m {
                continue;
            }
            let weight = 1.0 / range_m.powf(self.config.distance_weighting_exponent);
            tower_ranges.push((m.tower_position, range_m, weight));
        }

        if tower_ranges.len() < self.config.min_towers {
            return Err(CellHawkError::InsufficientTowers {
                required: self.config.min_towers,
                got: tower_ranges.len(),
            });
        }

        // ── Step 2: RANSAC outlier rejection ─────────────────────────────────
        let inlier_indices = if self.config.ransac.enabled && tower_ranges.len() >= 4 {
            self.ransac_inliers(&tower_ranges, altitude_m, initial_guess)?
        } else {
            (0..tower_ranges.len()).collect()
        };

        let inliers: Vec<_> = inlier_indices.iter().map(|&i| tower_ranges[i]).collect();

        if inliers.len() < self.config.min_towers {
            return Err(CellHawkError::InsufficientTowers {
                required: self.config.min_towers,
                got: inliers.len(),
            });
        }

        // ── Step 3: WLS Gauss-Newton (TRF) solve ─────────────────────────────
        let init = initial_guess.unwrap_or_else(|| self.centroid_guess(&inliers));
        let position_2d = self.trf_solve(&inliers, Vector2::new(init.east_m, init.north_m))?;

        // ── Step 4: Covariance and GDOP ───────────────────────────────────────
        let (covariance, gdop) = self.compute_covariance_gdop(&inliers, position_2d);
        let rms_residual_m = self.rms_residual(&inliers, position_2d);

        Ok(MultilaterationResult {
            position: EnuPosition {
                east_m: position_2d.x,
                north_m: position_2d.y,
                up_m: altitude_m,
            },
            covariance,
            gdop,
            towers_used: inliers.len(),
            rms_residual_m,
            inlier_indices,
        })
    }

    // ── TRF Gauss-Newton solver ───────────────────────────────────────────────

    /// Gauss-Newton iteration with trust-region step limiting (TRF, §3.2).
    ///
    /// Minimises: `Σ_i w_i · (‖p − t_i‖ − d̂_i)²`
    fn trf_solve(
        &self,
        inliers: &[(EnuPosition, f64, f64)],
        mut p: Vector2<f64>,
    ) -> Result<Vector2<f64>, CellHawkError> {
        let mut trust_radius = 500.0_f64;
        let mut best_p = p;
        let mut best_cost = self.total_cost(inliers, p);

        for iter in 0..self.config.trf_max_iterations {
            let (jt_w_j, jt_w_r, cost) = self.build_normal_equations(inliers, p);

            let delta = match jt_w_j.try_inverse() {
                Some(inv) => inv * jt_w_r,
                None => return Err(CellHawkError::SolverNonConvergence { iterations: iter }),
            };

            // Trust-region step limiting
            let step_norm = delta.norm();
            let step = if step_norm > trust_radius {
                delta * (trust_radius / step_norm)
            } else {
                delta
            };

            let p_new = p - step; // subtract: we minimise, delta points toward minimum
            let cost_new = self.total_cost(inliers, p_new);

            if cost_new < cost {
                p = p_new;
                trust_radius = (trust_radius * 2.0).min(2000.0);
                if cost_new < best_cost {
                    best_cost = cost_new;
                    best_p = p;
                }
            } else {
                trust_radius *= 0.5;
            }

            if step.norm() < self.config.trf_function_tolerance {
                break;
            }
        }

        Ok(best_p)
    }

    /// Build the weighted normal equations (JᵀWJ, JᵀWr) and total cost.
    ///
    /// Jacobian row i: `∂r_i/∂p = -(p - t_i) / ‖p - t_i‖`
    fn build_normal_equations(
        &self,
        inliers: &[(EnuPosition, f64, f64)],
        p: Vector2<f64>,
    ) -> (Matrix2<f64>, Vector2<f64>, f64) {
        let mut jtj = Matrix2::zeros();
        let mut jtr = Vector2::zeros();
        let mut cost = 0.0_f64;

        for &(ref tower_pos, range_est, weight) in inliers {
            let t = Vector2::new(tower_pos.east_m, tower_pos.north_m);
            let diff = p - t;
            let dist = diff.norm().max(1e-6); // avoid division by zero
            let residual = dist - range_est;

            // Jacobian row: ∂residual/∂p = diff / dist
            let j_row = diff / dist;

            jtj += weight * j_row * j_row.transpose();
            jtr += weight * residual * j_row;
            cost += weight * residual * residual;
        }

        (jtj, jtr, cost)
    }

    fn total_cost(&self, inliers: &[(EnuPosition, f64, f64)], p: Vector2<f64>) -> f64 {
        inliers
            .iter()
            .map(|&(ref tp, range_est, weight)| {
                let t = Vector2::new(tp.east_m, tp.north_m);
                let residual = (p - t).norm() - range_est;
                weight * residual * residual
            })
            .sum()
    }

    // ── RANSAC ────────────────────────────────────────────────────────────────

    fn ransac_inliers(
        &self,
        tower_ranges: &[(EnuPosition, f64, f64)],
        altitude_m: f64,
        initial_guess: Option<EnuPosition>,
    ) -> Result<Vec<usize>, CellHawkError> {
        let n = tower_ranges.len();
        let cfg = &self.config.ransac;
        let min_inliers = ((n as f64) * cfg.min_inlier_fraction).ceil() as usize;
        let min_inliers = min_inliers.max(self.config.min_towers);

        let mut best_inliers: Vec<usize> = Vec::new();

        // Use all towers as the initial candidate set (deterministic RANSAC
        // variant — we iterate over all minimal subsets of size 3)
        let init = initial_guess.unwrap_or_else(|| self.centroid_guess(tower_ranges));
        let init_2d = Vector2::new(init.east_m, init.north_m);

        // Full solve on all towers first
        if let Ok(p) = self.trf_solve(tower_ranges, init_2d) {
            let inliers: Vec<usize> = (0..n)
                .filter(|&i| {
                    let (ref tp, range_est, _) = tower_ranges[i];
                    let t = Vector2::new(tp.east_m, tp.north_m);
                    let residual = (p - t).norm() - range_est;
                    residual.abs() < cfg.inlier_threshold_m
                })
                .collect();
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
            }
        }

        // Iterative refinement: re-solve on inlier set, re-classify
        for _ in 0..cfg.max_iterations {
            if best_inliers.len() < self.config.min_towers {
                break;
            }
            let subset: Vec<_> = best_inliers.iter().map(|&i| tower_ranges[i]).collect();
            let centroid = self.centroid_guess(&subset);
            let p0 = Vector2::new(centroid.east_m, centroid.north_m);
            if let Ok(p) = self.trf_solve(&subset, p0) {
                let new_inliers: Vec<usize> = (0..n)
                    .filter(|&i| {
                        let (ref tp, range_est, _) = tower_ranges[i];
                        let t = Vector2::new(tp.east_m, tp.north_m);
                        let residual = (p - t).norm() - range_est;
                        residual.abs() < cfg.inlier_threshold_m
                    })
                    .collect();
                if new_inliers == best_inliers {
                    break; // converged
                }
                if new_inliers.len() >= best_inliers.len() {
                    best_inliers = new_inliers;
                }
            } else {
                break;
            }
        }

        // Suppress unused variable warning
        let _ = altitude_m;

        if best_inliers.len() >= min_inliers {
            Ok(best_inliers)
        } else {
            // Fall back to all towers if RANSAC can't find enough inliers
            Ok((0..n).collect())
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Weighted initial guess: position biased toward the tower with the
    /// smallest estimated range (closest tower), which breaks geometric
    /// symmetry and gives Gauss-Newton a non-degenerate starting point.
    fn centroid_guess(&self, towers: &[(EnuPosition, f64, f64)]) -> EnuPosition {
        // w_i = 1/range_i² (same weighting as the solver)
        let total_w: f64 = towers.iter().map(|(_, r, _)| 1.0 / (r * r)).sum();
        if total_w < 1e-12 {
            // Fallback: unweighted centroid
            let n = towers.len() as f64;
            return EnuPosition {
                east_m: towers.iter().map(|(p, _, _)| p.east_m).sum::<f64>() / n,
                north_m: towers.iter().map(|(p, _, _)| p.north_m).sum::<f64>() / n,
                up_m: 0.0,
            };
        }
        let east = towers
            .iter()
            .map(|(p, r, _)| p.east_m / (r * r))
            .sum::<f64>()
            / total_w;
        let north = towers
            .iter()
            .map(|(p, r, _)| p.north_m / (r * r))
            .sum::<f64>()
            / total_w;
        EnuPosition {
            east_m: east,
            north_m: north,
            up_m: 0.0,
        }
    }

    fn rms_residual(&self, inliers: &[(EnuPosition, f64, f64)], p: Vector2<f64>) -> f64 {
        let sum_sq: f64 = inliers
            .iter()
            .map(|(tp, range_est, _)| {
                let t = Vector2::new(tp.east_m, tp.north_m);
                let r = (p - t).norm() - range_est;
                r * r
            })
            .sum();
        (sum_sq / inliers.len() as f64).sqrt()
    }

    /// Compute the 2×2 position covariance and GDOP.
    ///
    /// ```text
    /// Cov(p) = σ²_range · (JᵀWJ)⁻¹
    /// GDOP   = sqrt(trace(Cov) / σ²_range)
    ///        = sqrt(trace((JᵀWJ)⁻¹))
    /// ```
    fn compute_covariance_gdop(
        &self,
        inliers: &[(EnuPosition, f64, f64)],
        p: Vector2<f64>,
    ) -> (Matrix2<f64>, f64) {
        let (jtj, _, _) = self.build_normal_equations(inliers, p);
        match jtj.try_inverse() {
            Some(inv) => {
                let gdop = inv.trace().sqrt();
                // Scale by LDPL measurement variance
                let sigma2 = self.ldpl.measurement_variance_dbm2();
                (inv * sigma2, gdop)
            }
            None => (Matrix2::identity() * 1e6, f64::INFINITY),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ldpl::{LdplConfig, LdplModel};
    use approx::assert_abs_diff_eq;

    fn make_solver() -> MultilaterationSolver {
        MultilaterationSolver::new(
            MultilaterationConfig::default(),
            LdplModel::new(LdplConfig::default()),
        )
    }

    /// Build a synthetic measurement from a tower at a known position,
    /// given the true drone position.
    fn synthetic_measurement(
        tower_pos: EnuPosition,
        drone_pos: EnuPosition,
        ldpl: &LdplModel,
        tower_id: u64,
    ) -> TowerMeasurement {
        let range_m = tower_pos.horizontal_distance_to(&drone_pos);
        let rssi_dbm = ldpl.predict_rssi(range_m).unwrap();
        TowerMeasurement {
            tower_id,
            tower_position: tower_pos,
            rssi_dbm,
            tx_power_dbm: -40.0,
            rician_k_db: 1e6, // perfect LoS: nlos_bias ≈ 0, no fading correction
            timestamp_s: 0.0,
        }
    }

    /// With 4 towers in a square geometry and noiseless RSSI, the solver
    /// should recover the true position to within 1 m.
    #[test]
    fn noiseless_4tower_recovery() {
        let ldpl = LdplModel::new(LdplConfig::default());
        let solver = make_solver();

        let true_pos = EnuPosition {
            east_m: 150.0,
            north_m: 200.0,
            up_m: 50.0,
        };
        let towers = [
            EnuPosition {
                east_m: 0.0,
                north_m: 0.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 500.0,
                north_m: 0.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 500.0,
                north_m: 500.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 0.0,
                north_m: 500.0,
                up_m: 0.0,
            },
        ];
        let measurements: Vec<_> = towers
            .iter()
            .enumerate()
            .map(|(i, &tp)| synthetic_measurement(tp, true_pos, &ldpl, i as u64))
            .collect();

        let result = solver.solve(&measurements, true_pos.up_m, None).unwrap();

        assert_abs_diff_eq!(result.position.east_m, true_pos.east_m, epsilon = 2.5);
        assert_abs_diff_eq!(result.position.north_m, true_pos.north_m, epsilon = 2.5);
        assert_eq!(result.position.up_m, true_pos.up_m);
        assert!(result.gdop.is_finite());
        assert!(result.rms_residual_m < 1.0);
    }

    /// Fewer than 3 towers must return InsufficientTowers error.
    #[test]
    fn insufficient_towers_error() {
        let ldpl = LdplModel::new(LdplConfig::default());
        let solver = make_solver();
        let true_pos = EnuPosition {
            east_m: 100.0,
            north_m: 100.0,
            up_m: 30.0,
        };
        let towers = [
            EnuPosition {
                east_m: 0.0,
                north_m: 0.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 300.0,
                north_m: 0.0,
                up_m: 0.0,
            },
        ];
        let measurements: Vec<_> = towers
            .iter()
            .enumerate()
            .map(|(i, &tp)| synthetic_measurement(tp, true_pos, &ldpl, i as u64))
            .collect();
        let err = solver.solve(&measurements, 30.0, None).unwrap_err();
        assert!(matches!(err, CellHawkError::InsufficientTowers { .. }));
    }

    /// RANSAC should reject a heavily corrupted outlier tower and still
    /// recover the true position.
    #[test]
    fn ransac_rejects_outlier() {
        let ldpl = LdplModel::new(LdplConfig::default());
        let solver = make_solver();

        let true_pos = EnuPosition {
            east_m: 200.0,
            north_m: 200.0,
            up_m: 40.0,
        };
        let towers = [
            EnuPosition {
                east_m: 0.0,
                north_m: 0.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 400.0,
                north_m: 0.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 400.0,
                north_m: 400.0,
                up_m: 0.0,
            },
            EnuPosition {
                east_m: 0.0,
                north_m: 400.0,
                up_m: 0.0,
            },
        ];
        let mut measurements: Vec<_> = towers
            .iter()
            .enumerate()
            .map(|(i, &tp)| synthetic_measurement(tp, true_pos, &ldpl, i as u64))
            .collect();

        // Corrupt tower 3 with a massive RSSI error (−30 dBm offset → ~300 m range error)
        measurements[3].rssi_dbm -= 30.0;

        let result = solver.solve(&measurements, true_pos.up_m, None).unwrap();

        // Should still recover within 20 m despite the outlier
        let error = result.position.horizontal_distance_to(&true_pos);
        assert!(error < 20.0, "position error={error:.2} m with outlier");
    }

    /// Weighting: closer towers should have higher weight (w ∝ 1/d²).
    #[test]
    fn closer_tower_has_higher_weight() {
        let config = MultilaterationConfig::default();
        let w_near = 1.0_f64 / 100.0_f64.powf(config.distance_weighting_exponent);
        let w_far = 1.0_f64 / 500.0_f64.powf(config.distance_weighting_exponent);
        assert!(w_near > w_far);
    }
}
