//! Handover timing benchmark (Gap 8).
//!
//! The paper (§5.4) claims tier handover completes in < 150 ms end-to-end.
//! This benchmark measures the hot path:
//!
//!   predict(IMU) → update_jnr(above_threshold) → tier transition
//!
//! Run with:
//!   cargo bench --bench handover -p cellhawk-ekf

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cellhawk_ekf::filter::{CellHawkEkf, EkfConfig};
use cellhawk_types::{EnuPosition, ImuMeasurement};
use nalgebra::Vector3;

fn make_ekf() -> CellHawkEkf {
    CellHawkEkf::new(
        EnuPosition { east_m: 0.0, north_m: 0.0, up_m: 50.0 },
        0.0,
        EkfConfig::default(),
    )
}

fn imu_zero(t: f64) -> ImuMeasurement {
    ImuMeasurement {
        acceleration_body: Vector3::zeros(),
        angular_velocity_body: Vector3::zeros(),
        timestamp_s: t,
    }
}

/// Single predict + JNR update cycle (the 10 Hz hot path).
fn bench_predict_jnr(c: &mut Criterion) {
    c.bench_function("predict_then_jnr_update", |b| {
        b.iter_batched(
            make_ekf,
            |mut ekf| {
                ekf.predict(black_box(&imu_zero(0.1)));
                ekf.update_jnr(black_box(8.0), black_box(0.1));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Full T1→T2 handover: 10 predict+JNR cycles until tier flips.
fn bench_full_handover(c: &mut Criterion) {
    c.bench_function("full_t1_to_t2_handover", |b| {
        b.iter_batched(
            make_ekf,
            |mut ekf| {
                for i in 1..=10_u32 {
                    let t = f64::from(i) * 0.1;
                    ekf.predict(black_box(&imu_zero(t)));
                    ekf.update_jnr(black_box(10.0), black_box(t));
                }
                black_box(ekf.active_tier())
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_predict_jnr, bench_full_handover);
criterion_main!(benches);
