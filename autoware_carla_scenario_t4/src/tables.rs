//! The records of a T4 dataset's `annotation/*.json`, field for field as
//! t4-devkit's schema tables (`t4_devkit/schema/tables/`) read them.

use serde::Serialize;

use crate::geometry::{Quat, Vec3};

/// A record's key: 32 hex digits, as t4-devkit's `token_hex(16)`.
///
/// Derived from a seed rather than drawn at random, so the same run written
/// twice is the same dataset.
pub struct TokenGen {
    state: u64,
}

impl TokenGen {
    pub fn new(seed: &str) -> Self {
        // FNV-1a: the seed only has to be spread over the state.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in seed.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        Self { state: hash }
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_token(&mut self) -> String {
        format!("{:016x}{:016x}", self.next_u64(), self.next_u64())
    }
}

#[derive(Serialize)]
pub struct Attribute {
    pub token: String,
    pub name: String,
    pub description: String,
}

#[derive(Serialize)]
pub struct CalibratedSensor {
    pub token: String,
    pub sensor_token: String,
    pub translation: Vec3,
    pub rotation: Quat,
    /// 3x3, or empty for a sensor that is not a camera.
    pub camera_intrinsic: Vec<[f64; 3]>,
    /// OpenCV's coefficients, or empty for a sensor that is not a camera.
    pub camera_distortion: Vec<f64>,
}

#[derive(Serialize)]
pub struct Category {
    pub token: String,
    pub name: String,
    pub description: String,
    /// Null for every category (REC007: all unique, or all null).
    pub index: Option<u32>,
}

#[derive(Serialize)]
pub struct EgoPose {
    pub token: String,
    pub translation: Vec3,
    pub rotation: Quat,
    pub timestamp: u64,
    /// `(vx, vy, vz, yaw_rate, pitch_rate, roll_rate)` in the ego's frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub twist: Option<[f64; 6]>,
    /// `(ax, ay, az)` in the ego's frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceleration: Option<Vec3>,
}

#[derive(Serialize)]
pub struct Instance {
    pub token: String,
    pub category_token: String,
    pub instance_name: String,
    pub nbr_annotations: usize,
    pub first_annotation_token: String,
    pub last_annotation_token: String,
}

#[derive(Serialize)]
pub struct Log {
    pub token: String,
    pub logfile: String,
    pub vehicle: String,
    /// `YYYY-MM-DD-HH-mm-ss` (t4-devkit spells the field `data_captured`).
    pub data_captured: String,
    pub location: String,
}

#[derive(Serialize)]
pub struct Map {
    pub token: String,
    pub log_tokens: Vec<String>,
    pub category: String,
    pub filename: String,
}

#[derive(Serialize)]
pub struct Sample {
    pub token: String,
    pub timestamp: u64,
    pub scene_token: String,
    pub next: String,
    pub prev: String,
}

#[derive(Serialize)]
pub struct SampleAnnotation {
    pub token: String,
    pub sample_token: String,
    pub instance_token: String,
    pub attribute_tokens: Vec<String>,
    pub visibility_token: String,
    pub translation: Vec3,
    /// `[width, length, height]`.
    pub size: Vec3,
    pub rotation: Quat,
    pub num_lidar_pts: u64,
    pub num_radar_pts: u64,
    pub next: String,
    pub prev: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity: Option<Vec3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceleration: Option<Vec3>,
}

#[derive(Serialize)]
pub struct SampleData {
    pub token: String,
    pub sample_token: String,
    pub ego_pose_token: String,
    pub calibrated_sensor_token: String,
    pub filename: String,
    pub fileformat: String,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
    pub is_key_frame: bool,
    pub next: String,
    pub prev: String,
    pub is_valid: bool,
}

#[derive(Serialize)]
pub struct Scene {
    pub token: String,
    pub name: String,
    pub description: String,
    pub log_token: String,
    pub nbr_samples: usize,
    pub first_sample_token: String,
    pub last_sample_token: String,
}

#[derive(Serialize)]
pub struct Sensor {
    pub token: String,
    pub channel: String,
    pub modality: String,
}

#[derive(Serialize)]
pub struct Visibility {
    pub token: String,
    pub level: String,
    pub description: String,
}

#[derive(Serialize, Default, Clone)]
pub struct AdditionalInfo {
    pub speed: Option<f64>,
}

#[derive(Serialize)]
pub struct VehicleState {
    pub token: String,
    pub timestamp: u64,
    pub accel_pedal: Option<f64>,
    pub brake_pedal: Option<f64>,
    pub steer_pedal: Option<f64>,
    pub steering_tire_angle: Option<f64>,
    pub steering_wheel_angle: Option<f64>,
    pub shift_state: Option<String>,
    pub additional_info: Option<AdditionalInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_32_hex_digits_and_repeat_with_the_seed() {
        let mut a = TokenGen::new("run");
        let mut b = TokenGen::new("run");
        let first = a.next_token();
        assert_eq!(first.len(), 32);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(first, b.next_token());
        assert_ne!(a.next_token(), first);
        assert_ne!(TokenGen::new("other").next_token(), first);
    }
}
