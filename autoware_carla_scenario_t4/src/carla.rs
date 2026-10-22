//! CARLA's world frame (left-handed: x forward/east, y right/south, z up,
//! degrees) into the dataset's map frame (right-handed, radians): `y` flipped,
//! CARLA's `yaw * pitch * roll` rebuilt with the yaw and pitch negated, and
//! the map's offset added -- `coordinate.transform.to_map_frame` in Python.

use crate::geometry::{Pose, Quat, Vec3};

/// A CARLA transform `[x, y, z, roll, pitch, yaw]` in the map frame.
pub fn pose(transform: &[f64; 6], offset: Vec3) -> Pose {
    let [x, y, z, roll, pitch, yaw] = *transform;
    Pose::new(
        [x + offset[0], -y + offset[1], z + offset[2]],
        rotation(roll, pitch, yaw),
    )
}

/// `Rz(-yaw) * Ry(-pitch) * Rx(roll)`, CARLA's angles in degrees.
pub fn rotation(roll: f64, pitch: f64, yaw: f64) -> Quat {
    let (r, p, y) = (
        roll.to_radians() / 2.0,
        -pitch.to_radians() / 2.0,
        -yaw.to_radians() / 2.0,
    );
    let (cr, sr) = (r.cos(), r.sin());
    let (cp, sp) = (p.cos(), p.sin());
    let (cy, sy) = (y.cos(), y.sin());
    [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ]
}

/// A CARLA world vector (a velocity, an acceleration) in the map frame.
pub fn vector(v: &[f64; 3]) -> Vec3 {
    [v[0], -v[1], v[2]]
}

/// A CARLA angular velocity (deg/s) in the map frame, in rad/s: a
/// pseudovector, so the flip of `y` negates `x` and `z` instead.
pub fn angular_velocity(w: &[f64; 3]) -> Vec3 {
    [-w[0].to_radians(), w[1].to_radians(), -w[2].to_radians()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::rotate;

    fn close(a: Vec3, b: Vec3) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-9, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn a_carla_yaw_to_the_right_is_a_map_yaw_to_the_right() {
        // CARLA yaw +90: facing +y (south, to the right); in the map, -y.
        let q = rotation(0.0, 0.0, 90.0);
        close(rotate(q, [1.0, 0.0, 0.0]), [0.0, -1.0, 0.0]);
    }

    #[test]
    fn a_carla_pitch_up_points_the_nose_up() {
        // CARLA pitch +30 raises the nose (UE's convention).
        let q = rotation(0.0, 30.0, 0.0);
        let f = rotate(q, [1.0, 0.0, 0.0]);
        assert!(f[2] > 0.49 && f[2] < 0.51);
    }

    #[test]
    fn positions_flip_y_and_add_the_offset() {
        let p = pose(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [100.0, 200.0, 10.0]);
        close(p.translation, [101.0, 198.0, 13.0]);
    }
}
