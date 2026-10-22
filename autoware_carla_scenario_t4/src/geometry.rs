//! Rigid transforms in a right-handed frame: positions in metres, rotations
//! as unit quaternions `[w, x, y, z]`, T4's (and nuScenes') order.

pub type Vec3 = [f64; 3];
pub type Quat = [f64; 4];

/// `q` scaled to unit length; the identity for a zero quaternion.
pub fn normalize(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n == 0.0 || !n.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn conjugate(q: Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}

fn multiply(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// `v` rotated by the unit quaternion `q`.
pub fn rotate(q: Quat, v: Vec3) -> Vec3 {
    // v' = v + 2w(u x v) + 2u x (u x v), u the vector part.
    let (w, u) = (q[0], [q[1], q[2], q[3]]);
    let t = cross(u, v);
    let t = [2.0 * t[0], 2.0 * t[1], 2.0 * t[2]];
    let ut = cross(u, t);
    [
        v[0] + w * t[0] + ut[0],
        v[1] + w * t[1] + ut[1],
        v[2] + w * t[2] + ut[2],
    ]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A frame placed in its parent: `parent = rotation * child + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Pose {
    pub fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation: normalize(rotation),
        }
    }

    /// The point `p`, given in this frame, in the parent frame.
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let r = rotate(self.rotation, p);
        [
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        ]
    }

    /// The parent frame placed in this one.
    pub fn inverse(&self) -> Self {
        let rotation = conjugate(self.rotation);
        let t = rotate(rotation, self.translation);
        Self {
            translation: [-t[0], -t[1], -t[2]],
            rotation,
        }
    }

    /// `child`, given in this frame, in the parent frame.
    pub fn compose(&self, child: &Pose) -> Self {
        Self {
            translation: self.apply(child.translation),
            rotation: normalize(multiply(self.rotation, child.rotation)),
        }
    }
}

/// A 3D box as T4 annotates it: its centre and orientation, and its size as
/// `[width, length, height]` -- length along the box's x axis, width along y.
#[derive(Clone, Copy, Debug)]
pub struct OrientedBox {
    pose: Pose,
    inverse: Pose,
    half: Vec3,
    radius_sq: f64,
}

impl OrientedBox {
    pub fn new(pose: Pose, size: Vec3) -> Self {
        let half = [size[1] / 2.0, size[0] / 2.0, size[2] / 2.0];
        Self {
            pose,
            inverse: pose.inverse(),
            half,
            radius_sq: half[0] * half[0] + half[1] * half[1] + half[2] * half[2],
        }
    }

    /// Whether `p` (in the box's parent frame) is inside the box, faces included.
    pub fn contains(&self, p: Vec3) -> bool {
        let c = self.pose.translation;
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        // Most points are nowhere near a given box: the bounding sphere first.
        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > self.radius_sq {
            return false;
        }
        let local = self.inverse.apply(p);
        local[0].abs() <= self.half[0]
            && local[1].abs() <= self.half[1]
            && local[2].abs() <= self.half[2]
    }
}

/// The quaternion of a rotation of `yaw` radians about z.
pub fn yaw_quaternion(yaw: f64) -> Quat {
    [(yaw / 2.0).cos(), 0.0, 0.0, (yaw / 2.0).sin()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn rotation_about_z_turns_x_into_y() {
        let q = yaw_quaternion(std::f64::consts::FRAC_PI_2);
        assert!(close(rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    }

    #[test]
    fn inverse_undoes_apply() {
        let pose = Pose::new([1.0, -2.0, 3.0], [0.9, 0.1, -0.3, 0.2]);
        let p = [4.0, 5.0, -6.0];
        assert!(close(pose.inverse().apply(pose.apply(p)), p));
    }

    #[test]
    fn compose_applies_child_then_parent() {
        let parent = Pose::new([10.0, 0.0, 0.0], yaw_quaternion(1.0));
        let child = Pose::new([0.0, 2.0, 1.0], yaw_quaternion(-0.4));
        let p = [1.0, 1.0, 1.0];
        assert!(close(
            parent.compose(&child).apply(p),
            parent.apply(child.apply(p))
        ));
    }

    #[test]
    fn box_length_runs_along_its_heading() {
        // 4 m long, 2 m wide, heading along +y.
        let b = OrientedBox::new(
            Pose::new([0.0, 0.0, 0.0], yaw_quaternion(std::f64::consts::FRAC_PI_2)),
            [2.0, 4.0, 1.5],
        );
        assert!(b.contains([0.0, 1.9, 0.0]));
        assert!(!b.contains([1.9, 0.0, 0.0]));
        assert!(b.contains([0.9, 0.0, 0.7]));
        assert!(!b.contains([0.0, 0.0, 0.8]));
    }
}
