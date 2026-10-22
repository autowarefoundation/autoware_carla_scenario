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

pub fn conjugate(q: Quat) -> Quat {
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

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
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

/// Boxes binned on a grid over the x-y plane, so that a point is tested
/// against the few boxes around it rather than all of them: a sweep is ~10^5
/// points, a frame tens of boxes.
pub struct BoxGrid {
    boxes: Vec<OrientedBox>,
    origin: [f64; 2],
    cell: f64,
    nx: usize,
    ny: usize,
    /// Each cell's boxes (indices into `boxes`), row-major.
    cells: Vec<Vec<u32>>,
}

impl BoxGrid {
    /// The most cells a grid has; a wider spread of boxes gets larger cells.
    const MAX_CELLS: f64 = 65_536.0;

    /// Bin `boxes` into cells of `cell` metres (or larger, see `MAX_CELLS`).
    pub fn new(boxes: Vec<OrientedBox>, cell: f64) -> Self {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for b in &boxes {
            let (c, r) = (b.pose.translation, b.radius_sq.sqrt());
            for k in 0..2 {
                lo[k] = lo[k].min(c[k] - r);
                hi[k] = hi[k].max(c[k] + r);
            }
        }
        if boxes.is_empty() {
            return Self {
                boxes,
                origin: [0.0; 2],
                cell,
                nx: 0,
                ny: 0,
                cells: Vec::new(),
            };
        }
        let area = (hi[0] - lo[0]).max(cell) * (hi[1] - lo[1]).max(cell);
        let cell = cell.max((area / Self::MAX_CELLS).sqrt());
        let nx = ((hi[0] - lo[0]) / cell).floor() as usize + 1;
        let ny = ((hi[1] - lo[1]) / cell).floor() as usize + 1;
        let mut cells = vec![Vec::new(); nx * ny];
        for (i, b) in boxes.iter().enumerate() {
            let (c, r) = (b.pose.translation, b.radius_sq.sqrt());
            let x0 = ((c[0] - r - lo[0]) / cell).floor() as usize;
            let x1 = (((c[0] + r - lo[0]) / cell).floor() as usize).min(nx - 1);
            let y0 = ((c[1] - r - lo[1]) / cell).floor() as usize;
            let y1 = (((c[1] + r - lo[1]) / cell).floor() as usize).min(ny - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    cells[y * nx + x].push(i as u32);
                }
            }
        }
        Self {
            boxes,
            origin: lo,
            cell,
            nx,
            ny,
            cells,
        }
    }

    /// Add to `counts[i]` the points of `points` (rows of `stride` floats,
    /// `x, y, z` first) inside box `i`.
    pub fn count(&self, points: &[f32], stride: usize, counts: &mut [u64]) {
        if self.cells.is_empty() {
            return;
        }
        let inv = 1.0 / self.cell;
        for row in points.chunks_exact(stride) {
            let (x, y) = (f64::from(row[0]), f64::from(row[1]));
            let (fx, fy) = ((x - self.origin[0]) * inv, (y - self.origin[1]) * inv);
            if !(fx >= 0.0 && fy >= 0.0) {
                continue;
            }
            let (cx, cy) = (fx as usize, fy as usize);
            if cx >= self.nx || cy >= self.ny {
                continue;
            }
            let candidates = &self.cells[cy * self.nx + cx];
            if candidates.is_empty() {
                continue;
            }
            let p = [x, y, f64::from(row[2])];
            for &i in candidates {
                if self.boxes[i as usize].contains(p) {
                    counts[i as usize] += 1;
                }
            }
        }
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
    fn a_grid_counts_what_testing_every_box_counts() {
        let boxes: Vec<OrientedBox> = (0..30)
            .map(|i| {
                let i = f64::from(i);
                OrientedBox::new(
                    Pose::new(
                        [i * 3.7 - 50.0, (i * 1.9).sin() * 20.0, 0.5],
                        yaw_quaternion(i),
                    ),
                    [1.8, 4.5, 1.5],
                )
            })
            .collect();
        let mut points = Vec::new();
        let mut seed = 1u64;
        for _ in 0..20_000 {
            let mut next = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f64 / f64::from(1u32 << 31)) as f32
            };
            points.extend_from_slice(&[
                next() * 120.0 - 60.0,
                next() * 50.0 - 25.0,
                next() * 2.0 - 0.5,
                1.0,
            ]);
        }
        let mut brute = vec![0u64; boxes.len()];
        for row in points.chunks_exact(4) {
            let p = [f64::from(row[0]), f64::from(row[1]), f64::from(row[2])];
            for (i, b) in boxes.iter().enumerate() {
                if b.contains(p) {
                    brute[i] += 1;
                }
            }
        }
        let mut gridded = vec![0u64; boxes.len()];
        BoxGrid::new(boxes, 2.0).count(&points, 4, &mut gridded);
        assert_eq!(gridded, brute);
        assert!(brute.iter().sum::<u64>() > 0);
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
