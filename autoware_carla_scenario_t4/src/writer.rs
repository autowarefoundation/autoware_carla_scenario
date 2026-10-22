//! Writes one scene as a T4 dataset, frame by frame.
//!
//! ```text
//! <data_root>/
//!   annotation/*.json   the tables, written by `finish`
//!   data/<CHANNEL>/     one file per frame: NNNNN.pcd.bin (LiDAR), NNNNN.jpg (camera)
//!   map/                lanelet2_map.osm, when given one
//!   status.json
//! ```
//!
//! Every frame is a key frame (a `sample`): the caller opens one with
//! `begin_frame` at the ego's pose, adds the boxes and the sensor data of that
//! instant, and closes it with `end_frame`, which counts each box's LiDAR
//! points.  The sensor files are encoded and written on worker threads, so a
//! frame costs its caller the bookkeeping only; `finish` waits for them and
//! writes the tables.

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use jpeg_encoder::{ColorType, Encoder};
use serde::Serialize;

use crate::carla;
use crate::geometry::{conjugate, cross, rotate, BoxGrid, OrientedBox, Pose, Quat, Vec3};
use crate::tables::*;

pub type Result<T> = std::result::Result<T, String>;

/// What is known of the run the scene is.
#[derive(Clone, Debug, Default)]
pub struct SceneInfo {
    pub name: String,
    pub description: String,
    pub logfile: String,
    pub vehicle: String,
    /// `YYYY-MM-DD-HH-mm-ss`.
    pub date_captured: String,
    pub location: String,
    /// Seeds the records' tokens: the same seed and frames, the same dataset.
    pub token_seed: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Lidar,
    Camera,
}

impl Modality {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "lidar" => Ok(Self::Lidar),
            "camera" => Ok(Self::Camera),
            other => Err(format!(
                "unknown sensor modality {other:?} (expected \"lidar\" or \"camera\")"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Lidar => "lidar",
            Self::Camera => "camera",
        }
    }
}

/// The pixel layout of a camera frame handed to `add_camera`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Bgra,
    Bgr,
    Rgba,
    Rgb,
}

impl PixelFormat {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "bgra" => Ok(Self::Bgra),
            "bgr" => Ok(Self::Bgr),
            "rgba" => Ok(Self::Rgba),
            "rgb" => Ok(Self::Rgb),
            other => Err(format!(
                "unknown pixel format {other:?} (expected bgra, bgr, rgba or rgb)"
            )),
        }
    }

    pub fn channels(self) -> usize {
        match self {
            Self::Bgra | Self::Rgba => 4,
            Self::Bgr | Self::Rgb => 3,
        }
    }

    fn color_type(self) -> ColorType {
        match self {
            Self::Bgra => ColorType::Bgra,
            Self::Bgr => ColorType::Bgr,
            Self::Rgba => ColorType::Rgba,
            Self::Rgb => ColorType::Rgb,
        }
    }
}

/// The ego's state at a frame, in the dataset's (right-handed) map frame.
#[derive(Clone, Copy, Debug)]
pub struct EgoState {
    pub pose: Pose,
    pub twist: Option<[f64; 6]>,
    pub acceleration: Option<Vec3>,
}

/// One road user's box at a frame, in the map frame.
#[derive(Clone, Debug)]
pub struct Annotation {
    /// Identifies the road user across frames (its CARLA actor id).
    pub instance_id: u64,
    pub category: String,
    pub instance_name: Option<String>,
    pub attributes: Vec<String>,
    pub translation: Vec3,
    /// `[width, length, height]`.
    pub size: Vec3,
    pub rotation: Quat,
    pub velocity: Option<Vec3>,
    pub acceleration: Option<Vec3>,
}

/// The ego's controls at a frame (the optional `vehicle_state` table).
#[derive(Clone, Debug, Default)]
pub struct VehicleControls {
    pub accel_pedal: Option<f64>,
    pub brake_pedal: Option<f64>,
    pub steer_pedal: Option<f64>,
    pub steering_tire_angle: Option<f64>,
    pub steering_wheel_angle: Option<f64>,
    pub shift_state: Option<String>,
    pub speed: Option<f64>,
}

/// What a finished dataset holds.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub data_root: String,
    pub samples: usize,
    pub sample_data: usize,
    pub sample_annotations: usize,
    pub instances: usize,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
}

const SHIFT_STATES: [&str; 7] = [
    "PARK", "REVERSE", "NEUTRAL", "HIGH", "FORWARD", "LOW", "NONE",
];

struct SensorEntry {
    channel: String,
    modality: Modality,
    calibrated_token: String,
    /// The sensor in the ego's frame (`base_link`).
    pose: Pose,
    /// The channel's latest sample_data, which the next one follows.
    last: Option<usize>,
    resolution: Option<(u32, u32)>,
    /// How many values the channel's last frame held (a LiDAR's).
    points_hint: usize,
}

struct InstanceEntry {
    token: String,
    category_token: String,
    name: String,
    first: Option<usize>,
    last: Option<usize>,
    count: usize,
}

struct OpenFrame {
    sample: usize,
    ego_pose: usize,
    ego: Pose,
    /// The frame's annotations: `annotations[first_annotation..]`.
    first_annotation: usize,
    /// Each LiDAR channel's points this frame, `(x, y, z, intensity)` rows in
    /// the channel's frame: every sweep added to a channel is concatenated.
    lidar: Vec<(usize, Vec<f32>)>,
    /// The channels with data this frame.
    channels: Vec<usize>,
}

/// A registered road user: what its boxes need that does not change.
struct ActorEntry {
    category: String,
    name: String,
    /// The box's centre in the actor's (right-handed) frame.
    center: Vec3,
    /// `[width, length, height]`.
    size: Vec3,
}

/// The CARLA frame the boxes and the ego are given in, and where they go.
#[derive(Clone, Copy, Debug, Default)]
pub struct CarlaFrame {
    /// Added to every position once `y` is flipped: the map's MGRS offset and
    /// elevation offset.
    pub map_offset: Vec3,
    /// From the ego actor's origin to its rear axle along its x axis (negative
    /// when the axle is behind it): where the ego's `base_link` is.
    pub rear_axle_offset_m: f64,
    /// Annotate the road users within this distance of the ego only.
    pub max_distance_m: Option<f64>,
}

enum Job {
    /// A frame's points of one channel: written as T4's `(x, y, z,
    /// intensity, ring = -1)`, and counted in `boxes` (in the channel's frame,
    /// each with its annotation's index).
    Lidar {
        path: PathBuf,
        points: Vec<f32>,
        boxes: Vec<(usize, OrientedBox)>,
    },
    Camera {
        path: PathBuf,
        pixels: Vec<u8>,
        width: u16,
        height: u16,
        format: PixelFormat,
        quality: u8,
    },
    /// An already encoded file, written as it is.
    Bytes { path: PathBuf, bytes: Vec<u8> },
}

/// What the workers report back: their errors, and each box's LiDAR points.
#[derive(Default)]
struct Outcome {
    errors: Vec<String>,
    counts: Vec<(usize, u64)>,
}

/// Point buffers the workers are done with, for the next frames to fill: a
/// buffer of a sweep's size taken fresh from the allocator is new pages,
/// which cost the caller's thread more to fault in than the copy itself.
type Pool = Mutex<Vec<Vec<f32>>>;

/// How many spare point buffers the pool keeps.
const POOL_SIZE: usize = 8;

impl Job {
    fn run(self, outcome: &Mutex<Outcome>, pool: &Pool) -> Result<()> {
        match self {
            Job::Lidar {
                path,
                points,
                boxes,
            } => {
                let mut points = points;
                if !points.iter().all(|v| v.is_finite()) {
                    points = points
                        .chunks_exact(4)
                        .filter(|row| row[..3].iter().all(|v| v.is_finite()))
                        .flatten()
                        .copied()
                        .collect();
                }
                let mut bytes = Vec::with_capacity(points.len() / 4 * 20);
                for row in points.chunks_exact(4) {
                    for value in row {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    bytes.extend_from_slice(&(-1.0f32).to_le_bytes());
                }
                fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
                let (indices, boxes): (Vec<usize>, Vec<OrientedBox>) = boxes.into_iter().unzip();
                let mut counts = vec![0u64; boxes.len()];
                BoxGrid::new(boxes, 2.0).count(&points, 4, &mut counts);
                if let Ok(mut outcome) = outcome.lock() {
                    outcome.counts.extend(indices.into_iter().zip(counts));
                }
                if let Ok(mut pool) = pool.lock() {
                    if pool.len() < POOL_SIZE {
                        points.clear();
                        pool.push(points);
                    }
                }
                Ok(())
            }
            Job::Camera {
                path,
                pixels,
                width,
                height,
                format,
                quality,
            } => Encoder::new_file(&path, quality)
                .and_then(|encoder| encoder.encode(&pixels, width, height, format.color_type()))
                .map_err(|e| format!("{}: {e}", path.display())),
            Job::Bytes { path, bytes } => {
                fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
            }
        }
    }
}

/// Threads that write the sensor files and count the boxes' points.
struct Workers {
    sender: Option<SyncSender<Job>>,
    handles: Vec<JoinHandle<()>>,
    outcome: Arc<Mutex<Outcome>>,
    pool: Arc<Pool>,
}

impl Workers {
    fn new(count: usize, queue: usize) -> Self {
        let (sender, receiver) = sync_channel::<Job>(queue.max(1));
        let receiver: Arc<Mutex<Receiver<Job>>> = Arc::new(Mutex::new(receiver));
        let outcome = Arc::new(Mutex::new(Outcome::default()));
        let pool: Arc<Pool> = Arc::new(Mutex::new(Vec::new()));
        let handles = (0..count.max(1))
            .map(|_| {
                let receiver = Arc::clone(&receiver);
                let outcome = Arc::clone(&outcome);
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || loop {
                    let job = match receiver.lock() {
                        Ok(guard) => guard.recv(),
                        Err(_) => return,
                    };
                    let Ok(job) = job else { return };
                    if let Err(error) = job.run(&outcome, &pool) {
                        if let Ok(mut outcome) = outcome.lock() {
                            outcome.errors.push(error);
                        }
                    }
                })
            })
            .collect();
        Self {
            sender: Some(sender),
            handles,
            outcome,
            pool,
        }
    }

    /// A point buffer for `capacity` values: a spare one when there is one.
    fn buffer(&self, capacity: usize) -> Vec<f32> {
        let mut buffer = self
            .pool
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_default();
        buffer.reserve(capacity);
        buffer
    }

    fn submit(&self, job: Job) -> Result<()> {
        match &self.sender {
            // Blocks while the queue is full: memory stays bounded when the
            // disk is slower than the simulation.
            Some(sender) => sender
                .send(job)
                .map_err(|_| "the T4 writer's worker threads have stopped".to_string()),
            None => Err("the T4 writer is finished".to_string()),
        }
    }

    /// Wait for every queued job; what they reported.
    fn join(&mut self) -> Outcome {
        self.sender = None;
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.outcome
            .lock()
            .map(|mut o| std::mem::take(&mut *o))
            .unwrap_or_default()
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.join();
    }
}

/// Writes one scene into `data_root` (see the module's documentation).
pub struct T4Writer {
    root: PathBuf,
    info: SceneInfo,
    tokens: TokenGen,
    jpeg_quality: u8,
    workers: Workers,
    finished: bool,

    log_token: String,
    scene_token: String,
    map_filename: String,
    visibility_token: String,

    sensors: Vec<SensorEntry>,
    sensor_records: Vec<Sensor>,
    calibrated_sensors: Vec<CalibratedSensor>,
    categories: Vec<Category>,
    attributes: Vec<Attribute>,
    instances: HashMap<u64, InstanceEntry>,
    /// Instance ids in the order they were first seen.
    instance_order: Vec<u64>,

    samples: Vec<Sample>,
    ego_poses: Vec<EgoPose>,
    sample_data: Vec<SampleData>,
    annotations: Vec<SampleAnnotation>,
    vehicle_states: Vec<VehicleState>,

    frame: Option<OpenFrame>,

    carla: CarlaFrame,
    actors: HashMap<u64, ActorEntry>,
}

fn check_finite(what: &str, values: &[f64]) -> Result<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(format!("{what} must be finite, got {values:?}"))
    }
}

fn write_json<T: Serialize>(path: &Path, records: &T) -> Result<()> {
    let file = fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut out, records)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    out.write_all(b"\n")
        .and_then(|_| out.flush())
        .map_err(|e| format!("{}: {e}", path.display()))
}

impl T4Writer {
    /// Start a dataset in `data_root`, which must not hold one already.
    ///
    /// `workers` threads write the sensor files, with up to `queue` files
    /// waiting for them.
    pub fn create(
        data_root: impl Into<PathBuf>,
        info: SceneInfo,
        jpeg_quality: u8,
        workers: usize,
        queue: usize,
    ) -> Result<Self> {
        Self::create_in(
            data_root,
            info,
            CarlaFrame::default(),
            jpeg_quality,
            workers,
            queue,
        )
    }

    /// `create`, with the CARLA frame `begin_frame_carla` and
    /// `add_carla_boxes` read their poses in.
    pub fn create_in(
        data_root: impl Into<PathBuf>,
        info: SceneInfo,
        carla: CarlaFrame,
        jpeg_quality: u8,
        workers: usize,
        queue: usize,
    ) -> Result<Self> {
        check_finite("the map offset", &carla.map_offset)?;
        if carla.max_distance_m.is_some_and(|d| d.is_nan() || d <= 0.0) {
            return Err("max_distance_m must be positive".to_string());
        }
        let root: PathBuf = data_root.into();
        if root.join("annotation").exists() {
            return Err(format!(
                "{} already holds a T4 dataset (annotation/ exists)",
                root.display()
            ));
        }
        if !(1..=100).contains(&jpeg_quality) {
            return Err(format!("jpeg_quality must be 1-100, got {jpeg_quality}"));
        }
        for dir in ["annotation", "data", "map"] {
            fs::create_dir_all(root.join(dir))
                .map_err(|e| format!("{}: {e}", root.join(dir).display()))?;
        }
        let mut tokens = TokenGen::new(&info.token_seed);
        let log_token = tokens.next_token();
        let scene_token = tokens.next_token();
        let visibility_token = tokens.next_token();
        Ok(Self {
            root,
            info,
            tokens,
            jpeg_quality,
            workers: Workers::new(workers, queue),
            finished: false,
            log_token,
            scene_token,
            map_filename: String::new(),
            visibility_token,
            sensors: Vec::new(),
            sensor_records: Vec::new(),
            calibrated_sensors: Vec::new(),
            categories: Vec::new(),
            attributes: Vec::new(),
            instances: HashMap::new(),
            instance_order: Vec::new(),
            samples: Vec::new(),
            ego_poses: Vec::new(),
            sample_data: Vec::new(),
            annotations: Vec::new(),
            vehicle_states: Vec::new(),
            frame: None,
            carla,
            actors: HashMap::new(),
        })
    }

    pub fn data_root(&self) -> &Path {
        &self.root
    }

    pub fn frames(&self) -> usize {
        self.samples.len()
    }

    fn check_open(&self) -> Result<()> {
        if self.finished {
            Err("the T4 dataset is finished".to_string())
        } else {
            Ok(())
        }
    }

    /// Copy the Lanelet2 map into `map/lanelet2_map.osm`, which the `map`
    /// record then names.
    pub fn add_lanelet2_map(&mut self, source: &Path) -> Result<()> {
        self.check_open()?;
        let target = self.root.join("map").join("lanelet2_map.osm");
        fs::copy(source, &target).map_err(|e| format!("{}: {e}", source.display()))?;
        self.map_filename = "map/lanelet2_map.osm".to_string();
        Ok(())
    }

    /// Declare a sensor on the ego: `channel` names it (`LIDAR_TOP`,
    /// `CAM_FRONT`, ...), `pose` places it in the ego's frame.  A camera has
    /// its 3x3 `intrinsic`; its frame is the optical one (x right, y down,
    /// z forward).
    pub fn add_sensor(
        &mut self,
        channel: &str,
        modality: Modality,
        pose: Pose,
        intrinsic: Option<[[f64; 3]; 3]>,
        distortion: Option<Vec<f64>>,
    ) -> Result<()> {
        self.check_open()?;
        if channel.is_empty() || channel.contains(['/', '\\']) || channel.starts_with('.') {
            return Err(format!("invalid sensor channel {channel:?}"));
        }
        if self.sensors.iter().any(|s| s.channel == channel) {
            return Err(format!("sensor {channel:?} is already declared"));
        }
        if !self.samples.is_empty() {
            return Err("sensors must be declared before the first frame".to_string());
        }
        check_finite("a sensor's translation", &pose.translation)?;
        check_finite("a sensor's rotation", &pose.rotation)?;
        let (intrinsic, distortion) = match modality {
            Modality::Camera => {
                let k = intrinsic
                    .ok_or_else(|| format!("camera {channel:?} needs its intrinsic matrix"))?;
                let distortion = distortion.unwrap_or_else(|| vec![0.0; 5]);
                if ![4, 5, 8, 12, 14].contains(&distortion.len()) {
                    return Err(format!(
                        "camera {channel:?}'s distortion has {} coefficients \
                         (OpenCV's models have 4, 5, 8, 12 or 14)",
                        distortion.len()
                    ));
                }
                (k.to_vec(), distortion)
            }
            Modality::Lidar => (Vec::new(), Vec::new()),
        };
        fs::create_dir_all(self.root.join("data").join(channel))
            .map_err(|e| format!("{}: {e}", self.root.join("data").join(channel).display()))?;
        let sensor_token = self.tokens.next_token();
        let calibrated_token = self.tokens.next_token();
        self.sensor_records.push(Sensor {
            token: sensor_token.clone(),
            channel: channel.to_string(),
            modality: modality.name().to_string(),
        });
        self.calibrated_sensors.push(CalibratedSensor {
            token: calibrated_token.clone(),
            sensor_token,
            translation: pose.translation,
            rotation: pose.rotation,
            camera_intrinsic: intrinsic,
            camera_distortion: distortion,
        });
        self.sensors.push(SensorEntry {
            channel: channel.to_string(),
            modality,
            calibrated_token,
            pose,
            last: None,
            resolution: None,
            points_hint: 0,
        });
        Ok(())
    }

    /// Open a frame at `timestamp_us` (Unix microseconds, later than the
    /// last frame's) with the ego in `ego`.
    pub fn begin_frame(&mut self, timestamp_us: u64, ego: EgoState) -> Result<()> {
        self.check_open()?;
        if self.frame.is_some() {
            return Err("a frame is already open (end_frame was not called)".to_string());
        }
        if self.sensors.is_empty() {
            return Err("declare at least one sensor before the first frame".to_string());
        }
        if let Some(last) = self.samples.last() {
            if timestamp_us <= last.timestamp {
                return Err(format!(
                    "frame timestamps must increase: {timestamp_us} after {}",
                    last.timestamp
                ));
            }
        }
        check_finite("the ego's translation", &ego.pose.translation)?;
        check_finite("the ego's rotation", &ego.pose.rotation)?;
        if let Some(twist) = ego.twist {
            check_finite("the ego's twist", &twist)?;
        }
        if let Some(acceleration) = ego.acceleration {
            check_finite("the ego's acceleration", &acceleration)?;
        }
        let sample_token = self.tokens.next_token();
        let prev = match self.samples.last_mut() {
            Some(last) => {
                last.next = sample_token.clone();
                last.token.clone()
            }
            None => String::new(),
        };
        self.samples.push(Sample {
            token: sample_token,
            timestamp: timestamp_us,
            scene_token: self.scene_token.clone(),
            next: String::new(),
            prev,
        });
        self.ego_poses.push(EgoPose {
            token: self.tokens.next_token(),
            translation: ego.pose.translation,
            rotation: ego.pose.rotation,
            timestamp: timestamp_us,
            twist: ego.twist,
            acceleration: ego.acceleration,
        });
        self.frame = Some(OpenFrame {
            sample: self.samples.len() - 1,
            ego_pose: self.ego_poses.len() - 1,
            ego: ego.pose,
            first_annotation: self.annotations.len(),
            lidar: Vec::new(),
            channels: Vec::new(),
        });
        Ok(())
    }

    fn open_frame(&self) -> Result<&OpenFrame> {
        self.frame
            .as_ref()
            .ok_or_else(|| "no frame is open (call begin_frame first)".to_string())
    }

    fn category_token(&mut self, name: &str) -> String {
        if let Some(c) = self.categories.iter().find(|c| c.name == name) {
            return c.token.clone();
        }
        let token = self.tokens.next_token();
        self.categories.push(Category {
            token: token.clone(),
            name: name.to_string(),
            description: String::new(),
            index: None,
        });
        token
    }

    fn attribute_token(&mut self, name: &str) -> String {
        if let Some(a) = self.attributes.iter().find(|a| a.name == name) {
            return a.token.clone();
        }
        let token = self.tokens.next_token();
        self.attributes.push(Attribute {
            token: token.clone(),
            name: name.to_string(),
            description: String::new(),
        });
        token
    }

    /// Add a road user's box to the open frame.
    pub fn add_annotation(&mut self, annotation: Annotation) -> Result<()> {
        self.check_open()?;
        let (sample, first_annotation) = {
            let frame = self.open_frame()?;
            (frame.sample, frame.first_annotation)
        };
        if annotation.category.is_empty() {
            return Err("an annotation needs a category".to_string());
        }
        check_finite("a box's translation", &annotation.translation)?;
        check_finite("a box's rotation", &annotation.rotation)?;
        check_finite("a box's size", &annotation.size)?;
        if annotation.size.iter().any(|s| *s < 0.0) {
            return Err(format!(
                "a box's size must not be negative: {:?}",
                annotation.size
            ));
        }
        for v in [annotation.velocity, annotation.acceleration]
            .into_iter()
            .flatten()
        {
            check_finite("a box's velocity and acceleration", &v)?;
        }
        let instance_id = annotation.instance_id;
        if let Some(entry) = self.instances.get(&instance_id) {
            if entry.last.is_some_and(|last| last >= first_annotation) {
                return Err(format!(
                    "instance {instance_id} is annotated twice in one frame"
                ));
            }
        }
        let category_token = self.category_token(&annotation.category);
        let attribute_tokens = annotation
            .attributes
            .iter()
            .map(|name| self.attribute_token(name))
            .collect();
        if !self.instances.contains_key(&instance_id) {
            let name = annotation
                .instance_name
                .clone()
                .unwrap_or_else(|| format!("{}_{instance_id}", annotation.category));
            self.instances.insert(
                instance_id,
                InstanceEntry {
                    token: self.tokens.next_token(),
                    category_token: category_token.clone(),
                    name,
                    first: None,
                    last: None,
                    count: 0,
                },
            );
            self.instance_order.push(instance_id);
        }
        let token = self.tokens.next_token();
        let index = self.annotations.len();
        let entry = self
            .instances
            .get_mut(&instance_id)
            .expect("inserted above");
        if entry.category_token != category_token {
            return Err(format!(
                "instance {instance_id} changed category to {:?}",
                annotation.category
            ));
        }
        let prev = match entry.last {
            Some(last) => {
                self.annotations[last].next = token.clone();
                self.annotations[last].token.clone()
            }
            None => String::new(),
        };
        entry.first.get_or_insert(index);
        entry.last = Some(index);
        entry.count += 1;
        let instance_token = entry.token.clone();
        self.annotations.push(SampleAnnotation {
            token,
            sample_token: self.samples[sample].token.clone(),
            instance_token,
            attribute_tokens,
            visibility_token: self.visibility_token.clone(),
            translation: annotation.translation,
            size: annotation.size,
            rotation: crate::geometry::normalize(annotation.rotation),
            num_lidar_pts: 0,
            num_radar_pts: 0,
            next: String::new(),
            prev,
            velocity: annotation.velocity,
            acceleration: annotation.acceleration,
        });
        Ok(())
    }

    /// Register a road user `add_carla_boxes` annotates: its `category`, its
    /// box's centre in its own (right-handed) frame and its size `[width,
    /// length, height]`.  Registering it again replaces what was known.
    pub fn register_actor(
        &mut self,
        id: u64,
        category: &str,
        center: Vec3,
        size: Vec3,
        name: Option<String>,
    ) -> Result<()> {
        if category.is_empty() {
            return Err("an actor needs a category".to_string());
        }
        check_finite("an actor's box centre", &center)?;
        check_finite("an actor's box size", &size)?;
        self.actors.insert(
            id,
            ActorEntry {
                category: category.to_string(),
                name: name.unwrap_or_else(|| format!("{category}_{id}")),
                center,
                size,
            },
        );
        Ok(())
    }

    /// Whether `id` is registered.
    pub fn knows_actor(&self, id: u64) -> bool {
        self.actors.contains_key(&id)
    }

    /// `begin_frame` with the ego actor's state as CARLA reports it:
    /// `transform` `[x, y, z, roll, pitch, yaw]`, its velocity, angular
    /// velocity (deg/s) and acceleration, in CARLA's world frame.  The pose is
    /// its rear axle's (`CarlaFrame::rear_axle_offset_m`) in the map frame,
    /// the twist and acceleration in its own frame.
    pub fn begin_frame_carla(
        &mut self,
        timestamp_us: u64,
        transform: [f64; 6],
        velocity: [f64; 3],
        angular_velocity: [f64; 3],
        acceleration: [f64; 3],
    ) -> Result<()> {
        for values in [&transform[..], &velocity, &angular_velocity, &acceleration] {
            check_finite("the ego's state", values)?;
        }
        let actor = carla::pose(&transform, self.carla.map_offset);
        let base = actor.compose(&Pose::new(
            [self.carla.rear_axle_offset_m, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ));
        let omega = carla::angular_velocity(&angular_velocity);
        let lever = [
            base.translation[0] - actor.translation[0],
            base.translation[1] - actor.translation[1],
            base.translation[2] - actor.translation[2],
        ];
        let spin = cross(omega, lever);
        let v = carla::vector(&velocity);
        let v = [v[0] + spin[0], v[1] + spin[1], v[2] + spin[2]];
        let to_local = conjugate(base.rotation);
        let v = rotate(to_local, v);
        let w = rotate(to_local, omega);
        let a = rotate(to_local, carla::vector(&acceleration));
        self.begin_frame(
            timestamp_us,
            EgoState {
                pose: base,
                twist: Some([v[0], v[1], v[2], w[2], w[1], w[0]]),
                acceleration: Some(a),
            },
        )
    }

    /// Annotate the registered road users `ids` in the open frame, from their
    /// CARLA states: `states` is `[N, 9]` row-major, `x, y, z, roll, pitch,
    /// yaw, vx, vy, vz` in CARLA's world frame.  Ids not registered, and road
    /// users beyond `CarlaFrame::max_distance_m` of the ego, are skipped.
    /// How many were annotated.
    pub fn add_carla_boxes(&mut self, ids: &[u64], states: &[f64]) -> Result<usize> {
        self.check_open()?;
        let ego = self.open_frame()?.ego.translation;
        if states.len() != ids.len() * 9 {
            return Err(format!(
                "{} ids need {} state values (9 each), got {}",
                ids.len(),
                ids.len() * 9,
                states.len()
            ));
        }
        let limit_sq = self.carla.max_distance_m.map(|d| d * d);
        let mut added = 0;
        for (id, state) in ids.iter().zip(states.chunks_exact(9)) {
            let Some(entry) = self.actors.get(id) else {
                continue;
            };
            let transform = [state[0], state[1], state[2], state[3], state[4], state[5]];
            let pose = carla::pose(&transform, self.carla.map_offset);
            let center = pose.apply(entry.center);
            if let Some(limit_sq) = limit_sq {
                let d = [center[0] - ego[0], center[1] - ego[1], center[2] - ego[2]];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > limit_sq {
                    continue;
                }
            }
            let annotation = Annotation {
                instance_id: *id,
                category: entry.category.clone(),
                instance_name: Some(entry.name.clone()),
                attributes: Vec::new(),
                translation: center,
                size: entry.size,
                rotation: pose.rotation,
                velocity: Some(carla::vector(&[state[6], state[7], state[8]])),
                acceleration: None,
            };
            self.add_annotation(annotation)?;
            added += 1;
        }
        Ok(added)
    }

    fn sensor_index(&self, channel: &str, modality: Modality) -> Result<usize> {
        let index = self
            .sensors
            .iter()
            .position(|s| s.channel == channel)
            .ok_or_else(|| format!("sensor {channel:?} was not declared"))?;
        if self.sensors[index].modality != modality {
            return Err(format!(
                "sensor {channel:?} is a {}, not a {}",
                self.sensors[index].modality.name(),
                modality.name()
            ));
        }
        Ok(index)
    }

    /// Record a sample_data of sensor `index` in the open frame: its record,
    /// linked after the channel's last one.
    fn push_sample_data(
        &mut self,
        index: usize,
        extension: &str,
        fileformat: &str,
        width: u32,
        height: u32,
    ) -> Result<PathBuf> {
        let frame = self.open_frame()?;
        let channel = self.sensors[index].channel.clone();
        if frame.channels.contains(&index) {
            return Err(format!("sensor {channel:?} already has data in this frame"));
        }
        let (sample, ego_pose) = (frame.sample, frame.ego_pose);
        let filename = format!("data/{channel}/{sample:05}.{extension}");
        let token = self.tokens.next_token();
        let prev = match self.sensors[index].last {
            Some(last) => {
                self.sample_data[last].next = token.clone();
                self.sample_data[last].token.clone()
            }
            None => String::new(),
        };
        self.sample_data.push(SampleData {
            token,
            sample_token: self.samples[sample].token.clone(),
            ego_pose_token: self.ego_poses[ego_pose].token.clone(),
            calibrated_sensor_token: self.sensors[index].calibrated_token.clone(),
            filename: filename.clone(),
            fileformat: fileformat.to_string(),
            width,
            height,
            timestamp: self.samples[sample].timestamp,
            is_key_frame: true,
            next: String::new(),
            prev,
            is_valid: true,
        });
        self.sensors[index].last = Some(self.sample_data.len() - 1);
        if let Some(frame) = self.frame.as_mut() {
            frame.channels.push(index);
        }
        Ok(self.root.join(filename))
    }

    /// Add a LiDAR sweep to the open frame: `points` is `[N, fields]`
    /// row-major, `x, y, z` (in the sensor's frame) and, when `fields > 3`,
    /// the intensity; any further fields are dropped.  The sweeps a channel
    /// gets in a frame are concatenated (`LIDAR_CONCAT`).  Written as
    /// `NNNNN.pcd.bin`, T4's default layout `(x, y, z, intensity, ring = -1)`,
    /// by `end_frame`.
    pub fn add_lidar(&mut self, channel: &str, points: &[f32], fields: usize) -> Result<()> {
        self.check_open()?;
        self.open_frame()?;
        let index = self.sensor_index(channel, Modality::Lidar)?;
        if fields < 3 {
            return Err(format!("a sweep needs x, y and z: got {fields} field(s)"));
        }
        if !points.len().is_multiple_of(fields) {
            return Err(format!(
                "a sweep of {} values does not split into rows of {fields}",
                points.len()
            ));
        }
        // The previous frame's size: one allocation, not one per sweep added.
        let reserve = self.sensors[index]
            .points_hint
            .max(points.len() / fields * 4);
        let frame = self.frame.as_mut().expect("checked open above");
        let buffer = match frame.lidar.iter().position(|(i, _)| *i == index) {
            Some(at) => &mut frame.lidar[at].1,
            None => {
                frame.lidar.push((index, self.workers.buffer(reserve)));
                &mut frame.lidar.last_mut().expect("pushed").1
            }
        };
        // A copy and nothing more on the caller's thread: the rows that are
        // not finite are dropped by the worker that writes them.
        if fields == 4 {
            buffer.extend_from_slice(points);
        } else {
            buffer.reserve(points.len() / fields * 4);
            for row in points.chunks_exact(fields) {
                let intensity = if fields > 3 { row[3] } else { 0.0 };
                buffer.extend_from_slice(&[row[0], row[1], row[2], intensity]);
            }
        }
        Ok(())
    }

    /// Add a camera frame already encoded as JPEG (`width` x `height`).
    pub fn add_camera_jpeg(
        &mut self,
        channel: &str,
        jpeg: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.check_open()?;
        self.open_frame()?;
        let index = self.sensor_index(channel, Modality::Camera)?;
        if jpeg.len() < 2 || jpeg[..2] != [0xff, 0xd8] {
            return Err(format!("camera {channel:?}'s frame is not a JPEG"));
        }
        self.check_resolution(index, width, height)?;
        let path = self.push_sample_data(index, "jpg", "jpg", width, height)?;
        self.workers.submit(Job::Bytes { path, bytes: jpeg })
    }

    fn check_resolution(&mut self, index: usize, width: u32, height: u32) -> Result<()> {
        if let Some(resolution) = self.sensors[index].resolution {
            if resolution != (width, height) {
                return Err(format!(
                    "camera {:?} changed resolution from {resolution:?} to {:?}",
                    self.sensors[index].channel,
                    (width, height)
                ));
            }
        }
        self.sensors[index].resolution = Some((width, height));
        Ok(())
    }

    /// Add a camera frame to the open frame: `pixels` is `height` rows of
    /// `width` pixels in `format`, written as `NNNNN.jpg`.
    pub fn add_camera(
        &mut self,
        channel: &str,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<()> {
        self.check_open()?;
        self.open_frame()?;
        let index = self.sensor_index(channel, Modality::Camera)?;
        if width == 0 || height == 0 || width > u32::from(u16::MAX) || height > u32::from(u16::MAX)
        {
            return Err(format!(
                "a camera frame of {width}x{height} cannot be a JPEG"
            ));
        }
        let expected = width as usize * height as usize * format.channels();
        if pixels.len() != expected {
            return Err(format!(
                "a {width}x{height} frame of {} channels is {expected} bytes, got {}",
                format.channels(),
                pixels.len()
            ));
        }
        self.check_resolution(index, width, height)?;
        let path = self.push_sample_data(index, "jpg", "jpg", width, height)?;
        self.workers.submit(Job::Camera {
            path,
            pixels,
            width: width as u16,
            height: height as u16,
            format,
            quality: self.jpeg_quality,
        })
    }

    /// Record the ego's controls at the open frame (`vehicle_state`).
    pub fn add_vehicle_state(&mut self, controls: VehicleControls) -> Result<()> {
        self.check_open()?;
        let timestamp = self.samples[self.open_frame()?.sample].timestamp;
        if self
            .vehicle_states
            .last()
            .is_some_and(|s| s.timestamp == timestamp)
        {
            return Err("the frame already has a vehicle state".to_string());
        }
        if let Some(shift) = &controls.shift_state {
            if !SHIFT_STATES.contains(&shift.as_str()) {
                return Err(format!(
                    "unknown shift state {shift:?} (expected one of {SHIFT_STATES:?})"
                ));
            }
        }
        let token = self.tokens.next_token();
        self.vehicle_states.push(VehicleState {
            token,
            timestamp,
            accel_pedal: controls.accel_pedal,
            brake_pedal: controls.brake_pedal,
            steer_pedal: controls.steer_pedal,
            steering_tire_angle: controls.steering_tire_angle,
            steering_wheel_angle: controls.steering_wheel_angle,
            shift_state: controls.shift_state,
            additional_info: controls
                .speed
                .map(|speed| AdditionalInfo { speed: Some(speed) }),
        });
        Ok(())
    }

    /// Close the open frame: hand each LiDAR channel's points to the workers,
    /// which write them and count the points inside each of the frame's boxes.
    pub fn end_frame(&mut self) -> Result<()> {
        self.check_open()?;
        let lidar = match self.frame.as_mut() {
            Some(frame) => std::mem::take(&mut frame.lidar),
            None => return Err("no frame is open (call begin_frame first)".to_string()),
        };
        let mut jobs = Vec::with_capacity(lidar.len());
        for (index, points) in lidar {
            self.sensors[index].points_hint = points.len();
            let path = self.push_sample_data(index, "pcd.bin", "pcd.bin", 0, 0)?;
            let frame = self.open_frame()?;
            // The boxes in the channel's frame, rather than every point in
            // the map's: a handful of transforms instead of a sweep's worth.
            let map_to_sensor = frame.ego.compose(&self.sensors[index].pose).inverse();
            let boxes = (frame.first_annotation..self.annotations.len())
                .map(|i| {
                    let a = &self.annotations[i];
                    let pose = map_to_sensor.compose(&Pose::new(a.translation, a.rotation));
                    (i, OrientedBox::new(pose, a.size))
                })
                .collect();
            jobs.push(Job::Lidar {
                path,
                points,
                boxes,
            });
        }
        self.frame = None;
        for job in jobs {
            self.workers.submit(job)?;
        }
        Ok(())
    }

    /// Close any open frame, wait for the sensor files, and write the tables
    /// and `status.json`.
    pub fn finish(&mut self) -> Result<Summary> {
        self.check_open()?;
        if self.frame.is_some() {
            self.end_frame()?;
        }
        self.finished = true;
        let Outcome { errors, counts } = self.workers.join();
        for (index, count) in counts {
            self.annotations[index].num_lidar_pts += count;
        }
        if !errors.is_empty() {
            return Err(format!(
                "{} sensor file(s) could not be written; the first: {}",
                errors.len(),
                errors[0]
            ));
        }
        if self.samples.is_empty() {
            return Err("no frame was recorded: a T4 dataset needs at least one".to_string());
        }
        if self.sample_data.is_empty() {
            return Err(
                "no sensor data was recorded: a T4 dataset needs at least one sample_data"
                    .to_string(),
            );
        }

        let annotation_dir = self.root.join("annotation");
        let write = |name: &str, records: &dyn erased::Records| -> Result<()> {
            records.write(&annotation_dir.join(format!("{name}.json")))
        };

        let scene = Scene {
            token: self.scene_token.clone(),
            name: self.info.name.clone(),
            description: self.info.description.clone(),
            log_token: self.log_token.clone(),
            nbr_samples: self.samples.len(),
            first_sample_token: self.samples[0].token.clone(),
            last_sample_token: self.samples[self.samples.len() - 1].token.clone(),
        };
        let log = Log {
            token: self.log_token.clone(),
            logfile: self.info.logfile.clone(),
            vehicle: self.info.vehicle.clone(),
            data_captured: self.info.date_captured.clone(),
            location: self.info.location.clone(),
        };
        let map = Map {
            token: self.tokens.next_token(),
            log_tokens: vec![self.log_token.clone()],
            category: "semantic_prior".to_string(),
            filename: self.map_filename.clone(),
        };
        let visibility = Visibility {
            token: self.visibility_token.clone(),
            level: "unavailable".to_string(),
            description: "Not measured: the boxes are the simulator's ground truth".to_string(),
        };
        let instances: Vec<Instance> = self
            .instance_order
            .iter()
            .map(|id| {
                let entry = &self.instances[id];
                let token_of = |i: Option<usize>| {
                    i.map(|i| self.annotations[i].token.clone())
                        .unwrap_or_default()
                };
                Instance {
                    token: entry.token.clone(),
                    category_token: entry.category_token.clone(),
                    instance_name: entry.name.clone(),
                    nbr_annotations: entry.count,
                    first_annotation_token: token_of(entry.first),
                    last_annotation_token: token_of(entry.last),
                }
            })
            .collect();

        write("attribute", &self.attributes)?;
        write("calibrated_sensor", &self.calibrated_sensors)?;
        write("category", &self.categories)?;
        write("ego_pose", &self.ego_poses)?;
        write("instance", &instances)?;
        write("log", &vec![log])?;
        write("map", &vec![map])?;
        write("sample", &self.samples)?;
        write("sample_annotation", &self.annotations)?;
        write("sample_data", &self.sample_data)?;
        write("scene", &vec![scene])?;
        write("sensor", &self.sensor_records)?;
        write("visibility", &vec![visibility])?;
        if !self.vehicle_states.is_empty() {
            write("vehicle_state", &self.vehicle_states)?;
        }

        let summary = Summary {
            data_root: self.root.display().to_string(),
            samples: self.samples.len(),
            sample_data: self.sample_data.len(),
            sample_annotations: self.annotations.len(),
            instances: instances.len(),
            start_timestamp: self.samples[0].timestamp,
            end_timestamp: self.samples[self.samples.len() - 1].timestamp,
        };
        self.write_status(&summary)?;
        Ok(summary)
    }

    fn write_status(&self, summary: &Summary) -> Result<()> {
        let sensors: serde_json::Map<String, serde_json::Value> = self
            .sensors
            .iter()
            .map(|s| {
                let mut value = serde_json::json!({ "modality": s.modality.name() });
                if let Some((w, h)) = s.resolution {
                    value["resolution"] = serde_json::json!([w, h]);
                }
                (s.channel.clone(), value)
            })
            .collect();
        let status = serde_json::json!({
            "dataset_name": self.info.name,
            "description": self.info.description,
            "purpose": "simulation",
            "status": "ready",
            "statistics": {
                "scenes": 1,
                "samples": summary.samples,
                "sample_data": summary.sample_data,
                "annotations_3d": summary.sample_annotations,
                "instances": summary.instances,
                "categories": self.categories.len(),
                "sensors": self.sensors.len(),
                "maps": 1,
            },
            "sensors": sensors,
            "timespan": {
                "start_timestamp": summary.start_timestamp,
                "end_timestamp": summary.end_timestamp,
                "duration_seconds":
                    (summary.end_timestamp - summary.start_timestamp) as f64 / 1e6,
            },
            "location": { "name": self.info.location },
        });
        write_json(&self.root.join("status.json"), &status)
    }
}

/// Tables of different record types, written through one closure.
mod erased {
    use super::*;

    pub trait Records {
        fn write(&self, path: &Path) -> Result<()>;
    }

    impl<T: Serialize> Records for Vec<T> {
        fn write(&self, path: &Path) -> Result<()> {
            write_json(path, self)
        }
    }
}
