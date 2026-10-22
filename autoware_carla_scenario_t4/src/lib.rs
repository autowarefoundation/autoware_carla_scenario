//! T4 datasets of autoware_carla_scenario runs: the writer (`writer`), and its
//! Python binding, `autoware_carla_scenario_t4._core.T4Writer`.

pub mod carla;
pub mod geometry;
pub mod tables;
pub mod writer;

use std::path::PathBuf;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::geometry::{Pose, Quat, Vec3};
use crate::writer::{
    Annotation, CarlaFrame, EgoState, Modality, PixelFormat, SceneInfo, VehicleControls,
};

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// Writes one scenario run as a T4 dataset in ``data_root``.
///
/// Every pose is in a right-handed frame: the ego's in the dataset's map
/// frame, a sensor's in the ego's (``base_link``), a box's in the map frame.
/// Rotations are quaternions ``(w, x, y, z)``.  Declare the sensors, then for
/// each frame call :meth:`begin_frame`, add its boxes and sensor data, and
/// :meth:`end_frame`; :meth:`finish` writes the tables.
#[pyclass(module = "autoware_carla_scenario_t4._core")]
struct T4Writer {
    inner: writer::T4Writer,
}

#[pymethods]
impl T4Writer {
    #[new]
    #[pyo3(signature = (
        data_root,
        *,
        scene_name,
        scene_description = String::new(),
        logfile = String::new(),
        vehicle = String::new(),
        date_captured = String::new(),
        location = String::new(),
        token_seed = None,
        map_offset = [0.0, 0.0, 0.0],
        rear_axle_offset_m = 0.0,
        max_distance_m = None,
        jpeg_quality = 90,
        workers = 2,
        queue = 16,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_root: PathBuf,
        scene_name: String,
        scene_description: String,
        logfile: String,
        vehicle: String,
        date_captured: String,
        location: String,
        token_seed: Option<String>,
        map_offset: Vec3,
        rear_axle_offset_m: f64,
        max_distance_m: Option<f64>,
        jpeg_quality: u8,
        workers: usize,
        queue: usize,
    ) -> PyResult<Self> {
        let info = SceneInfo {
            token_seed: token_seed.unwrap_or_else(|| scene_name.clone()),
            name: scene_name,
            description: scene_description,
            logfile,
            vehicle,
            date_captured,
            location,
        };
        let carla = CarlaFrame {
            map_offset,
            rear_axle_offset_m,
            max_distance_m,
        };
        writer::T4Writer::create_in(data_root, info, carla, jpeg_quality, workers, queue)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    /// The directory the dataset is written into.
    #[getter]
    fn data_root(&self) -> PathBuf {
        self.inner.data_root().to_path_buf()
    }

    /// How many frames have been begun.
    #[getter]
    fn frames(&self) -> usize {
        self.inner.frames()
    }

    /// Copy the Lanelet2 map into ``map/lanelet2_map.osm``.
    fn add_lanelet2_map(&mut self, path: PathBuf) -> PyResult<()> {
        self.inner.add_lanelet2_map(&path).map_err(value_error)
    }

    /// Declare a sensor on the ego (before the first frame).
    #[pyo3(signature = (
        channel, modality, translation, rotation, *,
        camera_intrinsic = None, camera_distortion = None,
    ))]
    fn add_sensor(
        &mut self,
        channel: &str,
        modality: &str,
        translation: Vec3,
        rotation: Quat,
        camera_intrinsic: Option<[[f64; 3]; 3]>,
        camera_distortion: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let modality = Modality::parse(modality).map_err(value_error)?;
        self.inner
            .add_sensor(
                channel,
                modality,
                Pose::new(translation, rotation),
                camera_intrinsic,
                camera_distortion,
            )
            .map_err(value_error)
    }

    /// Open a frame at ``timestamp_us`` with the ego at ``translation`` and
    /// ``rotation``; ``twist`` and ``acceleration`` are in the ego's frame.
    #[pyo3(signature = (timestamp_us, translation, rotation, *, twist = None, acceleration = None))]
    fn begin_frame(
        &mut self,
        timestamp_us: u64,
        translation: Vec3,
        rotation: Quat,
        twist: Option<[f64; 6]>,
        acceleration: Option<Vec3>,
    ) -> PyResult<()> {
        self.inner
            .begin_frame(
                timestamp_us,
                EgoState {
                    pose: Pose::new(translation, rotation),
                    twist,
                    acceleration,
                },
            )
            .map_err(value_error)
    }

    /// Add a road user's box (``size`` is ``(width, length, height)``).
    #[pyo3(signature = (
        instance_id, category, translation, size, rotation, *,
        velocity = None, acceleration = None, attributes = Vec::new(), instance_name = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_annotation(
        &mut self,
        instance_id: u64,
        category: String,
        translation: Vec3,
        size: Vec3,
        rotation: Quat,
        velocity: Option<Vec3>,
        acceleration: Option<Vec3>,
        attributes: Vec<String>,
        instance_name: Option<String>,
    ) -> PyResult<()> {
        self.inner
            .add_annotation(Annotation {
                instance_id,
                category,
                instance_name,
                attributes,
                translation,
                size,
                rotation,
                velocity,
                acceleration,
            })
            .map_err(value_error)
    }

    /// Register a road user :meth:`add_carla_boxes` annotates: its box's
    /// centre in its own (right-handed) frame and its size ``(width, length,
    /// height)``.
    #[pyo3(signature = (actor_id, category, center, size, name = None))]
    fn register_actor(
        &mut self,
        actor_id: u64,
        category: &str,
        center: Vec3,
        size: Vec3,
        name: Option<String>,
    ) -> PyResult<()> {
        self.inner
            .register_actor(actor_id, category, center, size, name)
            .map_err(value_error)
    }

    /// Whether ``actor_id`` is registered.
    fn knows_actor(&self, actor_id: u64) -> bool {
        self.inner.knows_actor(actor_id)
    }

    /// Open a frame with the ego actor's state as CARLA reports it
    /// (``transform`` ``(x, y, z, roll, pitch, yaw)``, velocity, angular
    /// velocity in deg/s, acceleration; CARLA's world frame).
    fn begin_frame_carla(
        &mut self,
        timestamp_us: u64,
        transform: [f64; 6],
        velocity: [f64; 3],
        angular_velocity: [f64; 3],
        acceleration: [f64; 3],
    ) -> PyResult<()> {
        self.inner
            .begin_frame_carla(
                timestamp_us,
                transform,
                velocity,
                angular_velocity,
                acceleration,
            )
            .map_err(value_error)
    }

    /// Annotate the registered road users ``ids`` (``uint64 [N]``) from their
    /// CARLA states (``float64 [N, 9]``: ``x, y, z, roll, pitch, yaw, vx, vy,
    /// vz``); how many were annotated.
    fn add_carla_boxes(
        &mut self,
        ids: PyReadonlyArray1<'_, u64>,
        states: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<usize> {
        let ids: Vec<u64> = ids.as_array().iter().copied().collect();
        let states: Vec<f64> = match states.as_slice() {
            Ok(values) => values.to_vec(),
            Err(_) => states.as_array().iter().copied().collect(),
        };
        self.inner
            .add_carla_boxes(&ids, &states)
            .map_err(value_error)
    }

    /// Add a camera frame already encoded as JPEG.
    fn add_camera_jpeg(
        &mut self,
        channel: &str,
        jpeg: &[u8],
        width: u32,
        height: u32,
    ) -> PyResult<()> {
        self.inner
            .add_camera_jpeg(channel, jpeg.to_vec(), width, height)
            .map_err(value_error)
    }

    /// Add a LiDAR sweep: ``points`` is ``float32 [N, >=3]`` in the sensor's
    /// frame, ``x, y, z[, intensity, ...]``.
    fn add_lidar(
        &mut self,
        py: Python<'_>,
        channel: &str,
        points: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        let fields = points.shape()[1];
        let owned;
        let values: &[f32] = match points.as_slice() {
            Ok(values) => values,
            Err(_) => {
                owned = points.as_array().iter().copied().collect::<Vec<f32>>();
                &owned
            }
        };
        let inner = &mut self.inner;
        py.detach(|| inner.add_lidar(channel, values, fields))
            .map_err(value_error)
    }

    /// Add a camera frame: ``image`` is ``uint8 [H, W, C]`` in ``pixel_format``
    /// (``"bgra"``, CARLA's, by default).
    #[pyo3(signature = (channel, image, pixel_format = "bgra"))]
    fn add_camera(
        &mut self,
        py: Python<'_>,
        channel: &str,
        image: PyReadonlyArray3<'_, u8>,
        pixel_format: &str,
    ) -> PyResult<()> {
        let format = PixelFormat::parse(pixel_format).map_err(value_error)?;
        let shape = image.shape();
        let (height, width, channels) = (shape[0], shape[1], shape[2]);
        if channels != format.channels() {
            return Err(value_error(format!(
                "a {pixel_format} image has {} channels, got {channels}",
                format.channels()
            )));
        }
        let pixels: Vec<u8> = match image.as_slice() {
            Ok(values) => values.to_vec(),
            Err(_) => image.as_array().iter().copied().collect(),
        };
        let inner = &mut self.inner;
        py.detach(|| inner.add_camera(channel, pixels, width as u32, height as u32, format))
            .map_err(value_error)
    }

    /// Record the ego's controls at the open frame (pedals and steering in %).
    #[pyo3(signature = (
        *, accel_pedal = None, brake_pedal = None, steer_pedal = None,
        steering_tire_angle = None, steering_wheel_angle = None, shift_state = None, speed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_vehicle_state(
        &mut self,
        accel_pedal: Option<f64>,
        brake_pedal: Option<f64>,
        steer_pedal: Option<f64>,
        steering_tire_angle: Option<f64>,
        steering_wheel_angle: Option<f64>,
        shift_state: Option<String>,
        speed: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .add_vehicle_state(VehicleControls {
                accel_pedal,
                brake_pedal,
                steer_pedal,
                steering_tire_angle,
                steering_wheel_angle,
                shift_state,
                speed,
            })
            .map_err(value_error)
    }

    /// Close the open frame, counting each box's LiDAR points.
    fn end_frame(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.end_frame()).map_err(value_error)
    }

    /// Wait for the sensor files and write the tables; what was written.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &mut self.inner;
        let summary = py
            .detach(|| inner.finish())
            .map_err(PyRuntimeError::new_err)?;
        let dict = PyDict::new(py);
        dict.set_item("data_root", summary.data_root)?;
        dict.set_item("samples", summary.samples)?;
        dict.set_item("sample_data", summary.sample_data)?;
        dict.set_item("sample_annotations", summary.sample_annotations)?;
        dict.set_item("instances", summary.instances)?;
        dict.set_item("start_timestamp", summary.start_timestamp)?;
        dict.set_item("end_timestamp", summary.end_timestamp)?;
        Ok(dict)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<T4Writer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
