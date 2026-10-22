//! A scene written end to end, read back as t4-devkit reads it: every
//! reference resolves, and the chains run frame to frame.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use autoware_carla_scenario_t4::geometry::{yaw_quaternion, Pose};
use autoware_carla_scenario_t4::writer::{
    Annotation, CarlaFrame, EgoState, Modality, PixelFormat, SceneInfo, T4Writer, VehicleControls,
};
use serde_json::Value;

fn table(root: &Path, name: &str) -> Vec<Value> {
    let text = fs::read_to_string(root.join("annotation").join(format!("{name}.json"))).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn by_token(records: &[Value]) -> HashMap<String, Value> {
    records
        .iter()
        .map(|r| (r["token"].as_str().unwrap().to_string(), r.clone()))
        .collect()
}

fn info() -> SceneInfo {
    SceneInfo {
        name: "CutInScenario".into(),
        token_seed: "CutInScenario".into(),
        ..SceneInfo::default()
    }
}

fn car(id: u64, x: f64) -> Annotation {
    Annotation {
        instance_id: id,
        category: "car".into(),
        instance_name: None,
        attributes: vec![],
        translation: [x, 0.0, 0.75],
        size: [2.0, 4.0, 1.5],
        rotation: yaw_quaternion(0.0),
        velocity: Some([1.0, 0.0, 0.0]),
        acceleration: None,
    }
}

fn write_scene(root: &Path) {
    let mut w = T4Writer::create(root, info(), 90, 2, 4).unwrap();
    w.add_sensor(
        "LIDAR_TOP",
        Modality::Lidar,
        Pose::new([0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0]),
        None,
        None,
    )
    .unwrap();
    w.add_sensor(
        "CAM_FRONT",
        Modality::Camera,
        Pose::new([1.5, 0.0, 1.5], [0.5, -0.5, 0.5, -0.5]),
        Some([[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]),
        None,
    )
    .unwrap();
    for i in 0..3u64 {
        let ego = EgoState {
            pose: Pose::new([i as f64, 0.0, 0.0], yaw_quaternion(0.0)),
            twist: Some([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            acceleration: None,
        };
        w.begin_frame(1_700_000_000_000_000 + i * 100_000, ego)
            .unwrap();
        w.add_annotation(car(7, 10.0 + i as f64)).unwrap();
        if i == 1 {
            w.add_annotation(car(9, -10.0)).unwrap();
        }
        // Two points inside car 7 (10 m ahead of the ego, LiDAR 2 m up),
        // one outside every box.
        let points: Vec<f32> = vec![
            10.0, 0.0, -1.0, 0.5, 10.5, 0.5, -1.5, 0.5, 0.0, 30.0, 0.0, 0.1,
        ];
        w.add_lidar("LIDAR_TOP", &points, 4).unwrap();
        w.add_camera("CAM_FRONT", vec![128; 8 * 6 * 4], 8, 6, PixelFormat::Bgra)
            .unwrap();
        w.add_vehicle_state(VehicleControls {
            accel_pedal: Some(30.0),
            shift_state: Some("FORWARD".into()),
            speed: Some(1.0),
            ..VehicleControls::default()
        })
        .unwrap();
        w.end_frame().unwrap();
    }
    let summary = w.finish().unwrap();
    assert_eq!(summary.samples, 3);
    assert_eq!(summary.sample_data, 6);
    assert_eq!(summary.sample_annotations, 4);
    assert_eq!(summary.instances, 2);
}

#[test]
fn every_reference_resolves() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_scene(root);

    let samples = by_token(&table(root, "sample"));
    let sample_data = by_token(&table(root, "sample_data"));
    let annotations = by_token(&table(root, "sample_annotation"));
    let instances = by_token(&table(root, "instance"));
    let ego_poses = by_token(&table(root, "ego_pose"));
    let calibrated = by_token(&table(root, "calibrated_sensor"));
    let sensors = by_token(&table(root, "sensor"));
    let categories = by_token(&table(root, "category"));
    let visibility = by_token(&table(root, "visibility"));
    let logs = by_token(&table(root, "log"));
    let scenes = table(root, "scene");
    let maps = table(root, "map");

    assert_eq!(scenes.len(), 1);
    assert!(logs.contains_key(scenes[0]["log_token"].as_str().unwrap()));
    assert!(samples.contains_key(scenes[0]["first_sample_token"].as_str().unwrap()));
    assert!(samples.contains_key(scenes[0]["last_sample_token"].as_str().unwrap()));
    assert_eq!(maps[0]["log_tokens"][0], scenes[0]["log_token"]);

    for sd in sample_data.values() {
        assert!(samples.contains_key(sd["sample_token"].as_str().unwrap()));
        assert!(ego_poses.contains_key(sd["ego_pose_token"].as_str().unwrap()));
        let cs = &calibrated[sd["calibrated_sensor_token"].as_str().unwrap()];
        assert!(sensors.contains_key(cs["sensor_token"].as_str().unwrap()));
        assert!(root.join(sd["filename"].as_str().unwrap()).is_file());
        for link in ["next", "prev"] {
            let t = sd[link].as_str().unwrap();
            assert!(t.is_empty() || sample_data.contains_key(t));
        }
    }
    for ann in annotations.values() {
        assert!(samples.contains_key(ann["sample_token"].as_str().unwrap()));
        assert!(visibility.contains_key(ann["visibility_token"].as_str().unwrap()));
        let inst = &instances[ann["instance_token"].as_str().unwrap()];
        assert!(categories.contains_key(inst["category_token"].as_str().unwrap()));
        for link in ["next", "prev"] {
            let t = ann[link].as_str().unwrap();
            assert!(t.is_empty() || annotations.contains_key(t));
        }
    }
    for inst in instances.values() {
        assert!(annotations.contains_key(inst["first_annotation_token"].as_str().unwrap()));
        assert!(annotations.contains_key(inst["last_annotation_token"].as_str().unwrap()));
    }
    assert!(root.join("status.json").is_file());
    assert_eq!(table(root, "vehicle_state").len(), 3);
}

#[test]
fn boxes_count_the_points_inside_them() {
    let dir = tempfile::tempdir().unwrap();
    write_scene(dir.path());
    let anns = table(dir.path(), "sample_annotation");
    // Car 7 moves with the ego, so both points stay inside it every frame.
    let car7: Vec<u64> = anns
        .iter()
        .filter(|a| {
            a["translation"][1].as_f64() == Some(0.0) && a["translation"][0].as_f64().unwrap() > 0.0
        })
        .map(|a| a["num_lidar_pts"].as_u64().unwrap())
        .collect();
    assert_eq!(car7, vec![2, 2, 2]);
    let behind: Vec<u64> = anns
        .iter()
        .filter(|a| a["translation"][0].as_f64() == Some(-10.0))
        .map(|a| a["num_lidar_pts"].as_u64().unwrap())
        .collect();
    assert_eq!(behind, vec![0]);
}

#[test]
fn sweeps_are_written_in_t4s_five_float_layout() {
    let dir = tempfile::tempdir().unwrap();
    write_scene(dir.path());
    let bytes = fs::read(dir.path().join("data/LIDAR_TOP/00000.pcd.bin")).unwrap();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(floats.len(), 3 * 5);
    assert_eq!(&floats[..5], &[10.0, 0.0, -1.0, 0.5, -1.0]);
    let jpeg = fs::read(dir.path().join("data/CAM_FRONT/00000.jpg")).unwrap();
    assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
}

#[test]
fn the_same_seed_writes_the_same_tables() {
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    write_scene(a.path());
    write_scene(b.path());
    for name in ["sample", "sample_data", "sample_annotation", "instance"] {
        assert_eq!(table(a.path(), name), table(b.path(), name), "{name}");
    }
}

#[test]
fn misuse_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = T4Writer::create(dir.path(), info(), 90, 1, 1).unwrap();
    let ego = EgoState {
        pose: Pose::new([0.0; 3], [1.0, 0.0, 0.0, 0.0]),
        twist: None,
        acceleration: None,
    };
    assert!(w.begin_frame(1, ego).is_err(), "no sensor declared");
    w.add_sensor("LIDAR_TOP", Modality::Lidar, ego.pose, None, None)
        .unwrap();
    assert!(
        w.add_sensor("CAM", Modality::Camera, ego.pose, None, None)
            .is_err(),
        "a camera needs its intrinsics"
    );
    w.begin_frame(10, ego).unwrap();
    assert!(w
        .add_camera("LIDAR_TOP", vec![0; 4], 1, 1, PixelFormat::Bgra)
        .is_err());
    w.add_annotation(car(1, 0.0)).unwrap();
    assert!(w.add_annotation(car(1, 0.0)).is_err(), "twice in a frame");
    w.add_lidar("LIDAR_TOP", &[0.0, 0.0, 0.0], 3).unwrap();
    assert!(
        w.add_lidar("LIDAR_TOP", &[0.0, 0.0], 3).is_err(),
        "not rows of 3"
    );
    w.end_frame().unwrap();
    assert!(w.begin_frame(10, ego).is_err(), "timestamps must increase");
    w.finish().unwrap();
    assert!(w.begin_frame(20, ego).is_err(), "finished");
    assert!(
        T4Writer::create(dir.path(), info(), 90, 1, 1).is_err(),
        "already a dataset"
    );
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn assert_close(a: &[f64], b: &[f64]) {
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-9, "{a:?} != {b:?}");
    }
}

#[test]
fn sweeps_of_one_channel_are_concatenated() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = T4Writer::create(dir.path(), info(), 90, 1, 1).unwrap();
    let identity = Pose::new([0.0; 3], [1.0, 0.0, 0.0, 0.0]);
    w.add_sensor("LIDAR_CONCAT", Modality::Lidar, identity, None, None)
        .unwrap();
    let ego = EgoState {
        pose: identity,
        twist: None,
        acceleration: None,
    };
    w.begin_frame(1, ego).unwrap();
    w.add_lidar("LIDAR_CONCAT", &[1.0, 2.0, 3.0, 0.5], 4)
        .unwrap();
    w.add_lidar("LIDAR_CONCAT", &[4.0, 5.0, 6.0, f32::NAN, 7.0, 8.0], 3)
        .unwrap();
    w.end_frame().unwrap();
    w.finish().unwrap();
    let bytes = fs::read(dir.path().join("data/LIDAR_CONCAT/00000.pcd.bin")).unwrap();
    assert_eq!(bytes.len(), 2 * 5 * 4, "the row with a NaN is dropped");
    assert_eq!(table(dir.path(), "sample_data").len(), 1);
}

#[test]
fn carla_states_become_map_frame_poses_and_boxes() {
    let dir = tempfile::tempdir().unwrap();
    let carla = CarlaFrame {
        map_offset: [100.0, 200.0, 10.0],
        rear_axle_offset_m: -1.5,
        max_distance_m: Some(50.0),
    };
    let mut w = T4Writer::create_in(dir.path(), info(), carla, 90, 1, 1).unwrap();
    let identity = Pose::new([0.0; 3], [1.0, 0.0, 0.0, 0.0]);
    w.add_sensor("LIDAR_CONCAT", Modality::Lidar, identity, None, None)
        .unwrap();
    w.register_actor(2, "car", [0.0, 0.0, 0.7], [1.8, 4.0, 1.4], None)
        .unwrap();
    w.register_actor(3, "pedestrian", [0.0, 0.0, 0.9], [0.6, 0.6, 1.8], None)
        .unwrap();
    // The ego at CARLA (10, 5) facing south (yaw 90), driving at 2 m/s.
    w.begin_frame_carla(
        1_000_000,
        [10.0, 5.0, 0.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap();
    // The car 20 m ahead of it (further south); the pedestrian 100 m away.
    let states = [
        10.0, 25.0, 0.0, 0.0, 0.0, 90.0, 0.0, 2.0, 0.0, //
        110.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert!(w.add_carla_boxes(&[2, 3, 4], &states).is_err());
    assert_eq!(w.add_carla_boxes(&[2, 3], &states).unwrap(), 1);
    // A point 21.5 m ahead of the rear axle, on the car (base_link frame).
    w.add_lidar("LIDAR_CONCAT", &[21.5, 0.0, 0.7, 1.0], 4)
        .unwrap();
    w.end_frame().unwrap();
    w.finish().unwrap();

    let ego = &table(dir.path(), "ego_pose")[0];
    // The rear axle 1.5 m behind the actor: north of it, at CARLA y = 3.5.
    assert_close(&floats(&ego["translation"]), &[110.0, 196.5, 10.0]);
    assert_close(&floats(&ego["twist"])[..2], &[2.0, 0.0]);

    let anns = table(dir.path(), "sample_annotation");
    assert_eq!(anns.len(), 1);
    assert_close(&floats(&anns[0]["translation"]), &[110.0, 175.0, 10.7]);
    assert_eq!(anns[0]["num_lidar_pts"], 1);
    assert_close(&floats(&anns[0]["velocity"]), &[0.0, -2.0, 0.0]);
}
