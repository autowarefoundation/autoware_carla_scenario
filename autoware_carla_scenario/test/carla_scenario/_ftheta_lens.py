"""Cosmos3's front wide camera (cosmos3_canvas.camera, resolution 480): an
F-theta lens as a carla_canvas renderer draws a camera through it."""

from typing import Any

COSMOS3_FRONT_WIDE: dict[str, Any] = {
    "model": "ftheta",
    "principal_point": [412.641208, 242.790324],
    "pixeldist_to_angle_poly": [
        0.0,
        0.00243165382,
        -4.05625414e-08,
        6.10818843e-10,
        -9.93181484e-13,
        8.73787714e-16,
    ],
    "angle_to_pixeldist_poly": [
        0.0,
        411.286296,
        2.40299018,
        -16.2530848,
        10.3586782,
        -3.39986006,
    ],
    "max_angle": 1.226783919,
    "linear_cde": [1.00234926, 0.0, 0.0],
}
