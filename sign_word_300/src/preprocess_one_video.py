#!/usr/bin/env python3
"""Convert one source MP4 directly to the project's SPOTER2-style HDF5 sample.

This intentionally skips the intermediate MediaPipe raw ``.json.gz`` file while
producing the same datasets and attributes as ``preprocess_mediapipe_sample.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

import cv2
import h5py
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "holistic_landmarker.task"
FACE_INDICES = [
    0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
    93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
    282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405,
    468, 473,
]
PARTS = ("pose", "right_hand", "left_hand", "face")
NAME_RE = re.compile(
    r"^(?P<video_id>NIA_SL_WORD(?P<word>\d{4})_REAL(?P<actor>\d{2})_(?P<camera>[DFLRU]))$",
    re.IGNORECASE,
)


def xy(items: list[dict], indices: list[int] | range) -> np.ndarray:
    return np.asarray([[items[i]["x"], items[i]["y"]] for i in indices], dtype=np.float32)


def normalize_pose(points: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    center = (points[11] + points[12]) / 2.0
    shoulder_distance = float(np.linalg.norm(points[11] - points[12]))
    if not np.isfinite(shoulder_distance) or shoulder_distance <= eps:
        return None
    return (points - center) / (1.5 * shoulder_distance)


def normalize_local(
    points: np.ndarray, padding: float = 0.2, eps: float = 1e-6
) -> np.ndarray | None:
    low, high = points.min(axis=0), points.max(axis=0)
    side = float(np.max(high - low))
    if not np.isfinite(side) or side <= eps:
        return None
    center = (low + high) / 2.0
    return (points - center) / (side * (0.5 + padding))


def process(raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = raw["frames"]
    features = np.zeros((len(frames), 208), dtype=np.float32)
    mask = np.zeros((len(frames), 4), dtype=np.uint8)
    frame_indices = np.zeros(len(frames), dtype=np.int32)
    timestamps = np.zeros(len(frames), dtype=np.float64)
    specs = (
        ("pose", range(25), 50, normalize_pose),
        ("right_hand", range(21), 42, normalize_local),
        ("left_hand", range(21), 42, normalize_local),
        ("face", FACE_INDICES, 74, normalize_local),
    )
    for row, frame in enumerate(frames):
        frame_indices[row] = frame["frame_index"]
        timestamps[row] = frame["timestamp_ms"]
        offset = 0
        for part_index, (name, indices, width, normalizer) in enumerate(specs):
            item = frame[name]
            if item.get("detected") and item.get("landmarks"):
                points = normalizer(xy(item["landmarks"], indices))
                if points is not None and np.all(np.isfinite(points)):
                    features[row, offset:offset + width] = points.reshape(-1)
                    mask[row, part_index] = 1
            offset += width
    return features, mask, frame_indices, timestamps


def finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def landmark(item: object) -> dict[str, float | None]:
    return {
        name: finite(getattr(item, name, None))
        for name in ("x", "y", "z", "visibility", "presence")
    }


def landmarks(items: object) -> list[dict[str, float | None]] | None:
    return [landmark(item) for item in items] if items else None


def video_identity(path: Path) -> dict[str, object]:
    match = NAME_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(
            "The MP4 filename must look like "
            "NIA_SL_WORD0001_REAL01_D.mp4 so its metadata can be inferred."
        )
    return {
        "video_id": match["video_id"],
        "word_id": int(match["word"]),
        "actor_id": match["actor"],
        "camera_id": match["camera"].upper(),
    }


def extract_raw(video: Path, model: Path, threshold: float) -> dict:
    """Decode every frame and return only the in-memory fields preprocessing uses."""
    identity = video_identity(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")

    fps = finite(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    options = vision.HolisticLandmarkerOptions(
        # A byte buffer avoids native Windows failures on Korean model paths.
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=model.read_bytes(),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
        min_face_detection_confidence=threshold,
        min_face_landmarks_confidence=threshold,
        min_face_suppression_threshold=threshold,
        min_pose_detection_confidence=threshold,
        min_pose_landmarks_confidence=threshold,
        min_pose_suppression_threshold=threshold,
        min_hand_landmarks_confidence=threshold,
    )

    frames: list[dict] = []
    last_api_timestamp = -1
    try:
        with vision.HolisticLandmarker.create_from_options(options) as detector:
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                decoder_ms = finite(capture.get(cv2.CAP_PROP_POS_MSEC))
                fallback_ms = index / fps * 1000.0 if fps > 0 else float(index)
                timestamp_ms = (
                    decoder_ms
                    if decoder_ms is not None and (decoder_ms > 0 or index == 0)
                    else fallback_ms
                )
                api_timestamp = max(last_api_timestamp + 1, int(round(timestamp_ms)))
                last_api_timestamp = api_timestamp
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), api_timestamp
                )
                frames.append(
                    {
                        "frame_index": index,
                        "timestamp_ms": timestamp_ms,
                        "pose": {
                            "detected": bool(result.pose_landmarks),
                            "landmarks": landmarks(result.pose_landmarks),
                        },
                        "right_hand": {
                            "detected": bool(result.right_hand_landmarks),
                            "landmarks": landmarks(result.right_hand_landmarks),
                        },
                        "left_hand": {
                            "detected": bool(result.left_hand_landmarks),
                            "landmarks": landmarks(result.left_hand_landmarks),
                        },
                        "face": {
                            "detected": bool(result.face_landmarks),
                            "landmarks": landmarks(result.face_landmarks),
                        },
                    }
                )
                index += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were decoded from: {video}")
    return {
        "source": {
            **identity,
            "fps": fps,
            "declared_frame_count": declared,
            "decoded_frame_count": len(frames),
        },
        "frames": frames,
    }


def write_h5(raw: dict, output: Path, overwrite: bool) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    features, mask, frame_indices, timestamps = process(raw)
    source = raw["source"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    try:
        with h5py.File(temporary_name, "w") as h5:
            group = h5.create_group(source["video_id"])
            group.create_dataset("features", data=features, compression="gzip", shuffle=True)
            group.create_dataset("part_mask", data=mask, compression="gzip", shuffle=True)
            group.create_dataset("frame_index", data=frame_indices)
            group.create_dataset("timestamp_ms", data=timestamps)
            group.attrs["feature_version"] = "spoter2_mp_xy_v1"
            group.attrs["feature_order"] = (
                "global_pose_0_24_xy,local_right_hand_0_20_xy,"
                "local_left_hand_0_20_xy,local_face_37_xy"
            )
            for key in (
                "word_id", "actor_id", "camera_id", "fps",
                "declared_frame_count", "decoded_frame_count",
            ):
                group.attrs[key] = source[key]
            group.attrs["original_frame_count"] = len(raw["frames"])
            group.attrs["processed_frame_count"] = len(features)
            for index, name in enumerate(PARTS):
                group.attrs[f"{name}_detection_rate"] = float(mask[:, index].mean())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return {
        "output": str(output),
        "video_id": source["video_id"],
        "features_shape": list(features.shape),
        "part_mask_shape": list(mask.shape),
        "detection_rates": {
            name: float(mask[:, index].mean()) for index, name in enumerate(PARTS)
        },
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one AI Hub source MP4 directly to a [T, 208] HDF5 sample."
    )
    parser.add_argument("input", type=Path, help="one NIA_SL_...mp4 source video")
    parser.add_argument("output", type=Path, nargs="?", help="output .h5 (default: beside input)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video = args.input.resolve()
    output = (args.output or video.with_suffix(".h5")).resolve()
    if not video.is_file():
        parser.error(f"input file does not exist: {video}")
    if not args.model.is_file():
        parser.error(f"MediaPipe model does not exist: {args.model}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    result = write_h5(extract_raw(video, args.model, args.threshold), output, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
