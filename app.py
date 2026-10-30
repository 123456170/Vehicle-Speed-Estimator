import os
import re
import json
import math
import time
import uuid
import shutil
import sqlite3
import tempfile
import hashlib
import base64
import traceback
import threading
import queue
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import pandas as pd
import streamlit as st


try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


st.set_page_config(
    page_title="Vehicle Speed Estimator",
    page_icon="🚗",
    layout="wide",
)


APP_NAME = "Vehicle Speed Estimator"
APP_VERSION = "5.1-smooth-no-duplicate-keys"
DISCLAIMER = "estimate - not for enforcement"

DEFAULT_W, DEFAULT_H = 960, 540
DEFAULT_FPS = 25.0
PREFILL_DEMO_FRAMES = 90
LIVE_FRAGMENT_HZ = 0.05


# ----------------------------
# Data models / settings
# ----------------------------

@dataclass
class Settings:
    mode: str = "demo"  # demo | file | webcam | rtsp
    scenario: str = "free_flow"
    source_path: str = ""
    webcam_index: int = 0
    rtsp_url: str = ""

    width: int = DEFAULT_W
    height: int = DEFAULT_H
    fps: float = DEFAULT_FPS
    demo_duration_sec: float = 30.0
    max_frames: int = 0
    process_every_n_frames: int = 1

    detector_backend: str = "auto"  # auto | motion | yolo
    yolo_model: str = "yolov8n.pt"
    device: str = "auto"
    half: bool = False

    confidence_threshold: float = 0.25
    nms_iou: float = 0.45

    track_high_thresh: float = 0.45
    track_low_thresh: float = 0.10
    match_thresh: float = 0.20
    track_buffer: int = 45
    min_hits: int = 3

    speed_window_sec: float = 0.8
    speed_min_samples: int = 3

    min_track_frames: int = 8
    overspeed_margin_kmh: float = 2.0
    confirm_frames: int = 5
    clear_frames: int = 10
    dedupe_cooldown_sec: float = 6.0

    left_lane_limit_kmh: float = 60.0
    right_lane_limit_kmh: float = 50.0

    road_width_m: float = 8.0
    road_length_m: float = 30.0
    zone_start_m: float = 2.0
    zone_end_m: float = 28.0

    calib_src: List[List[float]] = field(
        default_factory=lambda: [
            [150.0, 520.0],
            [810.0, 520.0],
            [600.0, 260.0],
            [360.0, 260.0],
        ]
    )

    blur_regions: bool = True
    blur_strength: int = 15
    save_evidence: bool = True
    clip_seconds: float = 3.0
    retention_days: int = 30
    hash_salt: str = ""

    show_overlay: bool = True
    show_ground_truth: bool = True
    night_glare: bool = False
    camera_shake: bool = False


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    conf: float
    cls: str
    true_speed_kmh: Optional[float] = None
    true_x: Optional[float] = None
    true_y: Optional[float] = None
    hint_id: Optional[str] = None


@dataclass
class SynthVehicle:
    vid: str
    cls: str
    lane: str
    y: float
    speed_kmh: float
    length: float
    width: float
    color: Tuple[int, int, int]
    motion: str = "constant"
    phase: float = 0.0
    amp: float = 0.0
    probe: bool = False
    current_speed_kmh: float = 0.0


@dataclass
class CalibrationModel:
    src: np.ndarray
    dst: np.ndarray
    H: np.ndarray
    H_inv: np.ndarray
    rmse: float
    errors: np.ndarray
    calibration_id: str


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_evidence_root() -> str:
    root = os.path.join(tempfile.gettempdir(), "vehicle_speed_estimator_evidence")
    ensure_dir(root)
    cleanup_old_runs(root, 30)
    return root


def cleanup_old_runs(root: str, days: int):
    cutoff = time.time() - max(1, int(days)) * 86400
    try:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                try:
                    if os.path.getmtime(path) < cutoff:
                        shutil.rmtree(path, ignore_errors=True)
                except OSError:
                    pass
    except OSError:
        pass


def to_rgb(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def sanitize_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def settings_from_session() -> Settings:
    base = Settings()
    vals: Dict[str, Any] = {}

    for f in fields(base):
        default = getattr(base, f.name)
        val = st.session_state.get(f.name, default)

        if isinstance(default, bool):
            val = bool(val)
        elif isinstance(default, int):
            val = int(float(val))
        elif isinstance(default, float):
            val = float(val)
        elif isinstance(default, list):
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    val = default
            try:
                parsed = [[float(p[0]), float(p[1])] for p in val]
                if len(parsed) == 4:
                    val = parsed
                else:
                    val = default
            except Exception:
                val = default
        elif isinstance(default, str):
            val = str(val)

        vals[f.name] = val

    if not vals.get("hash_salt"):
        vals["hash_salt"] = uuid.uuid4().hex[:8]
        st.session_state["hash_salt"] = vals["hash_salt"]

    s = Settings(**vals)

    # Safety clamps
    s.width = max(320, int(s.width))
    s.height = max(180, int(s.height))
    s.fps = max(1.0, float(s.fps))
    s.demo_duration_sec = max(1.0, float(s.demo_duration_sec))
    s.max_frames = max(0, int(s.max_frames))
    s.process_every_n_frames = max(1, int(s.process_every_n_frames))

    s.confidence_threshold = float(np.clip(s.confidence_threshold, 0.01, 0.99))
    s.nms_iou = float(np.clip(s.nms_iou, 0.05, 0.95))

    s.track_high_thresh = float(np.clip(s.track_high_thresh, 0.05, 0.99))
    s.track_low_thresh = float(np.clip(s.track_low_thresh, 0.0, 0.98))
    if s.track_low_thresh >= s.track_high_thresh:
        s.track_low_thresh = max(0.0, s.track_high_thresh - 0.10)

    s.match_thresh = float(np.clip(s.match_thresh, 0.01, 0.95))
    s.track_buffer = max(1, int(s.track_buffer))
    s.min_hits = max(1, int(s.min_hits))

    s.speed_window_sec = float(np.clip(s.speed_window_sec, 0.2, 5.0))
    s.speed_min_samples = max(1, int(s.speed_min_samples))

    s.min_track_frames = max(1, int(s.min_track_frames))
    s.overspeed_margin_kmh = max(0.0, float(s.overspeed_margin_kmh))
    s.confirm_frames = max(1, int(s.confirm_frames))
    s.clear_frames = max(1, int(s.clear_frames))
    s.dedupe_cooldown_sec = max(0.0, float(s.dedupe_cooldown_sec))

    s.left_lane_limit_kmh = max(0.0, float(s.left_lane_limit_kmh))
    s.right_lane_limit_kmh = max(0.0, float(s.right_lane_limit_kmh))

    s.road_width_m = max(1.0, float(s.road_width_m))
    s.road_length_m = max(2.0, float(s.road_length_m))
    s.zone_start_m = max(0.0, float(s.zone_start_m))
    s.zone_end_m = max(s.zone_start_m + 1.0, float(s.zone_end_m))

    s.blur_strength = max(3, int(s.blur_strength))
    s.clip_seconds = max(0.5, float(s.clip_seconds))
    s.retention_days = max(1, int(s.retention_days))

    return s


def save_uploaded_file(uploaded) -> str:
    suffix = os.path.splitext(getattr(uploaded, "name", ""))[1] or ".mp4"
    path = os.path.join(tempfile.gettempdir(), f"vse_input_{uuid.uuid4().hex[:8]}{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# ----------------------------
# Geometry / homography
# ----------------------------

def make_metric_targets(width_m: float, length_m: float) -> np.ndarray:
    return np.array(
        [
            [-width_m / 2.0, 0.0],
            [width_m / 2.0, 0.0],
            [width_m / 2.0, length_m],
            [-width_m / 2.0, length_m],
        ],
        dtype=np.float32,
    )


def project_metric_to_pixel(
    H_inv: np.ndarray, pts: np.ndarray, check: bool = True
) -> Optional[np.ndarray]:
    pts = np.asarray(pts, dtype=np.float32)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)

    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homog = np.hstack([pts, ones])
    out = (H_inv @ homog.T).T
    w = out[:, 2]

    if check and np.any(w <= 1e-6):
        return None

    safe_w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    pix = out[:, :2] / safe_w[:, None]

    if check and (
        not np.all(np.isfinite(pix))
        or np.any(pix < -10000)
        or np.any(pix > 10000)
    ):
        return None

    return pix[0] if single else pix


def project_pixel_to_metric(
    H: np.ndarray, pts: np.ndarray, check: bool = True
) -> Optional[np.ndarray]:
    pts = np.asarray(pts, dtype=np.float32)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)

    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homog = np.hstack([pts, ones])
    out = (H @ homog.T).T
    w = out[:, 2]

    if check and np.any(w <= 1e-6):
        return None

    safe_w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    metric = out[:, :2] / safe_w[:, None]

    if check and (
        not np.all(np.isfinite(metric))
        or np.any(metric < -10000)
        or np.any(metric > 10000)
    ):
        return None

    return metric[0] if single else metric


def compute_homography(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("Could not compute homography from selected points.")

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Homography is singular. Please choose better calibration points.") from exc

    reproj = project_metric_to_pixel(H_inv, dst, check=False)
    if reproj is None:
        errors = np.full(len(src), np.nan, dtype=float)
    else:
        errors = np.linalg.norm(reproj - src, axis=1)

    finite = errors[np.isfinite(errors)]
    rmse = float(np.sqrt(np.mean(finite ** 2))) if len(finite) else 999.0
    return H, H_inv, rmse, errors


def build_calibration(settings: Settings) -> CalibrationModel:
    src = np.asarray(settings.calib_src, dtype=np.float32)
    dst = make_metric_targets(float(settings.road_width_m), float(settings.road_length_m))
    H, H_inv, rmse, errors = compute_homography(src, dst)

    payload = {
        "src": src.tolist(),
        "dst": dst.tolist(),
        "H": H.tolist(),
    }
    calibration_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    return CalibrationModel(
        src=src,
        dst=dst,
        H=H,
        H_inv=H_inv,
        rmse=rmse,
        errors=errors,
        calibration_id=calibration_id,
    )


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


# ----------------------------
# Kalman filter
# ----------------------------

class KalmanXY:
    def __init__(self, x: float, y: float):
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 4.0, 4.0])
        self.r = 0.35

    def predict(self, dt: float):
        dt = float(max(dt, 1e-3))
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        Q = np.diag([0.08 * dt, 0.08 * dt, 1.2 * dt, 1.2 * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray):
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        z = np.asarray(z, dtype=float).reshape(2)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.r * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P

    def speed_mps(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


# ----------------------------
# Tracker state
# ----------------------------

@dataclass
class TrackState:
    id: int
    cls: str
    bbox: Tuple[int, int, int, int]
    conf: float
    last_ts: float

    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    kf: Optional[KalmanXY] = None
    history: deque = field(default_factory=lambda: deque(maxlen=300))
    trail: deque = field(default_factory=lambda: deque(maxlen=120))

    metric_x: float = 0.0
    metric_y: float = 0.0

    speed_kmh: float = 0.0
    speed_ci: float = 0.0
    speed_valid: bool = False

    lane: str = "unknown"
    ref: str = ""

    true_speed_kmh: Optional[float] = None
    hint_id: Optional[str] = None
    last_metric_ts: Optional[float] = None


# ----------------------------
# ByteTrack-like tracker
# ----------------------------

class ByteLikeTracker:
    def __init__(self):
        self.tracks: List[TrackState] = []
        self.next_id = 1
        self.id_switches = 0
        self.total_tracks = 0

    def _associate(
        self,
        tracks: List[TrackState],
        dets: List[Detection],
        thresh: float,
    ):
        pairs = []
        for ti, tr in enumerate(tracks):
            for di, de in enumerate(dets):
                iou = bbox_iou(tr.bbox, de.bbox)
                if iou >= thresh:
                    pairs.append((iou, ti, di))

        pairs.sort(reverse=True, key=lambda x: x[0])

        matched_t = set()
        matched_d = set()
        matches = []

        for _, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            matches.append((ti, di))

        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(dets)) if i not in matched_d]
        return matches, unmatched_t, unmatched_d

    def _update_track(self, tr: TrackState, det: Detection, ts: float):
        if tr.hint_id and det.hint_id and tr.hint_id != det.hint_id:
            self.id_switches += 1

        tr.bbox = det.bbox
        tr.conf = det.conf
        tr.cls = det.cls
        tr.hits += 1
        tr.time_since_update = 0
        tr.last_ts = ts
        tr.true_speed_kmh = det.true_speed_kmh
        tr.hint_id = det.hint_id

    def update(
        self,
        dets: List[Detection],
        ts: float,
        settings: Settings,
    ) -> List[TrackState]:
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        high = [d for d in dets if d.conf >= settings.track_high_thresh]
        low = [
            d
            for d in dets
            if settings.track_low_thresh <= d.conf < settings.track_high_thresh
        ]

        matches, unmatched_t, unmatched_d = self._associate(
            self.tracks, high, settings.match_thresh
        )

        for ti, di in matches:
            self._update_track(self.tracks[ti], high[di], ts)

        unmatched_tracks = [self.tracks[i] for i in unmatched_t]
        low_match_thresh = max(0.03, settings.match_thresh * 0.65)
        matches2, _, _ = self._associate(unmatched_tracks, low, low_match_thresh)

        for ti, di in matches2:
            self._update_track(unmatched_tracks[ti], low[di], ts)

        for di in unmatched_d:
            det = high[di]
            tr = TrackState(
                id=self.next_id,
                cls=det.cls,
                bbox=det.bbox,
                conf=det.conf,
                last_ts=ts,
                true_speed_kmh=det.true_speed_kmh,
                hint_id=det.hint_id,
            )
            self.next_id += 1
            self.total_tracks += 1
            self.tracks.append(tr)

        self.tracks = [
            tr for tr in self.tracks if tr.time_since_update <= settings.track_buffer
        ]
        return self.tracks


# ----------------------------
# Geometry module
# ----------------------------

class Geometry:
    def __init__(self, H: np.ndarray, H_inv: np.ndarray, salt: str):
        self.H = H
        self.H_inv = H_inv
        self.salt = salt

    def make_ref(self, track_id: int) -> str:
        raw = f"{self.salt}:{track_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:10]

    def update(
        self,
        tracks: List[TrackState],
        ts: float,
        settings: Settings,
    ):
        for tr in tracks:
            if tr.time_since_update != 0:
                continue

            x1, y1, x2, y2 = tr.bbox
            bottom_center = np.array([(x1 + x2) / 2.0, float(y2)], dtype=np.float32)
            mp = project_pixel_to_metric(self.H, bottom_center.reshape(1, 2), check=True)

            if mp is None:
                continue

            mp = mp[0]
            if not (-30.0 < mp[0] < 30.0 and -15.0 < mp[1] < 100.0):
                continue

            if tr.kf is None:
                tr.kf = KalmanXY(float(mp[0]), float(mp[1]))
                dt = 1.0 / max(1.0, settings.fps)
            else:
                if tr.last_metric_ts is None:
                    dt = 1.0 / max(1.0, settings.fps)
                else:
                    dt = float(np.clip(ts - tr.last_metric_ts, 1.0 / 120.0, 0.5))
                tr.kf.predict(dt)
                tr.kf.update(np.array([mp[0], mp[1]], dtype=float))

            tr.last_metric_ts = ts
            tr.metric_x = float(mp[0])
            tr.metric_y = float(mp[1])
            tr.history.append((ts, float(mp[0]), float(mp[1])))

            max_age = max(1.0, settings.speed_window_sec + 0.5)
            while tr.history and ts - tr.history[0][0] > max_age:
                tr.history.popleft()

            if tr.kf is not None:
                tr.trail.append((float(tr.kf.x[0]), float(tr.kf.x[1])))

            samples = []
            prev = None
            for item in tr.history:
                if prev is not None:
                    dt = item[0] - prev[0]
                    if 0.02 < dt < 1.5:
                        dist = math.hypot(item[1] - prev[1], item[2] - prev[2])
                        if dist < 80.0:
                            samples.append(dist / dt)
                prev = item

            if tr.kf is not None:
                kf_speed = tr.kf.speed_mps()
                if kf_speed < 60.0:
                    samples.append(kf_speed)

            if len(samples) >= max(1, int(settings.speed_min_samples)):
                tr.speed_kmh = float(np.median(samples) * 3.6)
                if len(samples) > 1:
                    tr.speed_ci = float(
                        1.96 * np.std(samples) / math.sqrt(len(samples)) * 3.6
                    )
                else:
                    tr.speed_ci = 0.0
                tr.speed_valid = True
            else:
                tr.speed_kmh = 0.0
                tr.speed_ci = 0.0
                tr.speed_valid = False

            if tr.kf is not None:
                tr.lane = "left" if tr.kf.x[0] < 0 else "right"
            else:
                tr.lane = "unknown"

            tr.ref = self.make_ref(tr.id)


# ----------------------------
# Detectors
# ----------------------------

class MotionDetector:
    name = "motion"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        mask = self.bg.apply(frame)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        dets: List[Detection] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = float(bw * bh)
            if area < 600 or area > 0.38 * frame_area:
                continue
            aspect = float(bw) / float(max(1, bh))
            if aspect < 0.35 or aspect > 4.5:
                continue

            conf = float(np.clip(0.22 + area / (0.08 * frame_area), 0.15, 0.88))
            dets.append(
                Detection(
                    bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                    conf=conf,
                    cls="vehicle",
                )
            )

        return dets


class YoloDetector:
    name = "yolo"

    def __init__(self, settings: Settings):
        if not HAS_YOLO:
            raise RuntimeError("ultralytics is not installed")

        self.settings = settings
        self.model = YOLO(settings.yolo_model)
        self.class_map = {
            2: "car",
            3: "unknown",
            5: "bus",
            7: "truck",
        }

        device = settings.device
        if device == "auto":
            device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(settings.half and self.device == "cuda")

        dummy = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
        try:
            self.model(dummy, verbose=False, device=self.device, half=self.half)
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.settings.confidence_threshold,
            iou=self.settings.nms_iou,
            verbose=False,
            device=self.device,
            half=self.half,
        )

        dets: List[Detection] = []
        if not results:
            return dets

        boxes = results[0].boxes
        for b in boxes:
            cls_id = int(b.cls[0])
            if cls_id not in self.class_map:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            dets.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    cls=self.class_map[cls_id],
                )
            )
        return dets


def make_detector(settings: Settings):
    backend = settings.detector_backend

    if backend == "yolo" or (backend == "auto" and HAS_YOLO):
        try:
            return YoloDetector(settings)
        except Exception:
            return MotionDetector(settings)

    return MotionDetector(settings)


# ----------------------------
# Synthetic demo scene
# ----------------------------

class SyntheticScene:
    def __init__(self, settings: Settings, calibration: CalibrationModel):
        self.settings = settings
        self.calibration = calibration
        self.w = int(settings.width)
        self.h = int(settings.height)
        self.fps = float(settings.fps)
        self.visible_y_max = max(15.0, float(settings.road_length_m) + 5.0)

        seed = abs(hash(settings.scenario)) % 100000
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        self.vehicles = self._make_vehicles()
        self.background = self._make_background()

    def _make_vehicles(self) -> List[SynthVehicle]:
        s = self.settings.scenario

        if s == "stop_go":
            return [
                SynthVehicle("q1", "car", "left", -2.0, 7.0, 4.2, 1.8, (100, 205, 110), "queue", 0.2, 5.5),
                SynthVehicle("q2", "car", "right", -7.0, 6.0, 4.2, 1.8, (230, 210, 130), "queue", 1.1, 5.0),
                SynthVehicle("q3", "car", "left", -13.0, 8.0, 4.2, 1.8, (170, 170, 230), "queue", 2.0, 6.0),
                SynthVehicle("q4", "truck", "right", -20.0, 5.0, 9.0, 2.5, (120, 125, 135), "queue", 0.7, 3.0),
            ]

        if s == "night_glare":
            return [
                SynthVehicle("n1", "car", "left", -4.0, 58.0, 4.3, 1.8, (90, 170, 255), "constant", 0.0, 0.0, True),
                SynthVehicle("n2", "car", "right", -9.0, 69.0, 4.4, 1.8, (60, 70, 220), "constant", 0.7, 0.0),
                SynthVehicle("n3", "truck", "left", -18.0, 45.0, 9.0, 2.5, (120, 125, 135), "constant", 1.2, 0.0),
                SynthVehicle("n4", "car", "right", -27.0, 51.0, 4.2, 1.8, (200, 130, 80), "constant", 1.8, 0.0),
            ]

        if s == "occlusion":
            return [
                SynthVehicle("o1", "truck", "left", 6.0, 45.0, 9.0, 2.5, (120, 125, 135), "constant", 0.0, 0.0),
                SynthVehicle("o2", "car", "left", 8.2, 72.0, 4.3, 1.8, (90, 170, 255), "constant", 0.3, 0.0, True),
                SynthVehicle("o3", "car", "right", -5.0, 54.0, 4.2, 1.8, (180, 90, 190), "constant", 1.0, 0.0),
                SynthVehicle("o4", "car", "right", -18.0, 49.0, 4.2, 1.8, (230, 210, 130), "constant", 1.7, 0.0),
            ]

        if s == "camera_shake":
            return [
                SynthVehicle("c1", "car", "left", -3.0, 60.0, 4.3, 1.8, (90, 170, 255), "constant", 0.0, 0.0, True),
                SynthVehicle("c2", "car", "right", -8.0, 71.0, 4.4, 1.8, (60, 70, 220), "constant", 0.7, 0.0),
                SynthVehicle("c3", "truck", "left", -16.0, 44.0, 9.0, 2.5, (120, 125, 135), "constant", 1.2, 0.0),
                SynthVehicle("c4", "car", "right", -24.0, 52.0, 4.2, 1.8, (200, 130, 80), "constant", 1.8, 0.0),
            ]

        # free_flow default
        return [
            SynthVehicle("f1", "car", "left", -4.0, 60.0, 4.3, 1.8, (90, 170, 255), "constant", 0.0, 0.0, True),
            SynthVehicle("f2", "car", "right", -8.0, 72.0, 4.4, 1.8, (60, 70, 220), "constant", 0.7, 0.0),
            SynthVehicle("f3", "truck", "left", -14.0, 45.0, 9.0, 2.5, (120, 125, 135), "constant", 1.2, 0.0),
            SynthVehicle("f4", "car", "right", -18.0, 48.0, 4.2, 1.8, (200, 130, 80), "constant", 1.8, 0.0),
            SynthVehicle("f5", "car", "right", -26.0, 68.0, 4.3, 1.8, (180, 90, 190), "constant", 2.4, 0.0),
            SynthVehicle("f6", "car", "left", -2.0, 0.0, 4.2, 1.8, (100, 205, 110), "queue", 0.2, 3.0),
            SynthVehicle("f7", "car", "right", -7.0, 0.0, 4.2, 1.8, (230, 210, 130), "queue", 1.1, 3.0),
        ]

    def _make_background(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (45, 55, 45)

        H_inv = self.calibration.H_inv
        road_half = float(self.settings.road_width_m) / 2.0

        shoulder = project_metric_to_pixel(
            H_inv,
            np.array(
                [
                    [-road_half - 5.0, 0.0],
                    [road_half + 5.0, 0.0],
                    [road_half + 5.0, self.visible_y_max],
                    [-road_half - 5.0, self.visible_y_max],
                ]
            ),
            check=True,
        )
        if shoulder is not None:
            cv2.fillPoly(img, [shoulder.astype(np.int32)], (38, 72, 38))

        road = project_metric_to_pixel(
            H_inv,
            np.array(
                [
                    [-road_half - 0.2, 0.0],
                    [road_half + 0.2, 0.0],
                    [road_half + 0.2, self.visible_y_max],
                    [-road_half - 0.2, self.visible_y_max],
                ]
            ),
            check=True,
        )
        if road is not None:
            cv2.fillPoly(img, [road.astype(np.int32)], (72, 72, 78))

        self._draw_metric_line(img, (-road_half, 0.0), (-road_half, self.visible_y_max), (0, 220, 220), 3)
        self._draw_metric_line(img, (road_half, 0.0), (road_half, self.visible_y_max), (0, 220, 220), 3)

        for y0 in np.arange(0.0, self.visible_y_max, 4.0):
            y1 = min(y0 + 2.0, self.visible_y_max)
            self._draw_metric_line(img, (0.0, y0), (0.0, y1), (245, 245, 245), 2)

        noise = self.rng.normal(0.0, 2.0, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _draw_metric_line(
        self,
        img: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        color: Tuple[int, int, int],
        thickness: int,
    ):
        pts = project_metric_to_pixel(
            self.calibration.H_inv,
            np.array([p1, p2], dtype=np.float32),
            check=True,
        )
        if pts is None:
            return
        cv2.line(
            img,
            tuple(pts[0].astype(int)),
            tuple(pts[1].astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    def step(self, dt: float) -> Tuple[np.ndarray, List[Detection]]:
        self.t += dt

        for v in self.vehicles:
            if v.motion == "queue":
                v.current_speed_kmh = max(
                    0.0,
                    v.speed_kmh + v.amp * math.sin(2.0 * math.pi * 0.13 * self.t + v.phase),
                )
            else:
                v.current_speed_kmh = v.speed_kmh

            v.y += v.current_speed_kmh / 3.6 * dt

            if v.y > self.visible_y_max + v.length:
                v.y = -float(self.rng.uniform(1.0, 12.0))

        return self._render()

    def _render(self) -> Tuple[np.ndarray, List[Detection]]:
        frame = self.background.copy()
        H_inv = self.calibration.H_inv
        road_half = float(self.settings.road_width_m) / 2.0

        dets: List[Detection] = []
        glare_points: List[Tuple[Tuple[int, int], int]] = []

        sorted_vehicles = sorted(self.vehicles, key=lambda x: x.y, reverse=True)

        for v in sorted_vehicles:
            if v.y + v.length < 0.0 or v.y > self.visible_y_max:
                continue

            lane_x = -road_half / 2.0 if v.lane == "left" else road_half / 2.0
            x = lane_x + 0.06 * math.sin(2.0 * math.pi * 0.20 * self.t + v.phase)

            corners_metric = np.array(
                [
                    [x - v.width / 2.0, v.y],
                    [x + v.width / 2.0, v.y],
                    [x + v.width / 2.0, v.y + v.length],
                    [x - v.width / 2.0, v.y + v.length],
                ],
                dtype=np.float32,
            )

            pix = project_metric_to_pixel(H_inv, corners_metric, check=True)
            if pix is None:
                continue

            pix_int = pix.astype(np.int32)

            cv2.fillPoly(frame, [pix_int], v.color)
            cv2.polylines(frame, [pix_int], True, (25, 25, 25), 2, cv2.LINE_AA)

            center = pix_int.mean(axis=0)
            roof = ((pix_int - center) * 0.62 + center).astype(np.int32)
            darker = tuple(max(0, int(c) - 45) for c in v.color)
            cv2.fillPoly(frame, [roof], darker)

            p0, p1 = pix_int[0], pix_int[1]
            bottom_mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2))

            plate_w = max(10, int(np.linalg.norm(p1 - p0) * 0.32))
            plate_h = max(5, int(plate_w * 0.34))

            x1 = bottom_mid[0] - plate_w // 2
            y1 = bottom_mid[1] - plate_h // 2
            x2 = x1 + plate_w
            y2 = y1 + plate_h

            x1 = int(np.clip(x1, 0, self.w - 1))
            y1 = int(np.clip(y1, 0, self.h - 1))
            x2 = int(np.clip(x2, 0, self.w - 1))
            y2 = int(np.clip(y2, 0, self.h - 1))

            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

                if self.settings.blur_regions:
                    roi = frame[y1:y2, x1:x2]
                    k = max(3, int(self.settings.blur_strength) // 2 * 2 + 1)
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

            bx_min = int(np.clip(np.floor(pix_int[:, 0].min()) - 1, 0, self.w - 1))
            bx_max = int(np.clip(np.ceil(pix_int[:, 0].max()) + 1, 0, self.w - 1))
            by_min = int(np.clip(np.floor(pix_int[:, 1].min()) - 1, 0, self.h - 1))
            by_max = int(np.clip(np.ceil(pix_int[:, 1].max()) + 1, 0, self.h - 1))

            if bx_max <= bx_min:
                bx_max = min(self.w - 1, bx_min + 8)
            if by_max <= by_min:
                by_max = min(self.h - 1, by_min + 8)

            conf = 0.96 - 0.004 * max(0.0, v.y)
            conf += 0.015 * math.sin(2.0 * math.pi * 0.35 * self.t + v.phase)
            if v.cls == "truck":
                conf += 0.02
            if self.settings.scenario == "night_glare" or self.settings.night_glare:
                conf -= 0.10
            conf = float(np.clip(conf, 0.10, 0.99))

            dets.append(
                Detection(
                    bbox=(bx_min, by_min, bx_max, by_max),
                    conf=conf,
                    cls=v.cls,
                    true_speed_kmh=float(v.current_speed_kmh),
                    true_x=float(x),
                    true_y=float(v.y),
                    hint_id=v.vid,
                )
            )

            radius = max(18, int((bx_max - bx_min) * 0.38))
            glare_points.append(((int(center[0]), int(center[1])), radius))

        # Simple occlusion model: truck suppresses smaller overlapping vehicle detections.
        for i, d1 in enumerate(dets):
            for j, d2 in enumerate(dets):
                if i == j:
                    continue
                if d1.cls == "truck" and d2.cls != "truck":
                    iou = bbox_iou(d1.bbox, d2.bbox)
                    if iou > 0.16:
                        d2.conf = float(min(d2.conf, 0.10))

        if self.settings.scenario == "night_glare" or self.settings.night_glare:
            frame = self._apply_night_glare(frame, glare_points)

        if self.settings.scenario == "camera_shake" or self.settings.camera_shake:
            frame, dets = self._apply_camera_shake(frame, dets)

        return frame, dets

    def _apply_night_glare(
        self,
        frame: np.ndarray,
        points: List[Tuple[Tuple[int, int], int]],
    ) -> np.ndarray:
        dark = cv2.convertScaleAbs(frame, alpha=0.40, beta=0)
        glare = np.zeros_like(dark)

        for pt, r in points:
            cv2.circle(glare, pt, r, (255, 230, 180), -1)

        glare = cv2.GaussianBlur(glare, (0, 0), 25)
        return cv2.addWeighted(dark, 1.0, glare, 0.58, 0)

    def _apply_camera_shake(
        self,
        frame: np.ndarray,
        dets: List[Detection],
    ) -> Tuple[np.ndarray, List[Detection]]:
        dx = float(self.rng.normal(0.0, 1.2))
        dy = float(self.rng.normal(0.0, 0.9))
        M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])

        frame = cv2.warpAffine(frame, M, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)

        for d in dets:
            x1, y1, x2, y2 = d.bbox
            pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )
            pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
            tp = (M @ pts_h.T).T

            nx1 = int(np.clip(tp[:, 0].min(), 0, self.w - 1))
            ny1 = int(np.clip(tp[:, 1].min(), 0, self.h - 1))
            nx2 = int(np.clip(tp[:, 0].max(), 0, self.w - 1))
            ny2 = int(np.clip(tp[:, 1].max(), 0, self.h - 1))

            if nx2 <= nx1:
                nx2 = min(self.w - 1, nx1 + 8)
            if ny2 <= ny1:
                ny2 = min(self.h - 1, ny1 + 8)

            d.bbox = (nx1, ny1, nx2, ny2)

        return frame, dets


# ----------------------------
# Event schema validation
# ----------------------------

REQUIRED_EVENT_FIELDS = [
    "event_id",
    "vehicle_id",
    "class",
    "lane",
    "speed_kmh",
    "speed_ci",
    "track_len_frames",
    "first_ts",
    "ts",
    "violation",
    "limit_kmh",
    "calibration_id",
]


def validate_event(event: Dict) -> List[str]:
    errors: List[str] = []

    for k in REQUIRED_EVENT_FIELDS:
        if k not in event:
            errors.append(f"missing:{k}")

    if "speed_kmh" in event and sanitize_float(event["speed_kmh"], -1) < 0:
        errors.append("speed_kmh_negative")

    if "limit_kmh" in event and sanitize_float(event["limit_kmh"], -1) < 0:
        errors.append("limit_kmh_negative")

    if "lane" in event and event["lane"] not in ("left", "right", "unknown"):
        errors.append("bad_lane")

    if "class" in event and event["class"] not in ("car", "truck", "bus", "unknown", "vehicle"):
        errors.append("bad_class")

    if "violation" in event and not isinstance(event["violation"], bool):
        errors.append("violation_not_bool")

    return errors


# ----------------------------
# Storage
# ----------------------------

class Storage:
    def __init__(self, evidence_dir: str, run_id: str, meta: Optional[Dict] = None):
        self.evidence_dir = evidence_dir
        self.run_id = run_id
        self.jsonl_path = os.path.join(evidence_dir, "events.jsonl")
        self.db_path = os.path.join(evidence_dir, "events.sqlite")
        meta = meta or {}

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                mode TEXT,
                scenario TEXT,
                calibration_id TEXT,
                rmse_px REAL,
                meta TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT,
                ts REAL,
                first_ts REAL,
                vehicle_id TEXT,
                class TEXT,
                lane TEXT,
                speed_kmh REAL,
                speed_ci REAL,
                track_len_frames INTEGER,
                violation INTEGER,
                limit_kmh REAL,
                severity TEXT,
                evidence_snapshot TEXT,
                evidence_clip TEXT,
                calibration_id TEXT,
                payload TEXT
            )
            """
        )
        self.conn.commit()

        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, started_at, mode, scenario, calibration_id, rmse_px, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                utc_now_iso(),
                meta.get("mode", ""),
                meta.get("scenario", ""),
                meta.get("calibration_id", ""),
                float(meta.get("rmse_px", 0.0)),
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def append(self, event: Dict):
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass

        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, run_id, ts, first_ts, vehicle_id, class, lane,
                    speed_kmh, speed_ci, track_len_frames, violation, limit_kmh,
                    severity, evidence_snapshot, evidence_clip, calibration_id, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("event_id"),
                    event.get("run_id"),
                    event.get("ts"),
                    event.get("first_ts"),
                    event.get("vehicle_id"),
                    event.get("class"),
                    event.get("lane"),
                    event.get("speed_kmh"),
                    event.get("speed_ci"),
                    event.get("track_len_frames"),
                    1 if event.get("violation") else 0,
                    event.get("limit_kmh"),
                    event.get("severity"),
                    event.get("evidence_snapshot"),
                    event.get("evidence_clip"),
                    event.get("calibration_id"),
                    json.dumps(event, ensure_ascii=False),
                ),
            )
            self.conn.commit()
        except sqlite3.Error:
            pass

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ----------------------------
# Event engine
# ----------------------------

class EventEngine:
    def __init__(
        self,
        evidence_dir: str,
        run_id: str,
        calibration_id: str,
        meta: Optional[Dict] = None,
    ):
        self.evidence_dir = evidence_dir
        self.run_id = run_id
        self.calibration_id = calibration_id
        self.storage = Storage(evidence_dir, run_id, meta)
        self.events: List[Dict] = []
        self.active: Dict[int, Dict] = {}
        self.last_event_time: Dict[Tuple[str, str], float] = {}

    def evaluate(
        self,
        tracks: List[TrackState],
        ts: float,
        frame: np.ndarray,
        frame_buffer: deque,
        settings: Settings,
        calibration_id: str,
    ) -> List[Dict]:
        emitted = []

        for tr in tracks:
            if tr.time_since_update != 0 or not tr.ref or not tr.speed_valid:
                continue

            if tr.hits < settings.min_track_frames:
                continue

            y_metric = float(tr.kf.x[1]) if tr.kf is not None else float(tr.metric_y)
            if y_metric < settings.zone_start_m or y_metric > settings.zone_end_m:
                continue

            limit = (
                settings.left_lane_limit_kmh
                if tr.lane == "left"
                else settings.right_lane_limit_kmh
            )

            state = self.active.setdefault(
                tr.id,
                {"exceed": 0, "clear": 0, "active": False},
            )

            if tr.speed_kmh > limit + settings.overspeed_margin_kmh:
                state["exceed"] += 1
                state["clear"] = 0
            elif tr.speed_kmh < limit - settings.overspeed_margin_kmh:
                state["clear"] += 1
                state["exceed"] = 0
                if state["active"] and state["clear"] >= settings.clear_frames:
                    state["active"] = False
            else:
                state["exceed"] = 0
                state["clear"] = 0

            if (not state["active"]) and state["exceed"] >= settings.confirm_frames:
                key = (tr.ref, tr.lane)
                if ts - self.last_event_time.get(key, -1e9) >= settings.dedupe_cooldown_sec:
                    event = self._create_event(
                        tr=tr,
                        ts=ts,
                        limit=limit,
                        settings=settings,
                        frame=frame,
                        frame_buffer=frame_buffer,
                        calibration_id=calibration_id,
                    )
                    errors = validate_event(event)
                    if errors:
                        event["validation_errors"] = errors

                    emitted.append(event)
                    self.storage.append(event)
                    self.events.append(event)
                    self.last_event_time[key] = ts
                    state["active"] = True

            self.active[tr.id] = state

        return emitted

    def _create_event(
        self,
        tr: TrackState,
        ts: float,
        limit: float,
        settings: Settings,
        frame: np.ndarray,
        frame_buffer: deque,
        calibration_id: str,
    ) -> Dict:
        event_id = uuid.uuid4().hex[:12]

        if settings.save_evidence:
            snap_rel, clip_rel = self._save_evidence(
                event_id, ts, frame, frame_buffer, settings
            )
        else:
            snap_rel, clip_rel = None, None

        over = float(tr.speed_kmh - limit)
        if over >= 20.0:
            severity = "high"
        elif over >= 10.0:
            severity = "medium"
        else:
            severity = "low"

        first_ts = float(tr.history[0][0]) if tr.history else float(ts)
        vehicle_class = tr.cls if tr.cls in ("car", "truck", "bus", "unknown", "vehicle") else "unknown"

        return {
            "event_id": event_id,
            "run_id": self.run_id,
            "ts": float(ts),
            "first_ts": first_ts,
            "vehicle_id": tr.ref,
            "class": vehicle_class,
            "lane": tr.lane,
            "speed_kmh": round(float(tr.speed_kmh), 2),
            "speed_ci": round(float(tr.speed_ci), 2),
            "track_len_frames": int(tr.hits),
            "violation": True,
            "limit_kmh": round(float(limit), 2),
            "severity": severity,
            "evidence_snapshot": snap_rel,
            "evidence_clip": clip_rel,
            "calibration_id": calibration_id,
            "label": DISCLAIMER,
            "created_at": utc_now_iso(),
        }

    def _save_evidence(
        self,
        event_id: str,
        ts: float,
        frame: np.ndarray,
        frame_buffer: deque,
        settings: Settings,
    ) -> Tuple[Optional[str], Optional[str]]:
        snap_rel = f"snapshot_{event_id}.jpg"
        snap_abs = os.path.join(self.evidence_dir, snap_rel)

        try:
            cv2.imwrite(snap_abs, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception:
            snap_rel = None

        clip_rel = None

        try:
            selected = [item for item in frame_buffer if item[0] >= ts - settings.clip_seconds]
            if len(selected) >= 3:
                base = os.path.join(self.evidence_dir, f"clip_{event_id}")

                for fourcc_name, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")]:
                    path = base + ext
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    writer = cv2.VideoWriter(
                        path,
                        fourcc,
                        float(max(1.0, settings.fps)),
                        (int(settings.width), int(settings.height)),
                    )

                    if not writer.isOpened():
                        continue

                    ok = True
                    for _, jpg in selected:
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is None or img.shape[:2] != (
                            int(settings.height),
                            int(settings.width),
                        ):
                            ok = False
                            break
                        writer.write(img)

                    writer.release()

                    if ok and os.path.exists(path) and os.path.getsize(path) > 0:
                        clip_rel = os.path.basename(path)
                        break

                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except Exception:
            clip_rel = None

        return snap_rel, clip_rel

    def close(self):
        self.storage.close()


# ----------------------------
# Threaded frame source for real video/webcam/RTSP
# ----------------------------

class FrameSource:
    def __init__(self, settings: Settings):
        self.mode = settings.mode
        self.w = int(settings.width)
        self.h = int(settings.height)
        self.sample = max(1, int(settings.process_every_n_frames))
        self.max_frames = max(0, int(settings.max_frames))

        self.q: queue.Queue = queue.Queue(maxsize=30 if self.mode == "file" else 3)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.fps = max(1.0, float(settings.fps))
        self.total_frames = 0
        self.dropped = 0
        self.raw_idx = 0
        self.started = time.time()

        self._open(settings)

    def _open(self, settings: Settings):
        if self.mode == "file":
            if not settings.source_path or not os.path.exists(settings.source_path):
                raise RuntimeError("Video file path is missing or invalid.")
            self.cap = cv2.VideoCapture(settings.source_path)
        elif self.mode == "webcam":
            self.cap = cv2.VideoCapture(int(settings.webcam_index))
        elif self.mode == "rtsp":
            if not settings.rtsp_url:
                raise RuntimeError("RTSP URL is empty.")
            self.cap = cv2.VideoCapture(settings.rtsp_url)
        else:
            raise RuntimeError(f"Unsupported mode: {self.mode}")

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.mode}")

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.h))
            self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        except Exception:
            pass

        fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
        if fps_prop and fps_prop > 0:
            self.fps = float(fps_prop)

        frames_prop = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frames_prop and frames_prop > 0:
            self.total_frames = int(frames_prop)

    def start(self):
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def _put(self, item):
        if self.stop_event.is_set():
            return

        if self.mode == "file":
            while not self.stop_event.is_set():
                try:
                    self.q.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
        else:
            try:
                self.q.put_nowait(item)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    self.dropped += 1

    def _reader(self):
        if self.cap is None:
            self._put(None)
            return

        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._put(None)
                break

            self.raw_idx += 1

            if self.sample > 1 and (self.raw_idx % self.sample) != 0:
                continue

            if self.mode == "file":
                msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if msec and msec > 0:
                    ts = float(msec) / 1000.0
                else:
                    ts = float(time.time() - self.started)
            else:
                ts = float(time.time() - self.started)

            if frame.shape[:2] != (self.h, self.w):
                frame = cv2.resize(frame, (self.w, self.h))

            self._put((frame, ts, int(self.raw_idx)))

        self._put(None)

    def get_frame(self, timeout: float = 0.0):
        try:
            if timeout <= 0:
                item = self.q.get_nowait()
            else:
                item = self.q.get(timeout=timeout)
        except queue.Empty:
            return None, False

        if item is None:
            return None, True

        return item, False


# ----------------------------
# Pipeline
# ----------------------------

class Pipeline:
    def __init__(self, settings: Settings):
        self.settings_snapshot = settings
        self.mode = settings.mode
        self.run_id = uuid.uuid4().hex[:12]

        self.evidence_root = get_evidence_root()
        self.run_dir = ensure_dir(os.path.join(self.evidence_root, self.run_id))

        self.calibration = build_calibration(settings)

        if not settings.hash_salt:
            settings.hash_salt = uuid.uuid4().hex[:8]

        self.geometry = Geometry(
            self.calibration.H,
            self.calibration.H_inv,
            settings.hash_salt,
        )
        self.tracker = ByteLikeTracker()

        self.event_engine = EventEngine(
            self.run_dir,
            self.run_id,
            self.calibration.calibration_id,
            meta={
                "mode": settings.mode,
                "scenario": settings.scenario,
                "calibration_id": self.calibration.calibration_id,
                "rmse_px": float(self.calibration.rmse),
            },
        )

        self.frame_buffer: deque = deque(maxlen=450)
        self.speed_samples: List[Tuple[float, str, str, float, float]] = []
        self.error_samples: List[Tuple[float, str, float, float, float]] = []
        self.stage_history: deque = deque(maxlen=180)
        self.fps_window: deque = deque(maxlen=120)

        self.finished = False
        self.frame_idx = 0
        self.dropped_frames = 0
        self.start_wall = time.time()

        self.last_frame: Optional[np.ndarray] = None
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_tracks: List[TrackState] = []
        self.last_stats: Dict = {}

        self.scene: Optional[SyntheticScene] = None
        self.source: Optional[FrameSource] = None
        self.detector = None
        self.backend = "synthetic"

        self.source_fps = max(1.0, float(settings.fps))
        self.total_frames = 0

        if self.mode == "demo":
            self.scene = SyntheticScene(settings, self.calibration)
            self.total_frames = int(max(1.0, settings.demo_duration_sec) * self.source_fps)
            if settings.max_frames > 0:
                self.total_frames = min(self.total_frames, int(settings.max_frames))
        else:
            self.source = FrameSource(settings)
            self.source.start()
            self.detector = make_detector(settings)
            self.backend = getattr(self.detector, "name", "motion")

            self.source_fps = max(1.0, float(self.source.fps))
            if self.source.total_frames > 0:
                self.total_frames = int(self.source.total_frames / max(1, self.source.sample))
            if settings.max_frames > 0:
                self.total_frames = min(self.total_frames, int(settings.max_frames)) if self.total_frames else int(settings.max_frames)

    def close(self):
        try:
            if self.source is not None:
                self.source.stop()
        except Exception:
            pass

        try:
            self.event_engine.close()
        except Exception:
            pass

    def process_next(self, settings: Settings):
        if self.finished:
            return None

        stage_start = time.perf_counter()

        if self.mode == "demo":
            if self.frame_idx >= self.total_frames:
                self.finished = True
                return None

            dt = 1.0 / max(1.0, self.source_fps)
            frame, dets = self.scene.step(dt)
            ts = float(self.frame_idx) / max(1.0, self.source_fps)
            self.frame_idx += 1
            self.last_raw_frame = frame.copy()

        else:
            if self.source is None:
                self.finished = True
                return None

            data, source_finished = self.source.get_frame(timeout=0.0)
            if source_finished:
                self.finished = True
                return None

            if data is None:
                return None

            frame, ts, raw_idx = data
            frame = np.ascontiguousarray(frame)
            self.last_raw_frame = frame.copy()
            self.frame_idx += 1

            if settings.max_frames > 0 and self.frame_idx > int(settings.max_frames):
                self.finished = True
                if self.source is not None:
                    self.source.stop()
                return None

            dets = self.detector.detect(frame) if self.detector is not None else []

        dets = [d for d in dets if d.conf >= settings.confidence_threshold]
        t_detect = time.perf_counter()

        tracks = self.tracker.update(dets, ts, settings)
        updated_tracks = [tr for tr in tracks if tr.time_since_update == 0]
        t_track = time.perf_counter()

        self.geometry.update(updated_tracks, ts, settings)
        t_geom = time.perf_counter()

        annotated = self._annotate(frame, tracks, updated_tracks, settings, ts)
        self.last_frame = annotated

        ok, jpg = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if ok:
            self.frame_buffer.append((ts, jpg.tobytes()))

        new_events = self.event_engine.evaluate(
            updated_tracks,
            ts,
            annotated,
            self.frame_buffer,
            settings,
            self.calibration.calibration_id,
        )
        t_event = time.perf_counter()

        self._record_samples(updated_tracks, ts, settings)

        proc_sec = max(1e-6, time.perf_counter() - stage_start)
        self.fps_window.append(time.perf_counter())

        rolling_fps = 0.0
        if len(self.fps_window) > 1:
            span = self.fps_window[-1] - self.fps_window[0]
            if span > 1e-6:
                rolling_fps = float(len(self.fps_window) - 1) / span

        mae = None
        within3 = None
        if self.error_samples:
            errs = [e[2] for e in self.error_samples]
            mae = float(np.mean(errs))
            within3 = float(100.0 * np.mean(np.asarray(errs) <= 3.0))

        dropped_total = int(self.dropped_frames + (self.source.dropped if self.source is not None else 0))

        stats = {
            "frame_index": self.frame_idx,
            "total_frames": self.total_frames,
            "timestamp": float(ts),
            "detect_ms": (t_detect - stage_start) * 1000.0,
            "track_ms": (t_track - t_detect) * 1000.0,
            "geometry_ms": (t_geom - t_track) * 1000.0,
            "event_ms": (t_event - t_geom) * 1000.0,
            "total_ms": proc_sec * 1000.0,
            "fps_processing": 1.0 / proc_sec,
            "rolling_fps": rolling_fps,
            "real_time_factor": proc_sec * self.source_fps,
            "dropped_frames": dropped_total,
            "active_tracks": len(updated_tracks),
            "total_tracks": int(self.tracker.total_tracks),
            "id_switches": int(self.tracker.id_switches),
            "events_total": len(self.event_engine.events),
            "calibration_rmse": float(self.calibration.rmse),
            "calibration_id": self.calibration.calibration_id,
            "mode": self.mode,
            "scenario": settings.scenario,
            "backend": self.backend,
            "ground_truth_mae_kmh": mae,
            "within_3_kmh_pct": within3,
            "new_events": new_events,
        }

        history_stats = {k: v for k, v in stats.items() if k != "new_events"}
        self.stage_history.append(history_stats)

        self.last_stats = stats
        self.last_tracks = tracks

        return annotated, stats, new_events

    def _record_samples(
        self,
        tracks: List[TrackState],
        ts: float,
        settings: Settings,
    ):
        for tr in tracks:
            if tr.time_since_update != 0 or not tr.speed_valid:
                continue

            self.speed_samples.append(
                (ts, tr.ref, tr.lane, float(tr.speed_kmh), float(tr.speed_ci))
            )

            if (
                tr.true_speed_kmh is not None
                and tr.hits >= settings.min_track_frames
                and self.mode == "demo"
            ):
                err = abs(float(tr.speed_kmh) - float(tr.true_speed_kmh))
                self.error_samples.append(
                    (ts, tr.ref, err, float(tr.speed_kmh), float(tr.true_speed_kmh))
                )

        if len(self.speed_samples) > 30000:
            self.speed_samples = self.speed_samples[-18000:]
        if len(self.error_samples) > 30000:
            self.error_samples = self.error_samples[-18000:]

    def _blur_plate_regions(
        self,
        img: np.ndarray,
        tracks: List[TrackState],
        settings: Settings,
    ):
        if not settings.blur_regions:
            return

        h, w = img.shape[:2]
        k = max(3, int(settings.blur_strength) // 2 * 2 + 1)

        for tr in tracks:
            if tr.time_since_update != 0:
                continue

            x1, y1, x2, y2 = tr.bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            pw = int(bw * 0.38)
            ph = int(max(6, bh * 0.14))

            cx = (x1 + x2) // 2
            cy = y2 - int(bh * 0.18)

            px1 = cx - pw // 2
            py1 = cy - ph // 2
            px2 = px1 + pw
            py2 = py1 + ph

            px1 = int(np.clip(px1, 0, w - 1))
            py1 = int(np.clip(py1, 0, h - 1))
            px2 = int(np.clip(px2, 0, w - 1))
            py2 = int(np.clip(py2, 0, h - 1))

            if px2 > px1 and py2 > py1:
                roi = img[py1:py2, px1:px2]
                img[py1:py2, px1:px2] = cv2.GaussianBlur(roi, (k, k), 0)

    def _annotate(
        self,
        frame: np.ndarray,
        tracks: List[TrackState],
        updated_tracks: List[TrackState],
        settings: Settings,
        ts: float,
    ) -> np.ndarray:
        img = frame.copy()
        road_half = float(settings.road_width_m) / 2.0

        if settings.show_overlay:
            # Zones
            for y_zone in [settings.zone_start_m, settings.zone_end_m]:
                pts = project_metric_to_pixel(
                    self.calibration.H_inv,
                    np.array([[-road_half, y_zone], [road_half, y_zone]], dtype=np.float32),
                    check=True,
                )
                if pts is not None:
                    cv2.line(
                        img,
                        tuple(pts[0].astype(int)),
                        tuple(pts[1].astype(int)),
                        (255, 180, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Tracks
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox

                limit = (
                    settings.left_lane_limit_kmh
                    if tr.lane == "left"
                    else settings.right_lane_limit_kmh
                )

                active = self.event_engine.active.get(tr.id, {}).get("active", False)
                provisional = tr.speed_valid and tr.speed_kmh > limit + settings.overspeed_margin_kmh

                if tr.time_since_update > 0:
                    color = (120, 120, 120)
                    thickness = 1
                elif active or provisional:
                    color = (0, 0, 255)
                    thickness = 3
                else:
                    color = (0, 210, 0)
                    thickness = 2

                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

                label = f"{tr.ref or tr.id} {tr.cls} {tr.speed_kmh:.1f} km/h"
                if tr.speed_valid:
                    label += f" ±{tr.speed_ci:.1f}"

                if (
                    settings.show_ground_truth
                    and self.mode == "demo"
                    and tr.true_speed_kmh is not None
                    and tr.speed_valid
                ):
                    err = tr.speed_kmh - tr.true_speed_kmh
                    label += f" GT:{tr.true_speed_kmh:.1f} E:{err:+.1f}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), (0, 0, 0), -1)
                cv2.putText(
                    img,
                    label,
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if len(tr.trail) > 1:
                    pts_list = []
                    for mx, my in tr.trail:
                        p = project_metric_to_pixel(
                            self.calibration.H_inv,
                            np.array([[mx, my]], dtype=np.float32),
                            check=True,
                        )
                        if p is None:
                            pts_list = []
                            continue
                        pts_list.append(p[0].astype(np.int32))

                    if len(pts_list) > 1:
                        cv2.polylines(
                            img,
                            [np.array(pts_list, dtype=np.int32)],
                            False,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # HUD
            cv2.putText(
                img,
                f"LEFT LIMIT {settings.left_lane_limit_kmh:.0f} km/h",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"RIGHT LIMIT {settings.right_lane_limit_kmh:.0f} km/h",
                (img.shape[1] - 250, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"{self.mode.upper()} | {settings.scenario} | {self.backend}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"FPS {self.last_stats.get('rolling_fps', 0):.1f} | RTF {self.last_stats.get('real_time_factor', 0):.2f}x | Drop {self.last_stats.get('dropped_frames', 0)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                DISCLAIMER.upper(),
                (20, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            for p in self.calibration.src:
                cv2.circle(img, tuple(p.astype(int)), 4, (255, 0, 255), -1)

        # Privacy blur always applies to active vehicle regions.
        self._blur_plate_regions(img, updated_tracks, settings)

        return img

    def save_buffer_video(self, path: str) -> Optional[str]:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                path,
                fourcc,
                float(max(1.0, self.source_fps)),
                (int(self.settings_snapshot.width), int(self.settings_snapshot.height)),
            )
            if not writer.isOpened():
                return None

            for _, jpg in self.frame_buffer:
                arr = np.frombuffer(jpg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None and img.shape[:2] == (
                    int(self.settings_snapshot.height),
                    int(self.settings_snapshot.width),
                ):
                    writer.write(img)

            writer.release()

            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
        except Exception:
            pass

        return None

    def calibration_preview_image(self) -> np.ndarray:
        if self.last_raw_frame is not None:
            img = self.last_raw_frame.copy()
        else:
            img = np.zeros(
                (int(self.settings_snapshot.height), int(self.settings_snapshot.width), 3),
                dtype=np.uint8,
            )

        for i, p in enumerate(self.calibration.src):
            pt = tuple(p.astype(int))
            cv2.circle(img, pt, 8, (255, 0, 255), -1)
            cv2.putText(
                img,
                f"P{i+1}",
                (pt[0] + 10, max(20, pt[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return img

    def bird_eye_preview(self, settings: Settings) -> np.ndarray:
        w, h = 820, 460
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (28, 28, 32)

        road_half = float(settings.road_width_m) / 2.0
        x_min, x_max = -road_half - 4.0, road_half + 4.0
        y_min, y_max = -2.0, max(35.0, float(settings.zone_end_m) + 5.0)

        def m2img(x: float, y: float) -> Tuple[int, int]:
            u = int((x - x_min) / (x_max - x_min) * (w - 1))
            v = int((1.0 - (y - y_min) / (y_max - y_min)) * (h - 1))
            return u, v

        for x in np.arange(x_min, x_max + 0.1, 2.0):
            p1 = m2img(float(x), y_min)
            p2 = m2img(float(x), y_max)
            cv2.line(img, p1, p2, (55, 55, 60), 1)

        for y in np.arange(y_min, y_max + 0.1, 5.0):
            p1 = m2img(x_min, float(y))
            p2 = m2img(x_max, float(y))
            cv2.line(img, p1, p2, (55, 55, 60), 1)

        cal_pts = [m2img(float(x), float(y)) for x, y in self.calibration.dst]
        cv2.polylines(img, [np.array(cal_pts, dtype=np.int32)], True, (0, 255, 255), 2)

        for y_zone in [settings.zone_start_m, settings.zone_end_m]:
            p1 = m2img(x_min, float(y_zone))
            p2 = m2img(x_max, float(y_zone))
            cv2.line(img, p1, p2, (0, 180, 255), 2)

        for tr in self.last_tracks:
            if tr.kf is None:
                continue

            x = float(tr.kf.x[0])
            y = float(tr.kf.x[1])
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            limit = (
                settings.left_lane_limit_kmh
                if tr.lane == "left"
                else settings.right_lane_limit_kmh
            )
            color = (0, 0, 255) if tr.speed_valid and tr.speed_kmh > limit + settings.overspeed_margin_kmh else (0, 220, 0)
            u, v = m2img(x, y)

            cv2.circle(img, (u, v), 6, color, -1)
            cv2.putText(
                img,
                f"{tr.ref or tr.id} {tr.speed_kmh:.0f}",
                (u + 8, max(12, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            "Bird's-eye metric preview (meters)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    def events_dataframe(self) -> pd.DataFrame:
        if not self.event_engine.events:
            return pd.DataFrame()
        return pd.DataFrame(self.event_engine.events)


# ----------------------------
# Intelligence / local report
# ----------------------------

def build_packet(pipe: Pipeline, settings: Settings) -> Dict:
    events = pipe.event_engine.events
    speeds = [s[3] for s in pipe.speed_samples]
    vehicle_refs = {s[1] for s in pipe.speed_samples}

    mae = None
    within3 = None
    if pipe.error_samples:
        errs = [e[2] for e in pipe.error_samples]
        mae = float(np.mean(errs))
        within3 = float(100.0 * np.mean(np.asarray(errs) <= 3.0))

    stage_avg = {}
    if pipe.stage_history:
        df = pd.DataFrame(list(pipe.stage_history))
        for col in ["detect_ms", "track_ms", "geometry_ms", "event_ms", "total_ms"]:
            if col in df.columns:
                stage_avg[col] = round(float(df[col].mean()), 3)

    return {
        "run_id": pipe.run_id,
        "created_at": utc_now_iso(),
        "mode": pipe.mode,
        "scenario": settings.scenario,
        "calibration_id": pipe.calibration.calibration_id,
        "calibration_rmse_px": round(float(pipe.calibration.rmse), 3),
        "window_sec": round(float(pipe.frame_idx / max(1.0, pipe.source_fps)), 2),
        "counts": {
            "tracked_vehicle_refs": len(vehicle_refs),
            "total_tracks": int(pipe.tracker.total_tracks),
            "id_switches": int(pipe.tracker.id_switches),
            "events": len(events),
            "overspeed_events": sum(1 for e in events if e.get("violation")),
            "dropped_frames": int(pipe.dropped_frames + (pipe.source.dropped if pipe.source is not None else 0)),
        },
        "speed_stats": {
            "max_kmh": round(float(max(speeds)), 2) if speeds else 0.0,
            "median_kmh": round(float(np.median(speeds)), 2) if speeds else 0.0,
            "mean_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        },
        "accuracy": {
            "ground_truth_mae_kmh": round(mae, 3) if mae is not None else None,
            "within_3_kmh_pct": round(within3, 2) if within3 is not None else None,
        },
        "limits": {
            "left_kmh": float(settings.left_lane_limit_kmh),
            "right_kmh": float(settings.right_lane_limit_kmh),
        },
        "performance": stage_avg,
        "recent_events": events[-12:],
        "disclaimer": DISCLAIMER,
    }


def local_summary(packet: Dict) -> Dict:
    c = packet["counts"]
    s = packet["speed_stats"]
    acc = packet["accuracy"]

    acc_text = ""
    if acc["ground_truth_mae_kmh"] is not None:
        acc_text = (
            f" Against synthetic ground truth, the mean absolute speed error is "
            f"{acc['ground_truth_mae_kmh']} km/h and {acc['within_3_kmh_pct']}% of samples "
            f"are within the configured tolerance."
        )

    summary = (
        f"The monitored window covers {packet['window_sec']} seconds and tracks "
        f"{c['tracked_vehicle_refs']} vehicle references. It recorded "
        f"{c['overspeed_events']} overspeed events, with a maximum estimated speed of "
        f"{s['max_kmh']} km/h and a median speed of {s['median_kmh']} km/h. "
        f"Calibration reprojection RMSE is {packet['calibration_rmse_px']} pixels.{acc_text} "
        f"All outputs are estimates and are not for enforcement."
    )

    findings = []
    for e in packet["recent_events"]:
        findings.append(
            {
                "event_id": e["event_id"],
                "severity": e["severity"],
                "text": (
                    f"Vehicle {e['vehicle_id']} in {e['lane']} lane estimated "
                    f"{e['speed_kmh']} km/h versus {e['limit_kmh']} km/h limit."
                ),
            }
        )

    actions = [
        "Review evidence clips for high-severity events.",
        "Verify calibration if reprojection RMSE exceeds the configured tolerance.",
        "Use per-lane limits and hysteresis before operational tuning.",
        "Keep outputs labeled as estimates, not enforcement.",
    ]

    confidence = "medium" if packet["calibration_rmse_px"] < 2.0 else "low"

    limitations = [
        "This local report uses only numbers from the evidence packet.",
        "Perspective speed estimation is sensitive to calibration quality.",
        "Occlusion, camera shake, and poor lighting increase error.",
        "No license plate recognition is performed.",
    ]

    return {
        "summary": summary,
        "key_findings": findings,
        "recommended_actions": actions,
        "confidence": confidence,
        "limitations": limitations,
        "mode": "local_fallback_no_api_key",
    }


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def flatten_numbers(obj: Any, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(f"{round(float(obj), 2):g}")
        out.add(str(int(round(float(obj)))))


def numeric_faithfulness(report: Dict, packet: Dict) -> float:
    # Check only the natural-language summary. Event IDs can contain hexadecimal
    # digits and should not be treated as numeric claims.
    text = report["summary"]
    nums = extract_numbers(text)
    if not nums:
        return 100.0

    allowed = set()
    flatten_numbers(packet, allowed)

    matched = 0
    for n in nums:
        try:
            v = float(n)
            candidates = {f"{round(v, 2):g}", str(int(round(v)))}
            if candidates & allowed:
                matched += 1
        except Exception:
            pass

    return round(100.0 * matched / max(1, len(nums)), 2)


def make_html_report(pipe: Pipeline, packet: Dict, report: Dict) -> str:
    frame = pipe.last_frame
    if frame is None:
        frame = np.zeros((pipe.settings_snapshot.height, pipe.settings_snapshot.width, 3), dtype=np.uint8)

    ok, jpg = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8") if ok else ""

    events = pipe.event_engine.events
    rows = ""
    for e in events[-50:]:
        rows += (
            "<tr>"
            f"<td>{e.get('event_id','')}</td>"
            f"<td>{e.get('vehicle_id','')}</td>"
            f"<td>{e.get('lane','')}</td>"
            f"<td>{e.get('speed_kmh','')}</td>"
            f"<td>{e.get('limit_kmh','')}</td>"
            f"<td>{e.get('severity','')}</td>"
            f"<td>{e.get('ts','')}</td>"
            "</tr>"
        )

    findings_html = "".join(
        f"<li><b>{f['severity']}</b> {f['event_id']}: {f['text']}</li>"
        for f in report["key_findings"]
    )
    actions_html = "".join(f"<li>{a}</li>" for a in report["recommended_actions"])
    limitations_html = "".join(f"<li>{a}</li>" for a in report["limitations"])

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{APP_NAME} Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h1 {{ margin-bottom: 4px; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:4px; background:#fff3cd; color:#664d03; }}
.grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin:16px 0; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:12px; }}
.card .v {{ font-size:24px; font-weight:700; }}
img {{ max-width:100%; border:1px solid #ddd; border-radius:8px; }}
table {{ border-collapse: collapse; width:100%; margin-top:12px; }}
th, td {{ border:1px solid #ddd; padding:6px 8px; font-size:13px; }}
th {{ background:#f5f5f5; text-align:left; }}
pre {{ background:#f6f8fa; padding:12px; border-radius:8px; overflow:auto; }}
</style>
</head>
<body>
<h1>{APP_NAME}</h1>
<div class="badge">{DISCLAIMER}</div>
<p><b>Run:</b> {packet['run_id']} | <b>Mode:</b> {packet['mode']} | <b>Scenario:</b> {packet['scenario']} | <b>Generated:</b> {packet['created_at']}</p>

<div class="grid">
  <div class="card"><div>Overspeed events</div><div class="v">{packet['counts']['overspeed_events']}</div></div>
  <div class="card"><div>Max speed km/h</div><div class="v">{packet['speed_stats']['max_kmh']}</div></div>
  <div class="card"><div>Calib RMSE px</div><div class="v">{packet['calibration_rmse_px']}</div></div>
  <div class="card"><div>ID switches</div><div class="v">{packet['counts']['id_switches']}</div></div>
</div>

<h2>Live frame</h2>
<img src="data:image/jpeg;base64,{img_b64}" />

<h2>Summary</h2>
<p>{report['summary']}</p>

<h2>Key findings</h2>
<ul>
{findings_html}
</ul>

<h2>Recommended actions</h2>
<ul>
{actions_html}
</ul>

<h2>Limitations</h2>
<ul>
{limitations_html}
</ul>

<h2>Recent events</h2>
<table>
<thead><tr><th>event_id</th><th>vehicle_id</th><th>lane</th><th>speed</th><th>limit</th><th>severity</th><th>ts</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>Evidence packet</h2>
<pre>{json.dumps(packet, indent=2)}</pre>

</body>
</html>
"""
    return html


# ----------------------------
# Diagnostics / self tests
# ----------------------------

def run_self_tests() -> List[Dict]:
    results = []

    def add(name: str, passed: bool, detail: str = ""):
        results.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. Geometry round trip
    try:
        src = np.array([[0, 100], [100, 100], [100, 0], [0, 0]], dtype=np.float32)
        dst = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        H, H_inv, rmse, _ = compute_homography(src, dst)
        p = project_pixel_to_metric(H, np.array([[50, 50]], dtype=np.float32))[0]
        back = project_metric_to_pixel(H_inv, p.reshape(1, 2))[0]
        err = float(np.linalg.norm(back - np.array([50, 50])))
        add("geometry_roundtrip", err < 1.0 and rmse < 1.0, f"err={err:.4f}, rmse={rmse:.4f}")
    except Exception as exc:
        add("geometry_roundtrip", False, str(exc))

    # 2. Tracker consistency
    try:
        settings = Settings(min_hits=2, track_buffer=10)
        tracker = ByteLikeTracker()
        d1 = [Detection(bbox=(100, 100, 200, 200), conf=0.9, cls="car")]
        d2 = [Detection(bbox=(102, 101, 202, 201), conf=0.88, cls="car")]
        tracker.update(d1, 0.0, settings)
        tracks = tracker.update(d2, 0.04, settings)
        ok = len(tracks) == 1 and tracks[0].hits == 2
        add("tracker_single_track", ok, f"tracks={len(tracks)}, hits={tracks[0].hits if tracks else 0}")
    except Exception as exc:
        add("tracker_single_track", False, str(exc))

    # 3. Event hysteresis and dedupe
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="vse_test_")
        settings = Settings(
            confirm_frames=1,
            min_track_frames=5,
            right_lane_limit_kmh=50.0,
            overspeed_margin_kmh=2.0,
            zone_start_m=0.0,
            zone_end_m=30.0,
            save_evidence=False,
        )
        engine = EventEngine(tmp, "test_run", "test_cal")

        tr = TrackState(id=1, cls="car", bbox=(0, 0, 20, 20), conf=0.9, last_ts=0.0)
        tr.hits = 20
        tr.ref = "abc123"
        tr.lane = "right"
        tr.speed_kmh = 80.0
        tr.speed_ci = 1.0
        tr.speed_valid = True
        tr.metric_y = 10.0
        tr.history.append((0.0, 2.0, 10.0))

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        e1 = engine.evaluate([tr], 1.0, frame, deque(), settings, "test_cal")
        e2 = engine.evaluate([tr], 1.1, frame, deque(), settings, "test_cal")

        ok = len(e1) == 1 and len(e2) == 0
        add("event_dedupe_hysteresis", ok, f"first={len(e1)}, second={len(e2)}")
        engine.close()
    except Exception as exc:
        add("event_dedupe_hysteresis", False, str(exc))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    # 4. Schema validation
    try:
        valid = {
            "event_id": "x",
            "vehicle_id": "y",
            "class": "car",
            "lane": "right",
            "speed_kmh": 70.0,
            "speed_ci": 1.0,
            "track_len_frames": 10,
            "first_ts": 1.0,
            "ts": 2.0,
            "violation": True,
            "limit_kmh": 50.0,
            "calibration_id": "c",
        }
        invalid = dict(valid)
        del invalid["vehicle_id"]
        ok = len(validate_event(valid)) == 0 and len(validate_event(invalid)) > 0
        add("event_schema_validation", ok, "")
    except Exception as exc:
        add("event_schema_validation", False, str(exc))

    # 5. Broken threshold detection
    tmp2 = None
    try:
        tmp2 = tempfile.mkdtemp(prefix="vse_test2_")
        settings = Settings(confirm_frames=1000, min_track_frames=5)
        engine = EventEngine(tmp2, "r", "c")
        tr = TrackState(id=1, cls="car", bbox=(0, 0, 20, 20), conf=0.9, last_ts=0.0)
        tr.hits = 20
        tr.ref = "abc"
        tr.lane = "right"
        tr.speed_kmh = 90.0
        tr.speed_valid = True
        tr.metric_y = 10.0
        events = engine.evaluate([tr], 1.0, np.zeros((10, 10, 3), dtype=np.uint8), deque(), settings, "c")
        ok = len(events) == 0
        add("threshold_sensitivity", ok, f"events_with_extreme_confirm={len(events)}")
        engine.close()
    except Exception as exc:
        add("threshold_sensitivity", False, str(exc))
    finally:
        if tmp2 is not None:
            shutil.rmtree(tmp2, ignore_errors=True)

    return results


# ----------------------------
# Streamlit session / app
# ----------------------------

def mark_rebuild():
    st.session_state["needs_rebuild"] = True


def rebuild_pipeline(playing: Optional[bool] = None):
    old = st.session_state.get("pipeline")
    if old is not None:
        try:
            old.close()
        except Exception:
            pass

    settings = settings_from_session()

    try:
        pipe = Pipeline(settings)
        err = None
    except Exception:
        err = traceback.format_exc()
        safe = Settings()
        safe.mode = "demo"
        safe.scenario = "free_flow"
        safe.calib_src = Settings().calib_src
        pipe = Pipeline(safe)

    st.session_state["pipeline"] = pipe
    st.session_state["pipeline_error"] = err
    st.session_state["pending_alerts"] = []
    st.session_state["approved_alerts"] = []
    st.session_state["rejected_alerts"] = []
    st.session_state["report"] = None
    st.session_state["auto_video"] = None
    st.session_state["needs_rebuild"] = False
    st.session_state["last_tick"] = time.perf_counter()
    st.session_state["accumulator"] = 0.0

    if playing is not None:
        st.session_state["playing"] = bool(playing)

    # Prefill demo so the first screen is never blank.
    try:
        current_settings = settings_from_session()

        if pipe.mode == "demo":
            n = min(PREFILL_DEMO_FRAMES, max(1, pipe.total_frames))
            for _ in range(n):
                res = pipe.process_next(current_settings)
                if res is None:
                    break

            video_path = os.path.join(pipe.run_dir, "live_demo.mp4")
            saved = pipe.save_buffer_video(video_path)
            st.session_state["auto_video"] = saved
        else:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                res = pipe.process_next(current_settings)
                if res is not None:
                    break
                time.sleep(0.05)
    except Exception:
        st.session_state["pipeline_error"] = traceback.format_exc()


def init_session():
    if st.session_state.get("_initialized", False):
        return

    base = Settings()
    for f in fields(base):
        st.session_state.setdefault(f.name, getattr(base, f.name))

    st.session_state.setdefault("_initialized", True)
    st.session_state.setdefault("playing", True)
    st.session_state.setdefault("speed_factor", 1.0)
    st.session_state.setdefault("needs_rebuild", False)
    st.session_state.setdefault("pipeline_error", None)
    st.session_state.setdefault("pending_alerts", [])
    st.session_state.setdefault("approved_alerts", [])
    st.session_state.setdefault("rejected_alerts", [])
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("auto_video", None)
    st.session_state.setdefault("last_frame", None)
    st.session_state.setdefault("last_stats", {})
    st.session_state.setdefault("diagnostics", None)
    st.session_state.setdefault("last_tick", time.perf_counter())
    st.session_state.setdefault("accumulator", 0.0)

    # Calibration-only helper keys. These avoid duplicate Streamlit widget keys.
    st.session_state.setdefault("cal_road_width_m", float(base.road_width_m))
    st.session_state.setdefault("cal_road_length_m", float(base.road_length_m))

    # Calibration point helper keys
    src = st.session_state.get("calib_src", base.calib_src)
    for i in range(4):
        st.session_state.setdefault(f"cal_x_{i}", float(src[i][0]))
        st.session_state.setdefault(f"cal_y_{i}", float(src[i][1]))

    rebuild_pipeline(True)


def add_pending_alerts(events: List[Dict]):
    if "pending_alerts" not in st.session_state:
        st.session_state.pending_alerts = []

    existing = {a.get("event_id") for a in st.session_state.pending_alerts}

    for e in events:
        eid = e.get("event_id")
        if eid in existing:
            continue
        if e.get("severity") in ("medium", "high"):
            st.session_state.pending_alerts.append(
                {
                    "event_id": eid,
                    "vehicle_id": e.get("vehicle_id"),
                    "lane": e.get("lane"),
                    "speed_kmh": e.get("speed_kmh"),
                    "limit_kmh": e.get("limit_kmh"),
                    "severity": e.get("severity"),
                    "message": (
                        f"Vehicle {e.get('vehicle_id')} in {e.get('lane')} lane "
                        f"estimated at {e.get('speed_kmh')} km/h vs {e.get('limit_kmh')} km/h."
                    ),
                }
            )


init_session()

if st.session_state.get("needs_rebuild", False):
    rebuild_pipeline(st.session_state.get("playing", True))


# ----------------------------
# Live fragment
# ----------------------------

def _live_fragment_body():
    pipe = st.session_state.get("pipeline")
    if pipe is None:
        st.info("Pipeline is initializing...")
        return

    settings = settings_from_session()

    now = time.perf_counter()
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = now
        st.session_state.accumulator = 0.0

    elapsed = max(0.0, min(0.5, now - float(st.session_state.last_tick)))
    st.session_state.last_tick = now

    playing = bool(st.session_state.get("playing", True))

    ctl1, ctl2, ctl3, ctl4 = st.columns([1, 1, 1, 5])

    if ctl1.button("Pause" if playing else "Play", key="live_play_btn", use_container_width=True):
        st.session_state.playing = not playing
        st.session_state.last_tick = time.perf_counter()
        st.session_state.accumulator = 0.0
        playing = bool(st.session_state.playing)

    if ctl2.button("Step", key="live_step_btn", use_container_width=True):
        st.session_state.playing = False
        playing = False
        res = pipe.process_next(settings)
        if res is not None:
            frame, stats, new_events = res
            st.session_state.last_frame = frame
            st.session_state.last_stats = stats
            add_pending_alerts(new_events)

    if ctl3.button("Stop", key="live_stop_btn", use_container_width=True):
        st.session_state.playing = False
        playing = False
        pipe.finished = True
        if pipe.source is not None:
            pipe.source.stop()

    ctl4.caption(
        f"Mode: {pipe.mode} | Scenario: {settings.scenario} | Backend: {pipe.backend} | "
        f"Playback speed: {float(st.session_state.get('speed_factor', 1.0)):.2f}x"
    )

    frame_ph = st.empty()

    frame = st.session_state.get("last_frame")
    stats = st.session_state.get("last_stats", {})
    new_events: List[Dict] = []

    if pipe.finished:
        st.session_state.playing = False
        playing = False
    elif playing:
        if pipe.mode == "demo":
            target = (
                elapsed
                * max(1.0, pipe.source_fps)
                * float(st.session_state.get("speed_factor", 1.0))
                + float(st.session_state.get("accumulator", 0.0))
            )
            n = int(target)
            st.session_state.accumulator = target - n

            max_catch = 5
            if n > max_catch:
                pipe.dropped_frames += n - max_catch
                n = max_catch

            for _ in range(n):
                res = pipe.process_next(settings)
                if res is None:
                    break
                frame, stats, evs = res
                new_events.extend(evs)
        else:
            max_batch = 10 if pipe.mode == "file" else 3
            for _ in range(max_batch):
                res = pipe.process_next(settings)
                if res is None:
                    break
                frame, stats, evs = res
                new_events.extend(evs)

        if new_events:
            add_pending_alerts(new_events)

        st.session_state.last_frame = frame
        st.session_state.last_stats = stats

    if frame is not None:
        frame_ph.image(to_rgb(frame), use_container_width=True)
    else:
        frame_ph.info("Waiting for first frame...")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rolling FPS", f"{stats.get('rolling_fps', 0):.1f}")
    m2.metric("Proc ms", f"{stats.get('total_ms', 0):.1f}")
    m3.metric("Tracks", stats.get("active_tracks", 0))
    m4.metric("Events", stats.get("events_total", 0))
    m5.metric("Drop", stats.get("dropped_frames", 0))
    m6.metric("RMSE px", f"{stats.get('calibration_rmse', 0):.2f}")

    st.markdown(
        f"**Frame** {stats.get('frame_index', 0)}/{stats.get('total_frames', 0)} | "
        f"**Video time** {stats.get('timestamp', 0):.2f}s | "
        f"**Real-time factor** {stats.get('real_time_factor', 0):.2f}x | "
        f"**ID switches** {stats.get('id_switches', 0)}"
    )

    total = int(stats.get("total_frames", 0) or 0)
    idx = float(stats.get("frame_index", 0) or 0)
    if total > 0:
        st.progress(min(1.0, idx / float(total)))
    else:
        st.progress(min(1.0, idx / 500.0))


if hasattr(st, "fragment"):
    try:
        live_fragment = st.fragment(run_every=LIVE_FRAGMENT_HZ)(_live_fragment_body)
    except TypeError:
        live_fragment = st.fragment(_live_fragment_body)
else:
    def live_fragment():
        _live_fragment_body()
        if st.session_state.get("playing", False):
            time.sleep(LIVE_FRAGMENT_HZ)
            st.rerun()


# ----------------------------
# Page layout
# ----------------------------

st.title(APP_NAME)
st.caption(f"{APP_VERSION} | {DISCLAIMER}")

if st.session_state.get("pipeline_error"):
    st.warning("Pipeline fallback warning: using safe demo mode after an initialization error.")
    with st.expander("Fallback traceback"):
        st.code(st.session_state["pipeline_error"])


side = st.sidebar
side.header("Controls")

side.selectbox(
    "Mode",
    options=["demo", "file", "webcam", "rtsp"],
    key="mode",
    on_change=mark_rebuild,
)

settings_preview = settings_from_session()

if settings_preview.mode == "demo":
    side.selectbox(
        "Demo scenario",
        options=["free_flow", "stop_go", "night_glare", "occlusion", "camera_shake"],
        key="scenario",
        on_change=mark_rebuild,
    )
    side.slider("Demo duration (s)", min_value=5.0, max_value=120.0, step=5.0, key="demo_duration_sec")
    side.checkbox("Night glare override", key="night_glare")
    side.checkbox("Camera shake override", key="camera_shake")
    if side.button("Apply demo settings / reset", use_container_width=True):
        rebuild_pipeline(st.session_state.get("playing", True))

elif settings_preview.mode == "file":
    uploaded = side.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="uploaded_file")
    if uploaded is not None and side.button("Load uploaded video", use_container_width=True):
        try:
            st.session_state.source_path = save_uploaded_file(uploaded)
            mark_rebuild()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save upload: {exc}")

    side.text_input("Or source path", key="source_path")
    if side.button("Use path / reset", use_container_width=True):
        mark_rebuild()
        st.rerun()

elif settings_preview.mode == "webcam":
    side.number_input("Webcam index", min_value=0, max_value=10, step=1, key="webcam_index")
    if side.button("Open webcam / reset", use_container_width=True):
        mark_rebuild()
        st.rerun()

elif settings_preview.mode == "rtsp":
    side.text_input("RTSP URL", key="rtsp_url", placeholder="rtsp://...")
    if side.button("Connect RTSP / reset", use_container_width=True):
        mark_rebuild()
        st.rerun()

side.slider("Playback speed", min_value=0.25, max_value=3.0, step=0.25, key="speed_factor")

if side.button("Reset pipeline", use_container_width=True):
    rebuild_pipeline(st.session_state.get("playing", True))

with side.expander("Performance / sampling", expanded=False):
    side.slider("Process every N frames", min_value=1, max_value=10, step=1, key="process_every_n_frames")
    side.number_input("Max frames (0 = unlimited)", min_value=0, max_value=100000, step=100, key="max_frames")
    side.slider("Resolution width", min_value=480, max_value=1920, step=80, key="width")
    side.slider("Resolution height", min_value=270, max_value=1080, step=30, key="height")
    side.slider("Source FPS hint", min_value=5.0, max_value=60.0, step=1.0, key="fps")
    side.caption("Press Reset pipeline to apply source/resolution changes.")

with side.expander("Detector", expanded=False):
    side.selectbox("Backend", options=["auto", "motion", "yolo"], key="detector_backend")
    side.text_input("YOLO model", key="yolo_model")
    side.selectbox("Device", options=["auto", "cpu", "cuda"], key="device")
    side.checkbox("FP16 / half", key="half")
    side.slider("Detection confidence", min_value=0.05, max_value=0.95, step=0.05, key="confidence_threshold")
    side.slider("NMS IoU", min_value=0.10, max_value=0.95, step=0.05, key="nms_iou")
    if not HAS_YOLO:
        side.caption("YOLO optional backend not installed. Motion detector will be used for real video.")
    side.caption("Press Reset pipeline to apply detector/backend changes.")

with side.expander("Tracker", expanded=False):
    side.slider("Track high threshold", min_value=0.10, max_value=0.95, step=0.05, key="track_high_thresh")
    side.slider("Track low threshold", min_value=0.00, max_value=0.90, step=0.05, key="track_low_thresh")
    side.slider("Match threshold", min_value=0.05, max_value=0.90, step=0.05, key="match_thresh")
    side.slider("Track buffer", min_value=5, max_value=180, step=5, key="track_buffer")
    side.slider("Min hits", min_value=1, max_value=30, step=1, key="min_hits")

with side.expander("Speed / events", expanded=False):
    side.slider("Speed window (s)", min_value=0.3, max_value=2.0, step=0.1, key="speed_window_sec")
    side.slider("Speed min samples", min_value=1, max_value=20, step=1, key="speed_min_samples")
    side.slider("Min track frames", min_value=1, max_value=60, step=1, key="min_track_frames")
    side.slider("Overspeed margin (km/h)", min_value=0.0, max_value=15.0, step=0.5, key="overspeed_margin_kmh")
    side.slider("Confirm frames", min_value=1, max_value=30, step=1, key="confirm_frames")
    side.slider("Clear frames", min_value=1, max_value=40, step=1, key="clear_frames")
    side.slider("Dedupe cooldown (s)", min_value=0.0, max_value=20.0, step=1.0, key="dedupe_cooldown_sec")
    side.number_input("Left lane limit", min_value=0.0, max_value=200.0, step=1.0, key="left_lane_limit_kmh")
    side.number_input("Right lane limit", min_value=0.0, max_value=200.0, step=1.0, key="right_lane_limit_kmh")
    side.slider("Zone start (m)", min_value=0.0, max_value=80.0, step=1.0, key="zone_start_m")
    side.slider("Zone end (m)", min_value=1.0, max_value=120.0, step=1.0, key="zone_end_m")

with side.expander("Privacy / evidence", expanded=False):
    side.checkbox("Blur plate regions", key="blur_regions")
    side.slider("Blur strength", min_value=3, max_value=51, step=2, key="blur_strength")
    side.checkbox("Save evidence", key="save_evidence")
    side.slider("Clip length (s)", min_value=1.0, max_value=10.0, step=1.0, key="clip_seconds")
    side.number_input("Retention days", min_value=1, max_value=365, step=1, key="retention_days")

with side.expander("Overlay", expanded=False):
    side.checkbox("Show overlay", key="show_overlay")
    side.checkbox("Show ground truth (demo)", key="show_ground_truth")


pipe = st.session_state.pipeline
settings = settings_from_session()

tab_live, tab_calib, tab_analytics, tab_events, tab_intel, tab_diag, tab_privacy, tab_docs = st.tabs(
    ["Live", "Calibration", "Analytics", "Events", "Intelligence", "Diagnostics", "Privacy", "Docs"]
)


# Live tab
with tab_live:
    st.subheader("Smooth live pipeline")
    st.caption(
        "This tab uses a Streamlit fragment for partial UI updates. "
        "The synthetic demo is paced by wall-clock time to reduce shaking."
    )

    live_fragment()

    st.divider()

    auto_video = st.session_state.get("auto_video")
    if auto_video and os.path.exists(auto_video):
        st.markdown("**Generated live demo video buffer**")
        st.video(auto_video)
    else:
        st.info("A live buffer video will appear after demo frames are generated.")

    b1, b2, b3 = st.columns(3)

    if b1.button("Generate live buffer video now"):
        out_path = os.path.join(pipe.run_dir, f"live_buffer_{int(time.time())}.mp4")
        saved = pipe.save_buffer_video(out_path)
        if saved:
            st.session_state.auto_video = saved
            st.rerun()
        else:
            st.warning("Video codec unavailable in this OpenCV build. Snapshots are still saved.")

    if auto_video and os.path.exists(auto_video):
        with open(auto_video, "rb") as f:
            b2.download_button(
                "Download live buffer",
                f.read(),
                file_name=os.path.basename(auto_video),
                mime="video/mp4",
                use_container_width=True,
            )

    if b3.button("Clear generated buffer"):
        pipe.frame_buffer.clear()
        st.session_state.auto_video = None
        st.success("Buffer cleared.")


# Calibration tab
with tab_calib:
    st.subheader("Perspective calibration tool")
    st.caption(
        "Edit pixel points and metric road dimensions, then apply to rebuild the homography. "
        "Zone start/end controls are in the sidebar under Speed / events."
    )

    st.image(to_rgb(pipe.calibration_preview_image()), use_container_width=True)

    cc1, cc2 = st.columns(2)
    road_width = cc1.number_input(
        "Road width (m)",
        min_value=2.0,
        max_value=40.0,
        step=0.5,
        key="cal_road_width_m",
    )
    road_length = cc2.number_input(
        "Calibration length (m)",
        min_value=5.0,
        max_value=120.0,
        step=1.0,
        key="cal_road_length_m",
    )

    st.markdown("**Pixel calibration points**")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.number_input(
                f"P{i+1} x",
                min_value=0.0,
                max_value=float(pipe.settings_snapshot.width - 1),
                step=1.0,
                key=f"cal_x_{i}",
            )
            st.number_input(
                f"P{i+1} y",
                min_value=0.0,
                max_value=float(pipe.settings_snapshot.height - 1),
                step=1.0,
                key=f"cal_y_{i}",
            )

    if st.button("Apply calibration and rebuild pipeline"):
        try:
            st.session_state.road_width_m = max(1.0, float(road_width))
            st.session_state.road_length_m = max(2.0, float(road_length))
            st.session_state.calib_src = [
                [float(st.session_state[f"cal_x_{i}"]), float(st.session_state[f"cal_y_{i}"])]
                for i in range(4)
            ]
            rebuild_pipeline(st.session_state.get("playing", True))
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid calibration: {exc}")

    st.metric("Reprojection RMSE (px)", f"{pipe.calibration.rmse:.3f}")

    err_values = np.nan_to_num(np.asarray(pipe.calibration.errors, dtype=float))
    err_df = pd.DataFrame(
        {
            "point": [f"P{i+1}" for i in range(len(err_values))],
            "error_px": err_values,
        }
    )
    st.bar_chart(err_df.set_index("point"))

    st.image(to_rgb(pipe.bird_eye_preview(settings_from_session())), use_container_width=True)

    with st.expander("Calibration JSON"):
        st.json(
            {
                "calibration_id": pipe.calibration.calibration_id,
                "src_pixels": pipe.calibration.src.tolist(),
                "dst_meters": pipe.calibration.dst.tolist(),
                "rmse_px": round(float(pipe.calibration.rmse), 4),
                "H": np.round(pipe.calibration.H, 6).tolist(),
            }
        )


# Analytics tab
with tab_analytics:
    st.subheader("Analytics")

    stats = pipe.last_stats or {}
    events = pipe.event_engine.events
    samples = pipe.speed_samples
    errors = pipe.error_samples

    vehicle_refs = {s[1] for s in samples}
    speeds = [s[3] for s in samples]

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Tracked refs", len(vehicle_refs))
    k2.metric("Total tracks", pipe.tracker.total_tracks)
    k3.metric("ID switches", pipe.tracker.id_switches)
    k4.metric("Overspeed events", sum(1 for e in events if e.get("violation")))
    k5.metric("Max speed", f"{max(speeds):.1f}" if speeds else "0.0")
    k6.metric("Median speed", f"{np.median(speeds):.1f}" if speeds else "0.0")
    k7.metric("Calib RMSE", f"{pipe.calibration.rmse:.2f}")

    if errors:
        errs = [e[2] for e in errors]
        st.metric("Ground-truth MAE (km/h)", f"{np.mean(errs):.2f}")
        st.metric("Within ±3 km/h (%)", f"{100.0 * np.mean(np.asarray(errs) <= 3.0):.1f}")

    if samples:
        sdf = pd.DataFrame(samples, columns=["ts", "vehicle", "lane", "speed_kmh", "speed_ci"])
        st.line_chart(sdf, x="ts", y="speed_kmh", color="vehicle")

        bins = pd.cut(
            sdf["speed_kmh"],
            bins=[0, 20, 40, 60, 80, 140],
            labels=["0-20", "20-40", "40-60", "60-80", "80+"],
        )
        hist = bins.value_counts().sort_index().rename_axis("speed_bin").reset_index(name="count")
        st.bar_chart(hist.set_index("speed_bin"))

        st.markdown("**Per-lane median speed**")
        st.dataframe(sdf.groupby("lane")["speed_kmh"].median().reset_index(), use_container_width=True)
    else:
        st.info("No speed samples yet.")

    if pipe.stage_history:
        st.markdown("**Stage latency trend (ms)**")
        tdf = pd.DataFrame(list(pipe.stage_history))
        keep = [c for c in ["timestamp", "detect_ms", "track_ms", "geometry_ms", "event_ms", "total_ms"] if c in tdf.columns]
        if keep:
            st.line_chart(tdf[keep], x="timestamp")

    if events:
        st.markdown("**Event table**")
        st.dataframe(pd.DataFrame(events), use_container_width=True)


# Events tab
with tab_events:
    st.subheader("Overspeed log and evidence")

    events = pipe.event_engine.events
    if not events:
        st.info("No overspeed events yet. Run the demo longer or lower the lane limit.")
    else:
        df = pd.DataFrame(events)

        sev_filter = st.multiselect(
            "Severity filter",
            options=["low", "medium", "high"],
            default=["low", "medium", "high"],
        )
        lane_filter = st.multiselect(
            "Lane filter",
            options=["left", "right", "unknown"],
            default=["left", "right", "unknown"],
        )

        if "severity" in df.columns:
            df = df[df["severity"].isin(sev_filter)]
        if "lane" in df.columns:
            df = df[df["lane"].isin(lane_filter)]

        st.dataframe(df, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.download_button(
            "Download events CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="events.csv",
            mime="text/csv",
        )

        jsonl_path = pipe.event_engine.storage.jsonl_path
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "rb") as f:
                c2.download_button(
                    "Download events JSONL",
                    f.read(),
                    file_name="events.jsonl",
                    mime="application/jsonl",
                )

        db_path = pipe.event_engine.storage.db_path
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                c3.download_button(
                    "Download SQLite DB",
                    f.read(),
                    file_name="events.sqlite",
                    mime="application/octet-stream",
                )

        selected_idx = st.selectbox(
            "Select event",
            range(len(events)),
            format_func=lambda i: f"{events[i]['event_id']} | {events[i]['vehicle_id']} | {events[i]['speed_kmh']} km/h",
        )

        ev = events[selected_idx]
        st.json(ev)

        snap_path = None
        clip_path = None

        if ev.get("evidence_snapshot"):
            snap_path = os.path.join(pipe.run_dir, ev["evidence_snapshot"])
        if ev.get("evidence_clip"):
            clip_path = os.path.join(pipe.run_dir, ev["evidence_clip"])

        ec1, ec2 = st.columns(2)

        if snap_path and os.path.exists(snap_path):
            with ec1:
                st.markdown("**Snapshot**")
                st.image(snap_path, use_container_width=True)
        else:
            with ec1:
                st.info("No snapshot file found.")

        if clip_path and os.path.exists(clip_path):
            with ec2:
                st.markdown("**Evidence clip**")
                st.video(clip_path)
        else:
            with ec2:
                st.info("No playable clip file found. Clip export depends on OpenCV codecs.")

    with st.expander("Event schema and deduplication rule"):
        st.markdown("**Required fields**")
        st.code("\n".join(REQUIRED_EVENT_FIELDS), language="text")
        st.markdown(
            "**Deduplication rule:** an event is suppressed for the same `(vehicle_ref, lane)` while the violation "
            "state is active, and for at least `dedupe_cooldown_sec` after the last emitted event. A new event "
            "requires the speed to clear the hysteresis band and then re-confirm."
        )


# Intelligence tab
with tab_intel:
    st.subheader("Local intelligence report (no API key)")
    st.caption("This report is grounded only in the evidence packet. Alerts require human approval.")

    if st.button("Generate local report"):
        packet = build_packet(pipe, settings_from_session())
        report = local_summary(packet)
        faith = numeric_faithfulness(report, packet)
        st.session_state.report = {
            "packet": packet,
            "report": report,
            "faithfulness": faith,
        }

    if st.session_state.get("report"):
        rep = st.session_state.report["report"]
        pack = st.session_state.report["packet"]
        faith = st.session_state.report.get("faithfulness", 0.0)

        st.metric("Numeric faithfulness", f"{faith:.1f}%")

        st.markdown("### Summary")
        st.write(rep["summary"])

        st.markdown("### Key findings")
        if rep["key_findings"]:
            for f in rep["key_findings"]:
                st.markdown(f"- **{f['severity']}** `{f['event_id']}`: {f['text']}")
        else:
            st.write("- No recent event findings.")

        st.markdown("### Recommended actions")
        for a in rep["recommended_actions"]:
            st.write(f"- {a}")

        st.markdown("### Confidence and limitations")
        st.write(f"Confidence: **{rep['confidence']}**")
        for lim in rep["limitations"]:
            st.write(f"- {lim}")

        html = make_html_report(pipe, pack, rep)
        st.download_button(
            "Download HTML report",
            html.encode("utf-8"),
            file_name=f"report_{pipe.run_id}.html",
            mime="text/html",
        )

        with st.expander("Evidence packet JSON"):
            st.json(pack)

        with st.expander("Report JSON"):
            st.json(rep)
    else:
        st.info("Press Generate local report to build a strict JSON summary from tracked events.")

    st.markdown("### Pending alerts")
    if not st.session_state.pending_alerts:
        st.success("No pending alerts.")
    else:
        for alert in list(st.session_state.pending_alerts):
            ac1, ac2, ac3 = st.columns([5, 1, 1])
            ac1.markdown(
                f"**{alert['severity'].upper()}** `{alert['event_id']}` — {alert['message']}"
            )

            if ac2.button("Approve", key=f"approve_{alert['event_id']}"):
                st.session_state.pending_alerts.remove(alert)
                st.session_state.approved_alerts.append(alert)
                st.rerun()

            if ac3.button("Reject", key=f"reject_{alert['event_id']}"):
                st.session_state.pending_alerts.remove(alert)
                st.session_state.rejected_alerts.append(alert)
                st.rerun()

    if st.session_state.approved_alerts:
        st.markdown("### Approved alerts")
        st.dataframe(pd.DataFrame(st.session_state.approved_alerts), use_container_width=True)

    if st.session_state.rejected_alerts:
        st.markdown("### Rejected alerts")
        st.dataframe(pd.DataFrame(st.session_state.rejected_alerts), use_container_width=True)


# Diagnostics tab
with tab_diag:
    st.subheader("Diagnostics and self-tests")

    info = {
        "app_version": APP_VERSION,
        "has_yolo": HAS_YOLO,
        "has_torch": HAS_TORCH,
        "mode": pipe.mode,
        "scenario": settings.scenario,
        "backend": pipe.backend,
        "run_id": pipe.run_id,
        "run_dir": pipe.run_dir,
        "calibration_id": pipe.calibration.calibration_id,
        "calibration_rmse_px": round(float(pipe.calibration.rmse), 4),
        "evidence_jsonl": pipe.event_engine.storage.jsonl_path,
        "evidence_sqlite": pipe.event_engine.storage.db_path,
        "streamlit_has_fragment": hasattr(st, "fragment"),
    }
    st.json(info)

    if st.button("Run self-tests"):
        st.session_state.diagnostics = run_self_tests()

    diag = st.session_state.get("diagnostics")
    if diag:
        ddf = pd.DataFrame(diag)
        st.dataframe(ddf, use_container_width=True)
        passed = int(ddf["passed"].sum())
        total = len(ddf)
        st.metric("Passed", f"{passed}/{total}")

        for row in diag:
            if not row["passed"]:
                st.error(f"{row['name']}: {row['detail']}")
    else:
        st.info("Run self-tests to validate geometry, tracker, event logic, and schema behavior.")


# Privacy tab
with tab_privacy:
    st.subheader("Privacy-by-design")

    st.markdown(
        f"""
        - Stored data: anonymized track events, calibration metadata, snapshots, and short evidence clips.
        - Not stored/read: license plates are not OCR-read; plate regions are blurred in evidence.
        - Vehicle identifiers: stored as salted hashes, not raw tracker IDs.
        - Retention: evidence auto-purge target is configurable; default is 30 days.
        - Output label: every result is marked **{DISCLAIMER}**.
        - Human oversight: alerts created by the report layer are pending until approved in the UI.
        - Opt-out: in a real deployment, signage and a request process would allow individuals to ask for deletion;
          this demo keeps all data local in a temporary directory.
        """
    )

    st.metric("Evidence directory", pipe.run_dir)

    pc1, pc2 = st.columns(2)

    if pc1.button("Purge old evidence"):
        cleanup_old_runs(get_evidence_root(), int(settings.retention_days))
        st.success("Old evidence purge completed.")

    if pc2.button("Delete current run evidence"):
        try:
            pipe.close()
            shutil.rmtree(pipe.run_dir, ignore_errors=True)
            rebuild_pipeline(False)
            st.success("Current run evidence deleted and demo reset in paused mode.")
        except Exception as exc:
            st.error(f"Could not delete evidence: {exc}")

    st.markdown("### Limitations")
    st.markdown(
        """
        - Perspective speed estimation is sensitive to calibration quality.
        - Camera shake, rolling shutter, occlusion, and poor lighting increase error.
        - The bottom-center of the detection box is assumed to approximate ground contact.
        - This demo is synthetic by default; real deployments require validated ground truth and legal review.
        """
    )


# Docs tab
with tab_docs:
    st.subheader("Architecture")

    st.code(
        """
video ingest -> frame sampler -> detector -> tracker -> geometry/homography -> event engine -> storage -> intelligence -> Streamlit
frame          frames          detections   tracks      metric tracks         events         SQLite/JSONL packet       UI

Smooth live design:
- Streamlit fragment updates only the Live panel.
- Wall-clock accumulator paces synthetic demo frames.
- Real sources use a bounded capture thread.
- Live RTSP/webcam drop oldest frames under load.
- File playback preserves frames with a larger queue.
        """,
        language="text",
    )

    st.markdown("### Named thresholds")
    st.json(
        {
            "confidence_threshold": settings.confidence_threshold,
            "nms_iou": settings.nms_iou,
            "track_high_thresh": settings.track_high_thresh,
            "track_low_thresh": settings.track_low_thresh,
            "match_thresh": settings.match_thresh,
            "track_buffer": settings.track_buffer,
            "min_hits": settings.min_hits,
            "speed_window_sec": settings.speed_window_sec,
            "speed_min_samples": settings.speed_min_samples,
            "min_track_frames": settings.min_track_frames,
            "overspeed_margin_kmh": settings.overspeed_margin_kmh,
            "confirm_frames": settings.confirm_frames,
            "clear_frames": settings.clear_frames,
            "dedupe_cooldown_sec": settings.dedupe_cooldown_sec,
            "left_lane_limit_kmh": settings.left_lane_limit_kmh,
            "right_lane_limit_kmh": settings.right_lane_limit_kmh,
            "zone_start_m": settings.zone_start_m,
            "zone_end_m": settings.zone_end_m,
            "clip_seconds": settings.clip_seconds,
            "retention_days": settings.retention_days,
        }
    )

    st.markdown("### Future upgrade path")
    st.markdown(
        """
        1. Replace motion detector with YOLOv8 / RT-DETR for production accuracy.
        2. Add TensorRT / ONNX export and half-precision inference.
        3. Add true ByteSort/BoT-SORT with appearance embeddings.
        4. Add multi-camera calibration fusion.
        5. Add radar / ground-truth synchronization for validation.
        6. Add formal enforcement policy layer only after legal review; this module remains estimate-only.
        """
    )