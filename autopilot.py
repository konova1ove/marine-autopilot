"""
Marine Autopilot HUD & Decision Engine
======================================
Single-file implementation.

Usage:
    python autopilot.py

Controls:
    Q  — quit
    C  — print current calibration points to console
    +/- — adjust projection horizon (seconds ahead)
"""

import cv2
import numpy as np
import requests
import base64
import time
import math
import os
import threading
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Roboflow ──────────────────────────────────────────────────────────────
    API_KEY        = os.getenv("ROBOFLOW_API_KEY", "")
    WORKSPACE      = os.getenv("ROBOFLOW_WORKSPACE", "")
    WORKFLOW_ID    = "boat-autopilot-assistant-1774813329814"

    # ── Camera ────────────────────────────────────────────────────────────────
    CAMERA_INDEX   = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_W        = 1280
    FRAME_H        = 720
    INFER_EVERY_N  = 2          # call Roboflow every N frames (latency trade-off)

    # ── Safe corridor (fraction of frame width) ───────────────────────────────
    CORRIDOR_LEFT  = 0.35
    CORRIDOR_RIGHT = 0.65

    # ── Speed tracker ─────────────────────────────────────────────────────────
    TRACK_BUFFER   = 10         # positions to keep per object
    PROJ_SECONDS   = 5.0        # how far ahead to project for collision check

    # ── Perspective calibration ───────────────────────────────────────────────
    # Source points: pixel positions of a known flat grid in the image.
    # Destination points: real-world coordinates (metres) of those same spots.
    # Defaults assume a camera at ~5 m height looking forward from the bow.
    # Run with --calibrate (TODO) or edit these manually to match your setup.
    PERSP_SRC = np.float32([
        [FRAME_W * 0.20, FRAME_H * 0.70],   # near-left
        [FRAME_W * 0.80, FRAME_H * 0.70],   # near-right
        [FRAME_W * 0.65, FRAME_H * 0.45],   # far-right
        [FRAME_W * 0.35, FRAME_H * 0.45],   # far-left
    ])
    PERSP_DST = np.float32([
        [0,   0],
        [50,  0],
        [50, 100],
        [0,  100],
    ])

    # ── Mock CAN-bus start values ─────────────────────────────────────────────
    MOCK_HEADING   = 87.0
    MOCK_SPEED_KN  = 6.5
    MOCK_RPM       = 1840.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — CAN-BUS TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════

class CANBusTelemetry:
    """
    Interface contract.
    Swap:   canbus = MockCANBus()
    For:    canbus = RealCANBus(channel='can0', bustype='socketcan')
    No other module needs to change.
    """
    def get_heading(self) -> float:     raise NotImplementedError
    def get_speed_knots(self) -> float: raise NotImplementedError
    def get_rpm(self) -> float:         raise NotImplementedError
    def get_all(self) -> dict:          raise NotImplementedError


class MockCANBus(CANBusTelemetry):
    """
    Simulated CAN-bus with realistic random walk.
    Runs in a background daemon thread — thread-safe reads.
    """
    def __init__(self):
        self._heading = Config.MOCK_HEADING
        self._speed   = Config.MOCK_SPEED_KN
        self._rpm     = Config.MOCK_RPM
        self._lock    = threading.Lock()
        t = threading.Thread(target=self._walk, daemon=True)
        t.start()

    def _walk(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                self._heading = (self._heading + random.gauss(0, 0.8)) % 360
                self._speed   = max(0.5, min(20.0, self._speed + random.gauss(0, 0.05)))
                self._rpm     = max(200, self._speed * 280 + random.gauss(0, 15))

    def get_heading(self) -> float:
        with self._lock: return round(self._heading, 1)

    def get_speed_knots(self) -> float:
        with self._lock: return round(self._speed, 1)

    def get_rpm(self) -> float:
        with self._lock: return round(self._rpm)

    def get_all(self) -> dict:
        with self._lock:
            return {
                "heading":  round(self._heading, 1),
                "speed_kn": round(self._speed, 1),
                "rpm":      round(self._rpm),
            }


class RealCANBus(CANBusTelemetry):
    """
    Stub for real python-can integration.
    Uncomment and install: pip install python-can
    Then replace MockCANBus() with RealCANBus(channel='can0').
    """
    def __init__(self, channel: str = "can0", bustype: str = "socketcan"):
        raise NotImplementedError(
            "RealCANBus: install python-can and implement PGN parsing for "
            "NMEA 2000 (heading PGN 127250, speed PGN 128259, RPM PGN 127488)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — SPEED TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    tracker_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2
    positions:  deque = field(default_factory=lambda: deque(maxlen=Config.TRACK_BUFFER))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=Config.TRACK_BUFFER))
    speed_knots:    float = 0.0
    velocity_px:    Tuple[float, float] = (0.0, 0.0)   # vx, vy px/s
    confidence:     float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SpeedTracker:
    """
    Maintains a rolling position buffer per tracker_id.
    Converts pixel displacement to metres via perspective transform,
    then adds own-ship speed to get absolute object speed in knots.
    """

    def __init__(self):
        self.objects: Dict[int, TrackedObject] = {}
        self._M = cv2.getPerspectiveTransform(Config.PERSP_SRC, Config.PERSP_DST)

    def _px_to_m(self, px: float, py: float) -> Tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._M)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def update(self, detections: list, own_speed_kn: float, ts: float) -> Dict[int, TrackedObject]:
        seen: Set[int] = set()

        for det in detections:
            tid = det.get("tracker_id")
            if tid is None:
                continue
            seen.add(tid)

            # Roboflow returns centre + width/height
            cx_f  = float(det.get("x", 0))
            cy_f  = float(det.get("y", 0))
            w_f   = float(det.get("width",  0))
            h_f   = float(det.get("height", 0))
            x1, y1 = int(cx_f - w_f / 2), int(cy_f - h_f / 2)
            x2, y2 = int(cx_f + w_f / 2), int(cy_f + h_f / 2)

            cls  = det.get("class", "vessel")
            conf = float(det.get("confidence", 1.0))

            if tid not in self.objects:
                self.objects[tid] = TrackedObject(
                    tracker_id=tid, class_name=cls,
                    bbox=(x1, y1, x2, y2), confidence=conf,
                )

            obj = self.objects[tid]
            obj.bbox       = (x1, y1, x2, y2)
            obj.confidence = conf
            obj.positions.append((cx_f, cy_f))
            obj.timestamps.append(ts)

            if len(obj.positions) >= 2:
                dt = obj.timestamps[-1] - obj.timestamps[0]
                if dt > 1e-4:
                    # ── pixel velocity ──
                    dpx = obj.positions[-1][0] - obj.positions[0][0]
                    dpy = obj.positions[-1][1] - obj.positions[0][1]
                    obj.velocity_px = (dpx / dt, dpy / dt)

                    # ── perspective → metres ──
                    mx0, my0 = self._px_to_m(*obj.positions[0])
                    mx1, my1 = self._px_to_m(*obj.positions[-1])
                    dist_m   = math.hypot(mx1 - mx0, my1 - my0)
                    speed_ms = dist_m / dt               # relative to camera

                    # ── absolute speed = relative + own-ship ──
                    own_ms         = own_speed_kn * 0.5144   # kn → m/s
                    speed_abs_ms   = speed_ms + own_ms
                    obj.speed_knots = round(speed_abs_ms * 1.944, 1)  # m/s → kn

        # drop objects that vanished from the frame
        for tid in list(self.objects):
            if tid not in seen:
                del self.objects[tid]

        return self.objects


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Decision:
    risk:          bool  = False
    threat_id:     Optional[int] = None
    threat_class:  str   = ""
    engine_action: str   = "MAINTAIN"
    rudder_action: str   = "HOLD"
    proj_seconds:  float = Config.PROJ_SECONDS


class DecisionEngine:
    """
    Linear projection: project each object's centre forward PROJ_SECONDS
    using current pixel velocity. Flag COLLISION RISK if the projected
    point falls inside the safe corridor and the object is approaching.

    Commands:
        ENGINE_ACTION  SLOW DOWN | MAINTAIN
        RUDDER_ACTION  PORT | STARBOARD | HOLD
    """

    def __init__(self, proj_seconds: float = Config.PROJ_SECONDS):
        self.proj_seconds = proj_seconds
        self.cx_left  = int(Config.FRAME_W * Config.CORRIDOR_LEFT)
        self.cx_right = int(Config.FRAME_W * Config.CORRIDOR_RIGHT)

    def evaluate(self, objects: Dict[int, TrackedObject]) -> Decision:
        decision = Decision(proj_seconds=self.proj_seconds)
        closest_threat_dist = float("inf")

        for tid, obj in objects.items():
            cx, cy    = obj.center
            vx, vy    = obj.velocity_px

            # projected position
            proj_x = cx + vx * self.proj_seconds
            proj_y = cy + vy * self.proj_seconds

            in_corridor  = self.cx_left < proj_x < self.cx_right
            approaching  = vy > 2.0   # px/s moving toward vessel (positive Y = down)

            if in_corridor and approaching:
                dist = abs(cx - Config.FRAME_W / 2)
                if dist < closest_threat_dist:
                    closest_threat_dist  = dist
                    decision.risk        = True
                    decision.threat_id   = tid
                    decision.threat_class = obj.class_name
                    decision.engine_action = "SLOW DOWN"
                    # steer away: if threat is on port side → go starboard
                    decision.rudder_action = "STARBOARD" if cx < Config.FRAME_W / 2 else "PORT"

        return decision


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — HUD OVERLAY  (pure OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════

class HUD:
    FONT  = cv2.FONT_HERSHEY_SIMPLEX

    # BGR colour palette
    C_SAFE     = (0, 255, 157)
    C_DANGER   = (60,  60, 255)
    C_CORRIDOR = (0, 220, 80)
    C_TELEM    = (255, 200, 0)
    C_CMD_OK   = (255, 200, 0)
    C_CMD_WARN = (0, 140, 255)
    C_WHITE    = (230, 235, 240)
    C_DIM      = (100, 120, 140)

    def __init__(self):
        self.w        = Config.FRAME_W
        self.h        = Config.FRAME_H
        self.cx_left  = int(self.w * Config.CORRIDOR_LEFT)
        self.cx_right = int(self.w * Config.CORRIDOR_RIGHT)
        self._fps_buf = deque(maxlen=30)

    # ── public entry ──────────────────────────────────────────────────────────

    def draw(self,
             frame:    np.ndarray,
             objects:  Dict[int, TrackedObject],
             decision: Decision,
             telem:    dict) -> np.ndarray:

        self._fps_buf.append(time.time())
        out = frame.copy()

        self._corridor(out)
        self._objects(out, objects, decision.threat_id)
        self._telemetry(out, telem)
        self._commands(out, decision)
        self._fps(out)
        self._crosshair(out)
        if decision.risk:
            self._alert(out, decision)

        return out

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _corridor(self, f):
        # translucent fill
        overlay = f.copy()
        pts = np.array([[self.cx_left, 0],[self.cx_right, 0],
                         [self.cx_right, self.h],[self.cx_left, self.h]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 35, 15))
        cv2.addWeighted(overlay, 0.3, f, 0.7, 0, f)

        # dashed borders
        self._dashed_vline(f, self.cx_left,  self.C_CORRIDOR)
        self._dashed_vline(f, self.cx_right, self.C_CORRIDOR)

        # label
        lbl = "SAFE CORRIDOR"
        (lw, _), _ = cv2.getTextSize(lbl, self.FONT, 0.38, 1)
        lx = (self.cx_left + self.cx_right) // 2 - lw // 2
        cv2.putText(f, lbl, (lx, 18), self.FONT, 0.38, self.C_CORRIDOR, 1, cv2.LINE_AA)

    def _dashed_vline(self, f, x, color, dash=14, gap=8):
        y = 0
        draw = True
        while y < self.h:
            end = min(y + (dash if draw else gap), self.h)
            if draw:
                cv2.line(f, (x, y), (x, end), color, 1)
            y = end
            draw = not draw

    def _objects(self, f, objects: Dict[int, TrackedObject], threat_id: Optional[int]):
        for tid, obj in objects.items():
            is_threat = (tid == threat_id)
            color = self.C_DANGER if is_threat else self.C_SAFE
            x1, y1, x2, y2 = obj.bbox

            # main box
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)

            # corner accents
            arm = 10
            for px, py, sx, sy in ((x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)):
                cv2.line(f, (px, py), (px + sx*arm, py), color, 2)
                cv2.line(f, (px, py), (px, py + sy*arm), color, 2)

            # label
            warn_tag = "  ⚠" if is_threat else ""
            label = f"{obj.class_name.upper()} #{tid}  {obj.speed_knots:.1f} kn{warn_tag}"
            (lw, lh), _ = cv2.getTextSize(label, self.FONT, 0.42, 1)
            ly = max(y1 - 6, lh + 6)
            cv2.rectangle(f, (x1, ly - lh - 4), (x1 + lw + 8, ly + 3), (0,0,0), -1)
            cv2.putText(f, label, (x1 + 4, ly - 1), self.FONT, 0.42, color, 1, cv2.LINE_AA)

            # velocity arrow
            vx, vy = obj.velocity_px
            if abs(vx) + abs(vy) > 1.0:
                ocx = int((x1 + x2) / 2)
                ocy = int((y1 + y2) / 2)
                tip_x = int(ocx + vx * 2.5)
                tip_y = int(ocy + vy * 2.5)
                cv2.arrowedLine(f, (ocx, ocy), (tip_x, tip_y), color, 1, tipLength=0.25)

    def _telemetry(self, f, telem: dict):
        rows = [
            f"HDG  {telem.get('heading', 0):.1f}°",
            f"SPD  {telem.get('speed_kn', 0):.1f} kn",
            f"RPM  {int(telem.get('rpm', 0))}",
        ]
        x, y0, row_h = 14, 52, 22
        pad = 8
        box_w = 148
        box_h = len(rows) * row_h + pad
        cv2.rectangle(f, (x-pad, y0-row_h+2), (x+box_w, y0+box_h-row_h), (0,0,0), -1)
        cv2.rectangle(f, (x-pad, y0-row_h+2), (x+box_w, y0+box_h-row_h), (40,60,80), 1)
        for i, row in enumerate(rows):
            cv2.putText(f, row, (x, y0 + i*row_h), self.FONT, 0.50,
                        self.C_TELEM, 1, cv2.LINE_AA)

    def _commands(self, f, d: Decision):
        rows = [f"ENGINE  {d.engine_action}", f"RUDDER  {d.rudder_action}"]
        color = self.C_CMD_WARN if d.risk else self.C_CMD_OK
        x  = self.w - 220
        y0 = self.h - 55
        cv2.rectangle(f, (x-8, y0-20), (x+216, y0+28), (0,0,0), -1)
        cv2.rectangle(f, (x-8, y0-20), (x+216, y0+28), (40,60,80), 1)
        for i, row in enumerate(rows):
            cv2.putText(f, row, (x, y0 + i*22), self.FONT, 0.50, color, 1, cv2.LINE_AA)

    def _alert(self, f, d: Decision):
        msg = f"!! COLLISION RISK  {d.threat_class.upper()} #{d.threat_id}"
        (tw, th), _ = cv2.getTextSize(msg, self.FONT, 0.62, 2)
        tx = (self.w - tw) // 2
        ty = 55
        # pulsing overlay
        alpha = 0.45 + 0.35 * abs(math.sin(time.time() * 5))
        bg = f.copy()
        cv2.rectangle(bg, (tx-12, ty-th-8), (tx+tw+12, ty+10), (0, 0, 160), -1)
        cv2.addWeighted(bg, alpha, f, 1-alpha, 0, f)
        cv2.putText(f, msg, (tx, ty), self.FONT, 0.62, (255,255,255), 2, cv2.LINE_AA)

    def _fps(self, f):
        if len(self._fps_buf) > 2:
            fps = (len(self._fps_buf) - 1) / (self._fps_buf[-1] - self._fps_buf[0] + 1e-9)
            cv2.putText(f, f"{fps:.0f} fps", (self.w - 78, 20),
                        self.FONT, 0.42, self.C_DIM, 1, cv2.LINE_AA)

    def _crosshair(self, f):
        cx, cy, r = self.w // 2, self.h // 2, 10
        cv2.line(f, (cx-r, cy), (cx+r, cy), self.C_DIM, 1)
        cv2.line(f, (cx, cy-r), (cx, cy+r), self.C_DIM, 1)
        cv2.circle(f, (cx, cy), 3, self.C_DIM, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — ROBOFLOW CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class RoboflowClient:
    """
    Calls the hosted Roboflow Workflow API via HTTPS.
    Returns the 'predictions' list from tracked_detections.
    Each item: {tracker_id, x, y, width, height, class, confidence, ...}
    """

    _WORKFLOW_URL = (
        "https://api.roboflow.com/{workspace}/workflows/{workflow_id}/run"
    )

    def __init__(self):
        self._url = self._WORKFLOW_URL.format(
            workspace=Config.WORKSPACE,
            workflow_id=Config.WORKFLOW_ID,
        )
        self._session = requests.Session()
        if not Config.API_KEY:
            print("[WARN] ROBOFLOW_API_KEY not set — inference calls will fail.")
        if not Config.WORKSPACE:
            print("[WARN] ROBOFLOW_WORKSPACE not set — check your .env file.")

    def infer(self, frame: np.ndarray) -> list:
        """Encode frame as JPEG base64, POST to Roboflow, return detections."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()
        try:
            resp = self._session.post(
                self._url,
                params={"api_key": Config.API_KEY},
                json={"inputs": {"image": {"type": "base64", "value": b64}}},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            outputs = data.get("outputs", [{}])
            tracked = outputs[0].get("tracked_detections", {})
            return tracked.get("predictions", [])
        except requests.exceptions.Timeout:
            print("[Roboflow] timeout — skipping frame")
        except requests.exceptions.HTTPError as e:
            print(f"[Roboflow] HTTP {e.response.status_code}: {e.response.text[:120]}")
        except Exception as e:
            print(f"[Roboflow] error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("⚓  Marine Autopilot HUD starting …")
    print(f"    Workflow : {Config.WORKFLOW_ID}")
    print(f"    Workspace: {Config.WORKSPACE or '(not set)'}")
    print(f"    Camera   : index {Config.CAMERA_INDEX}")
    print("    Controls : Q = quit  |  +/- = projection horizon  |  C = show calibration\n")

    canbus   = MockCANBus()          # ← swap for RealCANBus(channel='can0')
    tracker  = SpeedTracker()
    engine   = DecisionEngine()
    hud      = HUD()
    rf       = RoboflowClient()

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_H)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in .env")
        return

    frame_n       = 0
    last_objects  : Dict[int, TrackedObject] = {}
    last_decision : Decision = Decision()

    while True:
        ret, raw = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — camera disconnected?")
            break

        frame  = cv2.resize(raw, (Config.FRAME_W, Config.FRAME_H))
        telem  = canbus.get_all()
        ts     = time.time()

        # Only call Roboflow every N frames to balance latency vs. freshness
        if frame_n % Config.INFER_EVERY_N == 0:
            detections    = rf.infer(frame)
            last_objects  = tracker.update(detections, telem["speed_kn"], ts)
            last_decision = engine.evaluate(last_objects)

        frame_n += 1

        output = hud.draw(frame, last_objects, last_decision, telem)
        cv2.imshow("Marine Autopilot HUD  [Q = quit]", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            engine.proj_seconds = min(engine.proj_seconds + 1, 30)
            print(f"[HUD] Projection horizon: {engine.proj_seconds:.0f}s")
        elif key == ord('-'):
            engine.proj_seconds = max(engine.proj_seconds - 1, 1)
            print(f"[HUD] Projection horizon: {engine.proj_seconds:.0f}s")
        elif key == ord('c'):
            print("[Calibration] PERSP_SRC (pixel):")
            print(Config.PERSP_SRC)
            print("[Calibration] PERSP_DST (metres):")
            print(Config.PERSP_DST)

    cap.release()
    cv2.destroyAllWindows()
    print("\n⚓  Autopilot stopped.")


if __name__ == "__main__":
    main()
