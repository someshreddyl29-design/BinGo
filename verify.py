"""
BinGo verify.py - did an object that was OUTSIDE the bin at the start end up IN the bin by the end?

No waste class is used. The trained model (best.pt) is only used to find the BIN.
"Any object" is found by motion: whatever moves in the frame (camera shake compensated) and is not
the bin becomes a tracked object with an id and a trail. We then compare start and end:
  start : a moving object is seen outside the bin box
  entry : the same object overlaps the bin (or vanishes at the bin opening)
  end   : the object is not back outside, and the bin area looks different from the first frame

Usage (laptop, same folder as best.pt):
    py verify.py my_video.mp4              # processes, prints verdict, opens the result
    py verify.py my_video.mp4 --no-open
    py verify.py my_video.mp4 --bin 0.25,0.45,0.80,0.95   # fixed bin box (fractions x1,y1,x2,y2)

Environment overrides (optional): BINGO_MODEL, BINGO_CONF, BINGO_STRIDE, BINGO_BIN
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ------------------------------------------------------------------ settings (tune on your test clips)
MODEL_PATH = os.environ.get("BINGO_MODEL", "best.pt")
CONF = float(os.environ.get("BINGO_CONF", "0.25"))        # bin detector confidence (same as the old version)
MAX_BINS = int(os.environ.get("BINGO_MAX_BINS", "1"))     # keep only the N most confident bin boxes per frame
STRIDE = int(os.environ.get("BINGO_STRIDE", "2"))         # motion is analysed on every Nth frame
FIXED_BIN = os.environ.get("BINGO_BIN", "")               # "x1,y1,x2,y2" as fractions 0-1 (fallback)
MAX_SECONDS = 30
MAX_WIDTH = 720
BIN_CLASS = 0

MOTION_THRESH = 22        # pixel change (0-255) that counts as motion
MIN_BLOB = 0.0004         # object size limits, as a fraction of the frame area
MAX_BLOB = 0.12           # bigger than this = person / body, not an object
SHAKE_FRAC = 0.30         # if more than this share of the frame changed, skip it (camera shake)
LINK_DIST = 0.18          # max jump between two samples to be the same object (fraction of width)
GAP = 5                   # samples an object may disappear for and still be the same track
MIN_TRACK = 3             # a track needs this many samples to count as an object (kills flicker)
OUT_T = 0.20              # object overlap with bin below this  -> outside the bin
IN_T = 0.50               # object overlap with bin above this  -> inside the bin
VANISH_T = 0.35           # object last seen with at least this overlap and then gone -> went in
ROI_CHANGE = 10.0         # mean pixel change of the bin area between first and last frame
APPROVE, REVIEW = 0.75, 0.50

MESSAGES = {
    "OK": "An object was outside the bin at the start and is in the bin at the end.",
    "NO_BIN": "The bin was not found for enough of the video (try BINGO_BIN or retrain).",
    "NO_OBJECT": "No moving object was seen outside the bin at the start.",
    "NO_ENTRY": "The object never went into the bin.",
    "TAKEN_BACK": "The object went into the bin but came back out.",
    "UNCERTAIN": "Probably in the bin, but the evidence is weak.",
    "LOW_CONFIDENCE": "Could not confirm the object ended up in the bin.",
}

_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found. Put best.pt next to verify.py.")
        _model = YOLO(MODEL_PATH)
    return _model


def parse_bin(text):
    """'0.25,0.45,0.8,0.95' (fractions) -> tuple, or None."""
    try:
        v = [float(x) for x in str(text).replace("%", "").split(",")]
        if len(v) == 4:
            if max(v) > 1.5:
                v = [x / 100.0 for x in v]          # allow percentages
            if v[2] > v[0] and v[3] > v[1]:
                return tuple(v)
    except ValueError:
        pass
    return None


# ------------------------------------------------------------------ bin
def _bin_boxes(frame):
    """Bin boxes (x1,y1,x2,y2,conf) from the trained model. Replace in tests."""
    r = get_model().predict(frame, conf=CONF, imgsz=640, verbose=False)[0]
    bins = [(*[float(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
            for b in r.boxes if int(b.cls[0]) == BIN_CLASS]
    return keep_top_bins(bins)


def keep_top_bins(bins):
    """The model can output several overlapping boxes on one bin. Keep only the best one(s)."""
    return sorted(bins, key=lambda x: -x[4])[:MAX_BINS]


def median_bin(dets):
    """dets: list of (x1,y1,x2,y2,conf), one per frame where a bin was found. -> median box or None."""
    if not dets:
        return None
    a = np.array([d[:4] for d in dets])
    return tuple(float(v) for v in np.median(a, axis=0))


# ------------------------------------------------------------------ motion
def _fit(frame):
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
    return frame


def _gray(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)


def motion_blobs(prev_g, g):
    """Moving regions between two gray frames, camera translation compensated.
    -> list of (x1,y1,x2,y2) or None when the whole frame changed (shake / light change)."""
    h, w = g.shape
    (dx, dy), _ = cv2.phaseCorrelate(np.float32(prev_g), np.float32(g))
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        prev_g = cv2.warpAffine(prev_g, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
    diff = cv2.absdiff(prev_g, g)
    mask = (diff > MOTION_THRESH).astype(np.uint8) * 255
    if mask.mean() / 255.0 > SHAKE_FRAC:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        frac = (bw * bh) / float(w * h)
        if MIN_BLOB <= frac <= MAX_BLOB:
            out.append((float(x), float(y), float(x + bw), float(y + bh)))
    return out


class Tracker:
    """Links moving blobs from sample to sample (nearest centre) into objects with ids."""

    def __init__(self, width):
        self.tracks = {}          # id -> {"last": (cx,cy), "k": last sample, "pts": [(k, box)]}
        self.next_id = 1
        self.maxd = LINK_DIST * width

    def update(self, k, blobs):
        ids, used = [], set()
        for b in blobs:
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            best, bd = None, self.maxd
            for tid, t in self.tracks.items():
                if tid in used or k - t["k"] > GAP:
                    continue
                d = ((c[0] - t["last"][0]) ** 2 + (c[1] - t["last"][1]) ** 2) ** 0.5
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"last": c, "k": k, "pts": []}
            t = self.tracks[best]
            t["last"], t["k"] = c, k
            t["pts"].append((k, b))
            used.add(best)
            ids.append(best)
        return ids


# ------------------------------------------------------------------ pass 1
def analyse_video(path, bin_fixed=None):
    """Read the video once. Returns dict with per-frame blobs, bin box, fps, size, roi frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bin_dets, frames_blobs, prev_g, tracker, size = [], [], None, None, None
    first_gray = last_gray = None
    i = k = bin_seen_frames = bin_checked = 0
    per_frame, bin_frames, cur_bin = [], [], None   # per_frame: sample index; bin_frames: bin seen on that frame
    while i < fps * MAX_SECONDS:
        ok, f = cap.read()
        if not ok:
            break
        f = _fit(f)
        if size is None:
            size = (f.shape[1], f.shape[0])
            tracker = Tracker(size[0])
        if bin_fixed is None and i % STRIDE == 0:
            bin_checked += 1
            dets = _bin_boxes(f)
            cur_bin = max(dets, key=lambda d: d[4])[:4] if dets else None     # None = not identified now
            if dets:
                bin_dets.append(max(dets, key=lambda d: d[4]))
        if i % STRIDE == 0:
            g = _gray(f)
            if first_gray is None:
                first_gray = g
            last_gray = g
            blobs = [] if prev_g is None else motion_blobs(prev_g, g)
            blobs = blobs if blobs is not None else []
            ids = tracker.update(k, blobs)
            frames_blobs.append(list(zip(ids, blobs)))
            prev_g = g
            k += 1
        per_frame.append(k - 1 if k else 0)
        bin_frames.append(cur_bin)
        i += 1
    cap.release()
    if not per_frame:
        raise ValueError("The video has no readable frames.")
    W, H = size
    if bin_fixed is not None:
        box = (bin_fixed[0] * W, bin_fixed[1] * H, bin_fixed[2] * W, bin_fixed[3] * H)
        presence = 1.0
        bin_frames = [box] * len(per_frame)
    else:
        box = median_bin(bin_dets)
        presence = len(bin_dets) / max(bin_checked, 1)
    return dict(blobs=frames_blobs, tracks=tracker.tracks, bin=box, presence=presence, fps=fps,
                size=size, per_frame=per_frame, bin_frames=bin_frames, first_gray=first_gray, last_gray=last_gray)


# ------------------------------------------------------------------ rules
def inside_ratio(item, box):
    """Share of the object's area that lies inside the bin box."""
    ix1, iy1 = max(item[0], box[0]), max(item[1], box[1])
    ix2, iy2 = min(item[2], box[2]), min(item[3], box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max((item[2] - item[0]) * (item[3] - item[1]), 1e-6)


def roi_change(first_gray, last_gray, box):
    """How different the bin area looks at the end compared to the start (mean pixel change)."""
    if first_gray is None or last_gray is None or box is None:
        return 0.0
    h, w = first_gray.shape
    x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3])))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    a = cv2.resize(first_gray[y1:y2, x1:x2], (64, 64))
    b = cv2.resize(last_gray[y1:y2, x1:x2], (64, 64))
    return float(np.mean(cv2.absdiff(a, b)))


def judge(data, det_fps):
    """-> (status, code, confidence, info). info has the chosen object track and start/end samples."""
    n = len(data["blobs"])
    need = max(2, int(round(0.4 * det_fps)))                       # ~0.4 s of samples
    info = dict(track=None, start=None, end=None, n=n, roi=0.0)
    if data["bin"] is None or data["presence"] < 0.30:
        return "rejected", "NO_BIN", 0.0, info
    box = data["bin"]

    objs = {tid: t["pts"] for tid, t in data["tracks"].items() if len(t["pts"]) >= MIN_TRACK}
    started, entered = [], []
    for tid, pts in objs.items():
        ratios = [inside_ratio(b, box) for _, b in pts]
        out_idx = [j for j, r in enumerate(ratios) if r < OUT_T]
        if not out_idx:
            continue
        started.append(tid)
        j0 = out_idx[0]
        inside_after = sum(1 for r in ratios[j0:] if r >= IN_T)
        vanished = ratios[-1] >= VANISH_T and (n - 1 - pts[-1][0]) >= need
        if inside_after >= 2 or vanished:
            entered.append((pts[0][0], tid, inside_after, vanished))
    if not started:
        return "rejected", "NO_OBJECT", 0.0, info
    if not entered:
        info["track"] = started[0]
        return "rejected", "NO_ENTRY", 0.0, info

    entered.sort()
    _, tid, inside_after, vanished = entered[0]
    pts = objs[tid]
    ratios = [inside_ratio(b, box) for _, b in pts]
    info.update(track=tid, start=pts[0][0], end=pts[-1][0])
    taken_back = ratios[-1] < OUT_T and inside_after >= 2        # last seen outside again
    if taken_back:
        return "rejected", "TAKEN_BACK", 0.0, info

    A = min(len(pts) / (2.0 * MIN_TRACK), 1.0)                      # object followed for long enough
    B = 1.0 if inside_after >= 2 else 0.8                           # overlapped the bin / vanished at it
    rc = roi_change(data["first_gray"], data["last_gray"], box)
    info["roi"] = rc
    R = min(rc / ROI_CHANGE, 1.0)                                   # bin area changed start -> end
    stab = min(max((data["presence"] - 0.3) / 0.3, 0.0), 1.0)
    conf = 0.25 * A + 0.35 * B + 0.20 + 0.05 * stab + 0.15 * R
    if conf >= APPROVE:
        return "approved", "OK", conf, info
    if conf >= REVIEW:
        return "review", "UNCERTAIN", conf, info
    return "rejected", "LOW_CONFIDENCE", conf, info


# ------------------------------------------------------------------ drawing
GREEN, ORANGE, RED, AMBER, WHITE, GREY = (60, 190, 60), (0, 140, 255), (50, 50, 220), (0, 190, 255), (255, 255, 255), (90, 90, 90)


def _label(img, text, org, color, scale=0.55):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA)


def object_state(box, b):
    r = inside_ratio(b, box)
    return "inside bin" if r >= IN_T else ("entering" if r >= OUT_T else "outside")


def draw_frame(frame, box, show_verdict, verdict):
    """Green bin box only on frames where the detector identified the bin; verdict banner in the last
    2 seconds. No object boxes or trails."""
    h, w = frame.shape[:2]
    if box is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), GREEN, 3)
        _label(frame, "bin", (box[0], max(box[1], 22)), GREEN)
    if show_verdict:
        status, code, conf = verdict
        col = {"approved": GREEN, "review": AMBER, "rejected": RED}[status]
        text = f"{status.upper()}  ({conf:.2f})" if status != "rejected" else f"REJECTED: {code}"
        cv2.rectangle(frame, (0, h // 2 - 35), (w, h // 2 + 35), col, -1)
        cv2.putText(frame, text, (12, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    min(1.0, w / (len(text) * 18.0)), WHITE, 2, cv2.LINE_AA)
    return frame


def render_video(path, data, valid, info, verdict, out_path):
    tmp_dir = tempfile.mkdtemp(prefix="bingo_")
    raw = os.path.join(tmp_dir, "raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), data["fps"], data["size"])
    cap = cv2.VideoCapture(path)
    n = len(data["per_frame"])
    verdict_from = n - int(data["fps"] * 2.0)
    for i, k in enumerate(data["per_frame"]):
        ok, f = cap.read()
        if not ok:
            break
        writer.write(draw_frame(_fit(f), data["bin_frames"][i], i >= verdict_from, verdict))
    cap.release()
    writer.release()
    if shutil.which("ffmpeg"):                      # re-encode to H.264 so browsers can play it
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        if subprocess.run(cmd).returncode == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return out_path
    shutil.copy(raw, out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path


# ------------------------------------------------------------------ public API
def process_video(path, out_path=None, bin_text=None):
    fixed = parse_bin(bin_text) if bin_text else parse_bin(FIXED_BIN)
    data = analyse_video(path, bin_fixed=fixed)
    det_fps = data["fps"] / STRIDE
    status, code, conf, info = judge(data, det_fps)
    valid = {tid for tid, t in data["tracks"].items() if len(t["pts"]) >= MIN_TRACK}
    if out_path is None:
        out_path = os.path.join(tempfile.mkdtemp(prefix="bingo_out_"), "result.mp4")
    render_video(path, data, valid, info, (status, code, conf), out_path)
    sec = lambda k: None if k is None else round(k * STRIDE / data["fps"], 1)
    return dict(status=status, code=code, conf=round(conf, 3), message=MESSAGES[code], out_path=out_path,
                object_id=info["track"], start_s=sec(info["start"]), end_s=sec(info["end"]),
                bin_presence=round(data["presence"], 2), bin_changed=round(info["roi"], 1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: py verify.py video.mp4 [--no-open] [--bin x1,y1,x2,y2]")
    bt = sys.argv[sys.argv.index("--bin") + 1] if "--bin" in sys.argv else None
    res = process_video(sys.argv[1], out_path=os.path.splitext(sys.argv[1])[0] + "_result.mp4", bin_text=bt)
    print(f"{res['status'].upper()}  code={res['code']}  confidence={res['conf']}")
    print(res["message"])
    print({k: v for k, v in res.items() if k not in ("status", "code", "conf", "message", "out_path")})
    print("saved:", res["out_path"])
    if "--no-open" not in sys.argv and hasattr(os, "startfile"):
        os.startfile(res["out_path"])
