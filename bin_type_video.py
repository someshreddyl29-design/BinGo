"""
bin_type_video.py - run on a VIDEO: is the bin an OFFICIAL public bin or a LOCAL bin (and how big)?

Completely separate from verify.py / app.py. It only READS best.pt (never changes it) to find the bin,
then bin_type.pt (via bin_type.py) to decide the type, averaged over the whole video.

Put these in one folder:  best.pt  bin_type.pt  bin_type.py  bin_type_video.py
Run:
    py bin_type_video.py my_video.mp4
    py bin_type_video.py my_video.mp4 --out result.mp4 --stride 3

Output: result video (green box = official public bin, orange box = local bin, with label + score)
        and a printed summary.
"""
import argparse
import os
import sys

import cv2

import bin_type as BT

DET_PATH = os.environ.get("BINGO_MODEL", "best.pt")
DET_CONF = float(os.environ.get("BINGO_CONF", "0.25"))
HOLD = 8                    # samples a missed detection is carried forward for DRAWING only
GREEN, ORANGE, GREY = (0, 200, 0), (0, 140, 255), (160, 160, 160)

_det = None
_detector = None            # tests can replace the detector with set_detector(fn)


def set_detector(fn):
    """fn(frame_bgr) -> list of (x1, y1, x2, y2, conf) bins."""
    global _detector
    _detector = fn


def detect_bins(frame):
    if _detector is not None:
        return _detector(frame)
    global _det
    if _det is None:
        if not os.path.exists(DET_PATH):
            raise FileNotFoundError(f"{DET_PATH} not found. Put best.pt next to this file.")
        from ultralytics import YOLO
        _det = YOLO(DET_PATH)
    r = _det.predict(frame, conf=DET_CONF, verbose=False)[0]
    return [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == 0]


def analyse(video_path, stride=3):
    """Pass 1: find the bin every `stride` frames and collect its type evidence.
    Returns (summary, samples) where samples = {frame_index: box}."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    tracker = BT.BinTypeTracker()
    samples, i, total = {}, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            total += 1
            bins = detect_bins(frame)
            if bins:
                b = max(bins, key=lambda x: x[4])
                tracker.add(frame, b[:4], b[4])
                samples[i] = tuple(b[:4])
        i += 1
    cap.release()
    s = tracker.summary()
    s["bin_found_pct"] = round(100 * len(samples) / max(total, 1))
    s["frames_read"] = i
    return s, samples


def _label_text(s):
    return f"{s['label']}  score {s['score']:.2f}"


def render(video_path, out_path, summary, samples, stride=3):
    """Pass 2: write the video with the bin box and the (video-averaged) type label on every frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    colour = GREEN if summary.get("official") else (ORANGE if summary.get("type") else GREY)
    keys = sorted(samples)
    i, last_box, last_i = 0, None, -10 ** 9
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in samples:
            last_box, last_i = samples[i], i
        if last_box is not None and i - last_i <= HOLD * stride:
            x1, y1, x2, y2 = map(int, last_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            txt = _label_text(summary)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = max(y1, th + 12)
            cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 8, ty + 4), colour, -1)
            cv2.putText(frame, txt, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()


def process_video(video_path, out_path=None, stride=3):
    summary, samples = analyse(video_path, stride)
    out_path = out_path or os.path.splitext(video_path)[0] + "_bintype.mp4"
    render(video_path, out_path, summary, samples, stride)
    summary["out_path"] = out_path
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=3, help="check every Nth frame (3 is a good default)")
    a = ap.parse_args()
    s = process_video(a.video, a.out, a.stride)
    print("\n=== BIN TYPE RESULT ===")
    print("Bin found in      :", f"{s['bin_found_pct']}% of checked frames")
    print("Type              :", s["label"])
    print("Official public?  :", "YES" if s.get("official") else "no (local bin)")
    print("Bin score         :", s["score"])
    print("Class probabilities:", s["probs"])
    print("Saved video       :", s["out_path"])
    if s["frames"] == 0:
        print("\nNo bin was detected, so no type could be decided.")
