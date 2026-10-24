"""
bin_type.py - second stage for BinGo: WHAT KIND of bin is it?

best.pt (unchanged) finds the bin box. This module takes that box, classifies the crop with
bin_type.pt (trained by train_bin_type.ipynb) and returns a label + a bin score:

    official_public  -> "Official public bin"            score 1.0
    local_large      -> "Local bin (large bucket/drum)"  score ~0.9
    local_medium     -> "Local bin (medium)"             score ~0.8
    local_small      -> "Local bin (small bucket)"       score ~0.4  (0.3 if tiny in the frame)

Every one of them is still a BIN. Only official_public is called an official public bin.

Use (per sampled frame, with the box best.pt gave you):
    from bin_type import BinTypeTracker
    bt = BinTypeTracker()                 # once per video
    bt.add(frame, (x1, y1, x2, y2), det_conf)   # on every sampled frame where the bin was found
    info = bt.summary()                   # at the end
    info["label"], info["score"], info["probs"]

Quick test on a photo:
    py bin_type.py photo.jpg              # detects the bin with best.pt, then classifies it
"""
import os
import sys

import cv2
import numpy as np

TYPE_MODEL_PATH = os.environ.get("BINGO_TYPE_MODEL", "bin_type.pt")

CLASSES = ["official_public", "local_large", "local_medium", "local_small"]
CLASS_SCORE = {"official_public": 1.0, "local_large": 0.9, "local_medium": 0.8, "local_small": 0.4}
LABEL = {
    "official_public": "Official public bin",
    "local_large": "Local bin (large)",
    "local_medium": "Local bin (medium)",
    "local_small": "Local bin (small)",
}

OFFICIAL_MIN = 0.55    # official needs at least this probability, otherwise it is shown as a local bin
TINY_AREA = 0.015      # bin box smaller than 1.5% of the frame AND small class -> score capped at 0.3
TINY_SCORE = 0.3
LOCAL_MAX = 0.9        # a non-official bin never scores above this; 1.0 is reserved for official bins
PAD = 0.08             # context added around the bin box before classifying
COLOUR_BOOST = 1.15    # mild nudge when the bin has the standard green/blue/yellow municipal colours

# ------------------------------------------------------------------ model
_model = None
_predictor = None      # tests can replace the model with set_predictor(fn)


def set_predictor(fn):
    """fn(crop_bgr) -> {class_name: prob}. Used for tests / alternative models."""
    global _predictor
    _predictor = fn


def _load():
    global _model
    if _model is None:
        if not os.path.exists(TYPE_MODEL_PATH):
            raise FileNotFoundError(f"{TYPE_MODEL_PATH} not found. Train it with train_bin_type.ipynb "
                                    "and put bin_type.pt next to this file.")
        from ultralytics import YOLO
        _model = YOLO(TYPE_MODEL_PATH)
    return _model


def crop_box(frame, box, pad=PAD):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    return frame[y1:y2, x1:x2]


def predict_probs(crop):
    """{class: probability} for one bin crop."""
    if _predictor is not None:
        return _predictor(crop)
    m = _load()
    r = m.predict(crop, imgsz=224, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()
    out = {r.names[i]: float(probs[i]) for i in range(len(probs))}
    return {c: out.get(c, 0.0) for c in CLASSES}


# ------------------------------------------------------------------ cheap cues (no training needed)
def colour_cue(crop):
    """Share of strongly coloured pixels in the Indian municipal palette (green, blue, yellow).
    Ordinary buckets are often blue too, so this is only a small nudge, never a decision."""
    if crop is None or crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (s > 90) & (v > 60)
    pal = strong & (((h >= 35) & (h <= 85)) | ((h >= 95) & (h <= 130)) | ((h >= 20) & (h <= 34)))
    return float(pal.mean())


# ------------------------------------------------------------------ scoring
def score_from_probs(probs, area_frac=None, colour=0.0):
    """probs -> {type, label, score, probs}. area_frac = bin box area / frame area (0..1)."""
    p = {c: max(0.0, float(probs.get(c, 0.0))) for c in CLASSES}
    if colour >= 0.25:
        p["official_public"] *= COLOUR_BOOST
    tot = sum(p.values()) or 1.0
    p = {c: v / tot for c, v in p.items()}

    top = max(p, key=p.get)
    if top == "official_public" and p[top] < OFFICIAL_MIN:        # not sure enough: do not claim "official"
        top = max((c for c in CLASSES if c != "official_public"), key=p.get)

    score = sum(p[c] * CLASS_SCORE[c] for c in CLASSES)           # smooth: uncertainty lowers the score
    if top != "official_public":
        score = min(score, LOCAL_MAX)
    if top == "local_small" and area_frac is not None and area_frac < TINY_AREA:
        score = min(score, TINY_SCORE)
    return {"type": top, "label": LABEL[top], "score": round(score, 2),
            "official": top == "official_public", "probs": {c: round(v, 3) for c, v in p.items()}}


def classify_bin(frame, box, det_conf=1.0):
    """One frame, one box -> result dict (see score_from_probs)."""
    crop = crop_box(frame, box)
    if crop.size == 0:
        return None
    h, w = frame.shape[:2]
    area = (box[2] - box[0]) * (box[3] - box[1]) / float(w * h)
    return score_from_probs(predict_probs(crop), area, colour_cue(crop))


class BinTypeTracker:
    """Averages the bin type over many frames of a video (one frame can be fooled, the average rarely)."""

    def __init__(self, min_frames=3):
        self.min_frames = min_frames
        self.rows = []          # (probs dict, weight, area_frac, colour)

    def add(self, frame, box, det_conf=1.0):
        crop = crop_box(frame, box)
        if crop.size == 0:
            return
        h, w = frame.shape[:2]
        area = (box[2] - box[0]) * (box[3] - box[1]) / float(w * h)
        self.rows.append((predict_probs(crop), max(float(det_conf), 0.05), area, colour_cue(crop)))

    def summary(self):
        if not self.rows:
            return {"type": None, "label": "Bin (type unknown)", "score": 0.0, "official": False,
                    "probs": {}, "frames": 0}
        wts = np.array([r[1] for r in self.rows])
        avg = {c: float(np.average([r[0].get(c, 0.0) for r in self.rows], weights=wts)) for c in CLASSES}
        area = float(np.median([r[2] for r in self.rows]))
        colour = float(np.median([r[3] for r in self.rows]))
        res = score_from_probs(avg, area, colour)
        res["frames"] = len(self.rows)
        if len(self.rows) < self.min_frames:                      # too little evidence: be conservative
            res["score"] = min(res["score"], 0.8)
        return res


# ------------------------------------------------------------------ CLI test on a photo
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: py bin_type.py photo.jpg")
    from ultralytics import YOLO
    img = cv2.imread(sys.argv[1])
    det = YOLO(os.environ.get("BINGO_MODEL", "best.pt")).predict(img, conf=0.25, verbose=False)[0]
    bins = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in det.boxes if int(b.cls[0]) == 0]
    if not bins:
        sys.exit("best.pt found no bin in this photo")
    b = max(bins, key=lambda x: x[4])
    res = classify_bin(img, b[:4], b[4])
    print(res)
    x1, y1, x2, y2 = map(int, b[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
    cv2.putText(img, f"{res['label']}  {res['score']}", (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.imwrite("bin_type_result.jpg", img)
    print("saved bin_type_result.jpg")
