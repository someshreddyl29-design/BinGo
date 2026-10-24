"""
run_all.py - run BOTH on one video:
    1. verify.py          -> was the object really thrown INTO the bin?  (status + score)
    2. bin_type_video.py  -> is the bin an OFFICIAL public bin or a LOCAL bin? (label + score)
and merge them into ONE result video and ONE final score.

verify.py, bin_type.py and bin_type_video.py are not modified.

Put in one folder: best.pt  bin_type.pt  verify.py  bin_type.py  bin_type_video.py  run_all.py
Run:
    py run_all.py "D:\\bingo\\test_clips\\main.mp4"
    py run_all.py "D:\\bingo\\test_clips\\main.mp4" --no-open --stride 3

Final score = W_DISPOSAL * disposal score + W_BIN * bin score        (tune the numbers below)
    - If verify.py finds NO BIN, the bin-type step is not started at all (saves time, nothing to classify).
    - If verify.py REJECTS the video (no bin / no object / never inside / came back out) the final
      score stays 0 and it stays rejected: a nice bin cannot rescue a throw that did not happen.
    - Approved / review thresholds are read from your verify.py (APPROVE, REVIEW).
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2

import bin_type_video as BV
import verify as V

W_DISPOSAL = 0.80
W_BIN = 0.20
APPROVE = getattr(V, "APPROVE", 0.80)      # same thresholds as YOUR verify.py
REVIEW = getattr(V, "REVIEW", 0.55)

GREEN, AMBER, RED, ORANGE = (0, 200, 0), (0, 190, 255), (0, 0, 220), (0, 140, 255)


def evidence(d):
    """Disposal evidence as a dict, for either version of verify.py (new one has 'parts', the original has flat keys)."""
    if isinstance(d.get("parts"), dict):
        return d["parts"]
    skip = ("status", "code", "conf", "message", "out_path")
    return {k: v for k, v in d.items() if k not in skip}


def combine(res, bt):
    """res = verify.process_video result, bt = bin type summary -> (status, final_score)."""
    if res["status"] == "rejected" or res["conf"] <= 0 or bt is None:
        return "rejected", 0.0
    final = round(W_DISPOSAL * res["conf"] + W_BIN * bt["score"], 3)
    status = "approved" if final >= APPROVE else ("review" if final >= REVIEW else "rejected")
    # verify.py's own decision can only be confirmed or lowered here, never raised
    order = {"rejected": 0, "review": 1, "approved": 2}
    if order[status] > order[res["status"]]:
        status = res["status"]
    return status, final


def add_bin_banner(video_in, video_out, bt):
    """Write the bin-type label as a banner at the TOP of verify's result video."""
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = video_out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    col = GREEN if bt.get("official") else (ORANGE if bt.get("type") else (150, 150, 150))
    txt = f"BIN: {bt['label']}  (bin score {bt['score']:.2f})"
    scale = min(0.8, w / (len(txt) * 17.0))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.rectangle(f, (0, 0), (w, 34), col, -1)
        cv2.putText(f, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(f)
    cap.release()
    vw.release()
    if shutil.which("ffmpeg"):                          # H.264 so browsers can play it
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_out]
        if subprocess.run(cmd).returncode == 0:
            os.remove(tmp)
            return video_out
    os.replace(tmp, video_out)
    return video_out


def run_all(video, out_path=None, stride=3, bin_text=None):
    base = os.path.splitext(video)[0]
    out_path = out_path or base + "_final.mp4"
    print("1/2  disposal check (verify.py) ...")
    res = V.process_video(video, out_path=base + "_verify_tmp.mp4", bin_text=bin_text)
    if res["code"] == "NO_BIN":                          # no bin in verify -> bin type must not even start
        print("2/2  no bin found by verify.py -> bin type check skipped")
        bt = None
        status, final = "rejected", 0.0
        shutil.move(res["out_path"], out_path)
    else:
        print("2/2  bin type (best.pt + bin_type.pt) ...")
        bt, _ = BV.analyse(video, stride)
        status, final = combine(res, bt)
        add_bin_banner(res["out_path"], out_path, bt)
        try:
            os.remove(res["out_path"])
        except OSError:
            pass
    return dict(status=status, final=final, disposal=res, bin=bt, out_path=out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--bin", default=None, help="fixed bin box in percent x1,y1,x2,y2 (fallback for verify.py)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    r = run_all(a.video, a.out, a.stride, a.bin)
    d, b = r["disposal"], r["bin"]
    print("\n================ BINGO RESULT ================")
    print(f"FINAL           : {r['status'].upper()}   score {r['final']}")
    print(f"Disposal check  : {d['status'].upper()}  score {d['conf']}  ({d['message']})")
    if b is None:
        print("Bin type        : NOT CHECKED (no bin was found in the video)")
    else:
        print(f"Bin type        : {b['label']}  bin score {b['score']}  "
              f"({'OFFICIAL public bin' if b.get('official') else 'local bin, not official'})")
        print(f"Bin probabilities: {b['probs']}")
    print(f"Disposal evidence: {evidence(d)}")
    print(f"Saved video     : {r['out_path']}")
    if not a.no_open and hasattr(os, "startfile"):
        os.startfile(r["out_path"])
