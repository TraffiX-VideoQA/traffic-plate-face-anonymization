# video_mosaic_plate_framewise_plus.py
# -*- coding: utf-8 -*-
import os, sys, glob, subprocess, json, argparse, signal, shutil
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- Utility Functions -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_videos(dir_path: str) -> List[Path]:
    exts = ("*.mp4","*.avi","*.mov","*.mkv","*.MP4","*.AVI","*.MOV","*.MKV")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(dir_path, e))
    return [Path(p) for p in sorted(files)]

def mosaic_region(img, x1, y1, x2, y2, block: int):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    h, w = roi.shape[:2]
    sw, sh = max(1, w//block), max(1, h//block)
    small  = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    return img

def probe_fps_with_ffprobe(video_path: str) -> float:
    """Prefer ffprobe to read real fps; return 0 on failure"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json", video_path
        ]
        out = subprocess.check_output(cmd).decode("utf-8", "ignore")
        js = json.loads(out)
        st = js["streams"][0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            val = st.get(key, "0/0")
            num, den = val.split("/")
            num, den = float(num), float(den)
            if num > 0 and den > 0:
                return num / den
    except Exception:
        pass
    return 0.0

def probe_duration_sec(video_path: str) -> float:
    """Use ffprobe to read video total duration (seconds), return -1 on failure (indicates unavailable/corrupted)"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        out = subprocess.check_output(cmd).decode("utf-8", "ignore")
        js = json.loads(out)
        dur = float(js["format"]["duration"])
        return dur if dur > 0 else -1.0
    except Exception:
        return -1.0

# ----------------- Video Writing (with .part temp file + atomic rename) -----------------
class FFMPEGWriter:
    """Keep your previous encoding settings: H.264, yuv420p, preset=fast, crf=23; write to .part, rename externally after success"""
    def __init__(self, out_part_path: str, w: int, h: int, fps: float):
        cmd = [
                "ffmpeg", "-loglevel", "error",
                "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", f"{fps:.6f}",
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",   # Optional, more friendly for streaming playback
                "-f", "mp4",                 # Key: force container to mp4
                out_part_path
            ]

        self.out_part_path = out_part_path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        return self.proc.wait()  # Return code

# ----------------- NMS / TTA / Multi-scale -----------------
def nms_class_agnostic(boxes: np.ndarray, scores: np.ndarray, iou_thr=0.55, topk: Optional[int]=None):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if topk and len(keep) >= topk:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def flip_boxes_horizontally(boxes: np.ndarray, img_w: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    x1 = boxes[:, 0].copy()
    x2 = boxes[:, 2].copy()
    boxes[:, 0] = img_w - x2
    boxes[:, 2] = img_w - x1
    return boxes

def detect_boxes_unified(
    model: YOLO,
    frame: np.ndarray,
    mode: str,
    imgsz_base: int,
    conf_th: float,
    iou_th: float,
    device,
    tta_flips: List[bool],
    ms_scales: List[int],
    fuse_iou: float,
    heavy_every_n: int,
    frame_idx: int
) -> np.ndarray:
    H, W = frame.shape[:2]

    # Whether to enable "heavy mode" (TTA/multi-scale) for this frame, >1 means interval acceleration
    heavy_on = True
    if mode in ("tta", "multiscale", "tta_multiscale") and heavy_every_n > 1:
        heavy_on = (frame_idx % heavy_every_n == 0)

    def infer_single(img, imgsz):
        rs = model.predict(img, imgsz=imgsz, conf=conf_th, iou=iou_th, device=device, verbose=False)
        all_boxes, all_scores = [], []
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            all_boxes.append(b); all_scores.append(s)
        if not all_boxes:
            return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(all_boxes, axis=0), np.concatenate(all_scores, axis=0)

    boxes_pool, scores_pool = [], []

    if mode == "baseline" or not heavy_on:
        b, sc = infer_single(frame, imgsz_base)
        if len(b):
            b[:, 0::2] = np.clip(b[:, 0::2], 0, W-1)
            b[:, 1::2] = np.clip(b[:, 1::2], 0, H-1)
            boxes_pool.append(b); scores_pool.append(sc)
    else:
        scales = ms_scales if mode in ("multiscale", "tta_multiscale") else [imgsz_base]
        flips  = tta_flips if mode in ("tta", "tta_multiscale") else [False]
        for s in scales:
            for do_flip in flips:
                if do_flip:
                    frame_in = cv2.flip(frame, 1)
                    b, sc = infer_single(frame_in, s)
                    if len(b): b = flip_boxes_horizontally(b, W)
                else:
                    b, sc = infer_single(frame, s)
                if len(b):
                    b[:, 0::2] = np.clip(b[:, 0::2], 0, W-1)
                    b[:, 1::2] = np.clip(b[:, 1::2], 0, H-1)
                    boxes_pool.append(b); scores_pool.append(sc)

    if not boxes_pool:
        return np.empty((0,4), dtype=np.int32)

    boxes_all = np.concatenate(boxes_pool, axis=0)
    scores_all = np.concatenate(scores_pool, axis=0)

    keep = nms_class_agnostic(boxes_all, scores_all, iou_thr=fuse_iou)
    boxes_keep = boxes_all[keep]
    return boxes_keep.astype(np.int32)

# ----------------- Resume from Checkpoint Helper -----------------
def is_completed(out_path: Path, min_size_kb: int, in_path: Optional[Path]=None, dur_ratio_thr: float=0.985) -> bool:
    """
    Completion conditions:
    1) Exists and size meets requirements;
    2) If in_path is given, require out to be readable by ffprobe for duration, and ≥ input duration * threshold.
       —— Unable to read duration (missing moov, etc.) is considered incomplete.
    """
    if not (out_path.exists() and out_path.stat().st_size >= (min_size_kb * 1024)):
        return False
    if in_path is None:
        return True
    in_dur  = probe_duration_sec(str(in_path))
    out_dur = probe_duration_sec(str(out_path))
    if in_dur < 0 or out_dur < 0:
        # Either side cannot get duration => considered incomplete
        return False
    return (out_dur >= in_dur * dur_ratio_thr)

def planned_outputs(videos: List[Path], out_dir: Path) -> List[Tuple[Path, Path]]:
    """Return [(video_path, final_out_path)]"""
    pairs = []
    for vp in videos:
        base = out_dir / f"{vp.stem}_redacted"
        final_mp4 = base.with_suffix(".mp4")
        pairs.append((vp, final_mp4))
    return pairs

# ----------------- Main Process (Frame-by-frame detection mosaic, no tracker) -----------------
def process_video(
    path: Path,
    model: YOLO,
    out_dir: str,
    imgsz: int,
    conf_th: float,
    iou_th: float,
    device,
    mode: str,
    tta_flips: List[bool],
    ms_scales: List[int],
    fuse_iou: float,
    mosaic_block: int,
    heavy_every_n: int,
    resume: bool,
    min_size_kb: int
):
    ensure_dir(Path(out_dir))

    # Target output path
    base = Path(out_dir) / f"{path.stem}_redacted"
    final_out = base.with_suffix(".mp4")
    part_out  = base.with_suffix(".mp4.part")

    if resume and is_completed(final_out, min_size_kb, in_path=path):
        print(f"[Skip] Output already exists (≥{min_size_kb}KB & duration check passed): {final_out.name}")
        return

    # Clean up historical .part
    if part_out.exists():
        print(f"[Info] Removing historical temp file: {part_out.name}")
        part_out.unlink(missing_ok=True)

    # FPS
    fps = probe_fps_with_ffprobe(str(path))
    if fps <= 0:
        cap0 = cv2.VideoCapture(str(path))
        fps = cap0.get(cv2.CAP_PROP_FPS) or 25.0
        cap0.release()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[Skip] Cannot open: {path.name}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = FFMPEGWriter(str(part_out), W, H, float(fps))

    # Interrupt protection during processing: close ffmpeg and clean .part on Ctrl+C
    def _sigint(sig, frame):
        print("\n[Interrupt] Signal received, safely cleaning up...")
        retcode = writer.close()
        cap.release()
        print(f"[Incomplete] Keeping temp file: {part_out.name} (ret={retcode})")
        sys.exit(130)
    old_handler = signal.signal(signal.SIGINT, _sigint)

    print(f"\n[Processing] {path.name} -> {final_out.name}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_boxes_unified(
            model, frame, mode, imgsz, conf_th, iou_th, device,
            tta_flips, ms_scales, fuse_iou, heavy_every_n, idx
        )
        for (x1, y1, x2, y2) in boxes:
            frame = mosaic_region(frame, x1, y1, x2, y2, block=mosaic_block)

        writer.write(frame)
        idx += 1

    # Cleanup: close writer and rename .part to final output
    retcode = writer.close()
    cap.release()
    signal.signal(signal.SIGINT, old_handler)

    if retcode == 0:
        # Success: atomic rename
        try:
            if final_out.exists():
                final_out.unlink()
            part_out.rename(final_out)
            print(f"[Complete] Export -> {final_out}")
        except Exception as e:
            print(f"[Warning] Rename failed, keeping temp file: {part_out}, error: {e}")
    else:
        print(f"[Failed] ffmpeg exit code {retcode}, keeping temp file: {part_out}")

def parse_args():
    ap = argparse.ArgumentParser("Framewise License Plate Mosaic (with TTA/Multi-Scale) + Resume/Skip")
    ap.add_argument("--videos_dir", type=str, default="face_included_videos", help="Input video directory")
    ap.add_argument("--out_dir", type=str, default="face_mosaic_videos_yolo", help="Output directory")
    ap.add_argument("--plate_model", type=str, default="runs/face_detect/y11s_face2/weights/best.pt", help="YOLO model (.pt)")
    ap.add_argument("--device", type=str, default="0", help="GPU ID, e.g. '0'; use 'cpu' for CPU")
    ap.add_argument("--imgsz", type=int, default=1280, help="Base inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IOU threshold (YOLO internal)")
    ap.add_argument("--mosaic_block", type=int, default=12, help="Mosaic block size")
    # Mode & TTA/Multi-scale
    ap.add_argument("--mode", type=str, default="multiscale",
                    choices=["baseline","tta","multiscale","tta_multiscale"],
                    help="Detection mode: baseline/tta/multiscale/tta_multiscale")
    ap.add_argument("--tta_flips", type=str, default="false,true",
                    help="TTA horizontal flip options, comma-separated, e.g. 'false,true' or 'false'")
    ap.add_argument("--ms_scales", type=str, default="640,960,1280",
                    help="Multi-scale imgsz list, comma-separated, e.g. '640,960' or '640,960,1280'")
    ap.add_argument("--fuse_iou", type=float, default=0.55,
                    help="IoU threshold for multi-prediction fusion (class-agnostic NMS)")
    ap.add_argument("--heavy_every_n", type=int, default=1,
                    help="Heavy mode frame interval: >1 means use TTA/multi-scale every N frames, other frames use baseline for acceleration")
    # Resume from checkpoint/skip completed
    ap.add_argument("--resume", action="store_true", default=True, help="Skip existing outputs (default enabled)")
    ap.add_argument("--overwrite", action="store_true", help="Ignore existing outputs, force rerun")
    ap.add_argument("--min_size_kb", type=int, default=100, help="Minimum size in KB to consider output valid")
    # Duration threshold for completeness determination
    ap.add_argument("--dur_ratio_thr", type=float, default=0.985, help="Output duration must be ≥ input duration × this ratio to be considered complete")
    return ap.parse_args()

def main():
    args = parse_args()

    # Parse list parameters
    tta_flips = []
    for s in args.tta_flips.split(","):
        s = s.strip().lower()
        tta_flips.append(s in ("1","true","t","yes","y"))
    ms_scales = [int(x.strip()) for x in args.ms_scales.split(",") if x.strip()]

    # Check model
    if not Path(args.plate_model).exists():
        print(f"[Error] Cannot find model: {args.plate_model}")
        sys.exit(1)

    # Device: directly pass user parameter to YOLO/predict ('cpu' or '0','1',...)
    device = args.device if args.device else None

    model = YOLO(args.plate_model)

    vids = list_videos(args.videos_dir)
    if not vids:
        print(f"[Error] No videos found in {args.videos_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    pairs = planned_outputs(vids, out_dir)
    total = len(pairs)

    # Estimated statistics (for overview output, optional)
    pre_to_skip = [vp for (vp, outp) in pairs if (not args.overwrite) and is_completed(outp, args.min_size_kb, in_path=vp, dur_ratio_thr=args.dur_ratio_thr)]
    pre_to_run  = [vp for (vp, outp) in pairs if (args.overwrite) or (not is_completed(outp, args.min_size_kb, in_path=vp, dur_ratio_thr=args.dur_ratio_thr))]

    print(f"[Info] Mode: {args.mode} | Base imgsz={args.imgsz} | Multi-scale={ms_scales if args.mode in ('multiscale','tta_multiscale') else '-'} | TTA flips={tta_flips if args.mode in ('tta','tta_multiscale') else '-'} | Fusion IoU={args.fuse_iou}")
    if args.overwrite:
        print("[Info] Overwrite mode: ON (will ignore existing outputs, full rerun)")
    else:
        print(f"[Info] Resume from checkpoint: ON (min_size_kb={args.min_size_kb}KB, dur_ratio_thr={args.dur_ratio_thr}) | Estimated skip {len(pre_to_skip)}, process {len(pre_to_run)}")

    # Unified progress output: each video shows "skip/process"
    for idx, (vp, outp) in enumerate(pairs, 1):
        if (not args.overwrite) and is_completed(outp, args.min_size_kb, in_path=vp, dur_ratio_thr=args.dur_ratio_thr):
            print(f"[Progress] {idx}/{total} Skip: {vp.name}")
            continue

        print(f"[Progress] {idx}/{total} Processing: {vp.name}")
        process_video(
            path=vp,
            model=model,
            out_dir=args.out_dir,
            imgsz=args.imgsz,
            conf_th=args.conf,
            iou_th=args.iou,
            device=device,
            mode=args.mode,
            tta_flips=tta_flips,
            ms_scales=ms_scales,
            fuse_iou=args.fuse_iou,
            mosaic_block=args.mosaic_block,
            heavy_every_n=args.heavy_every_n,
            resume=(not args.overwrite and args.resume),
            min_size_kb=args.min_size_kb
        )

if __name__ == "__main__":
    main()
