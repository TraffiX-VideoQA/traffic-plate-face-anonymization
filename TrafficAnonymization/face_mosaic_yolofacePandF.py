# -*- coding: utf-8 -*-
# video_mosaic_face_framewise_hybrid.py
# Function: Hybrid mode (full-frame face ∪ person ROI face) → NMS deduplication → frame-by-frame mosaic
# Features: No smoothing, no tracking, no TTA; structure follows your license plate script

import os, sys, glob, subprocess, json, argparse, signal
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

def clamp_tuple(box, W, H):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def expand_box_ratio(box, ratio, W, H):
    x1, y1, x2, y2 = box
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    bw, bh = (x2-x1), (y2-y1)
    bw2, bh2 = bw*(1.0+ratio), bh*(1.0+ratio)
    nx1 = max(0, int(cx - bw2*0.5)); ny1 = max(0, int(cy - bh2*0.5))
    nx2 = min(W, int(cx + bw2*0.5)); ny2 = min(H, int(cy + bh2*0.5))
    return (nx1, ny1, nx2, ny2)

def ensure_even(frame):
    H, W = frame.shape[:2]
    pr, pb = W % 2, H % 2
    if pr or pb:
        frame = cv2.copyMakeBorder(frame, 0, pb, 0, pr, cv2.BORDER_REPLICATE)
    return frame

def probe_fps_with_ffprobe(video_path: str) -> float:
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

# ----------------- FFmpeg Video Writing -----------------
class FFMPEGWriter:
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
            "-movflags", "+faststart",
            "-f", "mp4",
            out_part_path
        ]
        self.out_part_path = out_part_path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())
    def close(self):
        try:
            if self.proc.stdin: self.proc.stdin.close()
        except Exception:
            pass
        return self.proc.wait()

# ----------------- Detection Modules -----------------
def person_detect(model: YOLO, frame_bgr, imgsz, conf, iou, device, min_person_px: int, expand_ratio: float):
    H, W = frame_bgr.shape[:2]
    rs = model.predict(frame_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    boxes = []
    if rs and len(rs) and rs[0].boxes is not None and len(rs[0].boxes):
        b = rs[0].boxes.xyxy.cpu().numpy()
        c = rs[0].boxes.cls.cpu().numpy() if rs[0].boxes.cls is not None else np.zeros(len(b))
        for (x1,y1,x2,y2), cls_id in zip(b, c):
            if int(cls_id) != 0:  # COCO: person=0
                continue
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            if (x2-x1) < min_person_px or (y2-y1) < min_person_px:
                continue
            boxes.append(expand_box_ratio((x1,y1,x2,y2), expand_ratio, W, H))
    return boxes

def face_detect_in_rois(model: YOLO, frame_bgr, rois, imgsz, conf, iou, device, upscale, face_expand, W, H):
    det_faces = []
    if not rois:
        return det_faces
    sf = max(1.0, upscale)
    crops, meta = [], []
    for (px1,py1,px2,py2) in rois:
        roi = frame_bgr[py1:py2, px1:px2]
        if roi.size == 0: continue
        if sf > 1.0:
            roi_in = cv2.resize(roi, (int((px2-px1)*sf), int((py2-py1)*sf)), interpolation=cv2.INTER_CUBIC)
        else:
            roi_in = roi
        crops.append(roi_in)
        meta.append((px1,py1,sf))
    if not crops:
        return det_faces
    rs = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    for r, (offx,offy,sf_i) in zip(rs, meta):
        if r.boxes is None or len(r.boxes)==0: continue
        b = r.boxes.xyxy.cpu().numpy()
        for x1,y1,x2,y2 in b:
            if sf_i > 1.0:
                x1 /= sf_i; y1 /= sf_i; x2 /= sf_i; y2 /= sf_i
            X1 = int(round(x1 + offx)); Y1 = int(round(y1 + offy))
            X2 = int(round(x2 + offx)); Y2 = int(round(y2 + offy))
            c = clamp_tuple((X1,Y1,X2,Y2), W, H)
            if c is None: continue
            ex = expand_box_ratio(c, face_expand, W, H)
            c2 = clamp_tuple(ex, W, H)
            if c2 is not None:
                det_faces.append(c2)
    return det_faces

# Class-agnostic NMS for hybrid mode deduplication
def nms_union_xyxy(boxes: List[Tuple[int,int,int,int]], iou_thr=0.5) -> List[Tuple[int,int,int,int]]:
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    areas = (x2-x1) * (y2-y1)
    order = np.argsort(areas)[::-1]  # Large boxes first, more conservative
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return [tuple(map(int, b[i])) for i in keep]

# ----------------- Main Process (Frame-by-frame, Hybrid Optional) -----------------
def process_video(
    path: Path,
    person_model: YOLO,
    face_model: YOLO,
    out_dir: str,
    device,
    imgsz: int,
    person_conf: float, person_iou: float,
    min_person_px: int, person_expand: float,
    face_conf: float, face_iou: float,
    face_expand: float, upscale: float,
    mosaic_block: int,
    hybrid: bool, nms_union_iou: float,
    fallback_fullframe: bool,
    resume: bool, min_size_kb: int, dur_ratio_thr: float
):
    ensure_dir(Path(out_dir))

    base = Path(out_dir) / f"{path.stem}_redacted"
    final_out = base.with_suffix(".mp4")
    part_out  = base.with_suffix(".mp4.part")

    if resume and is_completed(final_out, min_size_kb, in_path=path, dur_ratio_thr=dur_ratio_thr):
        print(f"[Skip] Output already exists: {final_out.name}")
        return

    if part_out.exists():
        print(f"[Info] Cleaning up historical temp file: {part_out.name}")
        part_out.unlink(missing_ok=True)

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
    writer = FFMPEGWriter(str(part_out), W + (W % 2), H + (H % 2), float(fps))

    def _sigint(sig, frame):
        print("\n[Interrupt] Cleaning up...")
        ret = writer.close(); cap.release()
        print(f"[Incomplete] Keeping temp file: {part_out.name} (ret={ret})")
        sys.exit(130)

    old_handler = signal.signal(signal.SIGINT, _sigint)
    print(f"\n[Processing] {path.name} -> {final_out.name}")
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        # 1) Person detection (full frame)
        person_boxes = person_detect(
            person_model, frame, imgsz, person_conf, person_iou, device,
            min_person_px=min_person_px, expand_ratio=person_expand
        )

        # 2) Face detection
        face_boxes = []
        if hybrid:
            # Two paths: ROI & full frame
            faces_cascade = []
            if person_boxes:
                faces_cascade = face_detect_in_rois(
                    face_model, frame, person_boxes, imgsz, face_conf, face_iou, device,
                    upscale, face_expand, W, H
                )
            faces_full = face_detect_in_rois(
                face_model, frame, [(0,0,W,H)], imgsz, face_conf, face_iou, device,
                1.0, face_expand, W, H
            )
            merged = faces_cascade + faces_full
            face_boxes = nms_union_xyxy(merged, iou_thr=nms_union_iou)
        else:
            # Cascade only (optional fallback to full frame)
            if person_boxes:
                face_boxes = face_detect_in_rois(
                    face_model, frame, person_boxes, imgsz, face_conf, face_iou, device,
                    upscale, face_expand, W, H
                )
            elif fallback_fullframe:
                face_boxes = face_detect_in_rois(
                    face_model, frame, [(0,0,W,H)], imgsz, face_conf, face_iou, device,
                    1.0, face_expand, W, H
                )

        # 3) Frame-by-frame mosaic
        for (x1, y1, x2, y2) in face_boxes:
            frame = mosaic_region(frame, x1, y1, x2, y2, block=mosaic_block)

        writer.write(ensure_even(frame))

        if idx % 100 == 0:
            print(f"[{path.name}] Frame {idx} | person={len(person_boxes)} | face={len(face_boxes)}")

    retcode = writer.close()
    cap.release()
    signal.signal(signal.SIGINT, old_handler)

    if retcode == 0:
        try:
            if final_out.exists(): final_out.unlink()
            part_out.rename(final_out)
            print(f"[Complete] Export -> {final_out}")
        except Exception as e:
            print(f"[Warning] Rename failed, keeping temp file: {part_out}, error: {e}")
    else:
        print(f"[Failed] ffmpeg exit code {retcode}, keeping temp file: {part_out}")

# ----------------- Resume from Checkpoint -----------------
def is_completed(out_path: Path, min_size_kb: int, in_path: Optional[Path]=None, dur_ratio_thr: float=0.985) -> bool:
    if not (out_path.exists() and out_path.stat().st_size >= (min_size_kb * 1024)):
        return False
    if in_path is None:
        return True
    in_dur  = probe_duration_sec(str(in_path))
    out_dur = probe_duration_sec(str(out_path))
    if in_dur < 0 or out_dur < 0:
        return False
    return (out_dur >= in_dur * dur_ratio_thr)

# ----------------- Parameters -----------------
def parse_args():
    ap = argparse.ArgumentParser("Framewise Face Mosaic (Hybrid: full-frame ∪ person-ROI, no smoothing)")
    ap.add_argument("--videos_dir", type=str, default="face_included_videos", help="Input video directory")
    ap.add_argument("--out_dir", type=str, default="face_mosaic_videos_yoloPF", help="Output directory")
    ap.add_argument("--person_model", type=str, default="yolo11s.pt", help="COCO person detection model (person=0)")
    ap.add_argument("--face_model", type=str, default="runs/face_detect/weights/best.pt", help="Face detection model (.pt)")
    ap.add_argument("--device", type=str, default="0", help="GPU ID, e.g. '0'; use 'cpu' for CPU")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    # person
    ap.add_argument("--person_conf", type=float, default=0.30, help="Person detection confidence threshold")
    ap.add_argument("--person_iou", type=float, default=0.6, help="Person NMS IoU")
    ap.add_argument("--min_person_px", type=int, default=16, help="Minimum person width/height in pixels")
    ap.add_argument("--person_expand", type=float, default=0.22, help="Person ROI expansion ratio")
    # face
    ap.add_argument("--face_conf", type=float, default=0.22, help="Face detection confidence threshold")
    ap.add_argument("--face_iou", type=float, default=0.6, help="Face NMS IoU")
    ap.add_argument("--face_expand", type=float, default=0.50, help="Face box expansion ratio")
    ap.add_argument("--upscale", type=float, default=1.6, help="ROI upsampling ratio before face detection (ROI path only)")
    # hybrid & fallback
    ap.add_argument("--hybrid", action="store_true", help="Enable hybrid mode (full frame ∪ ROI)")
    ap.add_argument("--nms_union_iou", type=float, default=0.5, help="Hybrid mode merge IoU threshold")
    ap.add_argument("--fallback_fullframe", action="store_true", help="When hybrid not enabled: no person → full frame fallback")
    # Output & Resume from checkpoint
    ap.add_argument("--mosaic_block", type=int, default=13, help="Mosaic block size")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip existing outputs (default enabled)")
    ap.add_argument("--overwrite", action="store_true", help="Ignore existing outputs, force rerun")
    ap.add_argument("--min_size_kb", type=int, default=100, help="Minimum size in KB to consider output valid")
    ap.add_argument("--dur_ratio_thr", type=float, default=0.985, help="Output duration must be ≥ input duration × this ratio")
    return ap.parse_args()

def main():
    args = parse_args()

    if not Path(args.face_model).exists():
        print(f"[Error] Cannot find face model: {args.face_model}")
        sys.exit(1)

    device = args.device if args.device else None
    person_model = YOLO(args.person_model)
    face_model   = YOLO(args.face_model)

    vids = list_videos(args.videos_dir)
    if not vids:
        print(f"[Error] No videos found in {args.videos_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    pairs = [(vp, out_dir / f"{vp.stem}_redacted.mp4") for vp in vids]
    total = len(pairs)

    for idx, (vp, outp) in enumerate(pairs, 1):
        if (not args.overwrite) and is_completed(outp, args.min_size_kb, in_path=vp, dur_ratio_thr=args.dur_ratio_thr):
            print(f"[Progress] {idx}/{total} Skip: {vp.name}")
            continue
        print(f"[Progress] {idx}/{total} Processing: {vp.name}")
        process_video(
            path=vp,
            person_model=person_model,
            face_model=face_model,
            out_dir=args.out_dir,
            device=device,
            imgsz=args.imgsz,
            person_conf=args.person_conf, person_iou=args.person_iou,
            min_person_px=args.min_person_px, person_expand=args.person_expand,
            face_conf=args.face_conf, face_iou=args.face_iou,
            face_expand=args.face_expand, upscale=args.upscale,
            mosaic_block=args.mosaic_block,
            hybrid=args.hybrid, nms_union_iou=args.nms_union_iou,
            fallback_fullframe=args.fallback_fullframe,
            resume=(not args.overwrite and args.resume),
            min_size_kb=args.min_size_kb, dur_ratio_thr=args.dur_ratio_thr
        )

if __name__ == "__main__":
    main()
