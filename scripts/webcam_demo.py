import argparse
import time
import cv2
import numpy as np
import torch

# === IMPORT YOUR MODEL CLASS ===
# If you put UNet in models/model.py and exported it in models/__init__.py:
from models.model import UNet

# ---- colors for 9 classes (BGR for OpenCV) ----
PALETTE = np.array([
    [  0,   0,   0],  # 0 background - black
    [ 50, 180, 255],  # 1 face skin - light orange-ish (BGR)
    [180,  50, 255],  # 2 left eyebrow
    [200,  80, 200],  # 3 right eyebrow
    [255, 100, 100],  # 4 left eye
    [100, 255, 100],  # 5 right eye
    [255, 255, 100],  # 6 mouth (lips)
    [180, 255, 255],  # 7 teeth
    [255, 150, 200],  # 8 hair/other part (adjust if needed)
], dtype=np.uint8)

def colorize_mask(mask_hw: np.ndarray) -> np.ndarray:
    """mask: (H,W) ints in [0..8] -> (H,W,3) BGR"""
    return PALETTE[mask_hw.clip(0, len(PALETTE)-1)]

def preprocess(frame_bgr: np.ndarray, img_size: int, mean, std, device) -> torch.Tensor:
    """BGR uint8 -> (1,3,H,W) float normalized, resized to (img_size,img_size)."""
    # resize
    img = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # BGR -> RGB
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    # normalize
    img[..., 0] = (img[..., 0] - mean[0]) / std[0]
    img[..., 1] = (img[..., 1] - mean[1]) / std[1]
    img[..., 2] = (img[..., 2] - mean[2]) / std[2]
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    t = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return t

def overlay_mask(frame_bgr: np.ndarray, mask_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend color mask onto original frame (resized to the same shape)."""
    mask_resized = cv2.resize(mask_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = cv2.addWeighted(frame_bgr, 1 - alpha, mask_resized, alpha, 0.0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="path to .pt checkpoint (best.pt)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=9)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std",  type=float, nargs=3, default=[0.229, 0.224, 0.225])
    ap.add_argument("--half", action="store_true", help="use half precision on CUDA")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---- build model & load weights ----
    model = UNet(in_channels=3, num_classes=args.num_classes, use_transpose=False).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model_state", ckpt)  # support both raw state_dict or Trainer checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    if args.half and device.type == "cuda":
        model.half()

    # ---- open webcam ----
    cap = cv2.VideoCapture(args.cam)
    # try a nicer default size on some cams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_t = time.time()
    fps = 0.0

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            # keep original frame for display; preprocess copy
            inp = preprocess(frame.copy(), args.img_size, args.mean, args.std, device)
            if args.half and device.type == "cuda":
                inp = inp.half()

            # inference
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(inp)           # (1,C,H,W)
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int32)  # (H,W)

            # colorize + overlay
            mask_color = colorize_mask(pred)                    # (H,W,3) BGR
            out = overlay_mask(frame, mask_color, alpha=args.alpha)

            # fps
            now = time.time()
            dt = now - last_t
            last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

            cv2.putText(out, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("UNet Segmentation (press q to quit)", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()