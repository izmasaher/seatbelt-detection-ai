"""
Seatbelt Detection — local upload UI
====================================
Runs fully on your own computer (CPU). Upload an image -> see if the seatbelt is worn.

Setup (once):
    pip install -r requirements.txt
Put the model next to this file, either as the zip from Google Drive
(seatbelt_model.zip) or as the unzipped folder (seatbelt_model/). Then:
    python app.py
and open http://127.0.0.1:7860 (it opens automatically).

Options:
    python app.py --model path/to/seatbelt_model(.zip)   # model somewhere else
    python app.py --port 8000                             # different port
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms as T
from transformers import AutoModelForImageClassification

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- model loading
def find_model(user_path=None):
    """Return a folder containing config.json + weights. Unzips seatbelt_model.zip if needed."""
    candidates = [Path(user_path)] if user_path else [
        HERE / "seatbelt_model", HERE / "model", HERE / "seatbelt_model.zip", HERE / "model.zip",
    ]
    for c in candidates:
        c = c.expanduser().resolve()
        if c.is_dir() and (c / "config.json").exists():
            return c
        if c.is_dir():                                   # zip extracted into a sub-folder
            hits = list(c.rglob("config.json"))
            if hits:
                return hits[0].parent
        if c.is_file() and c.suffix == ".zip":
            out = c.with_suffix("")
            if not (out / "config.json").exists():
                print(f"Unzipping {c.name} -> {out.name}/ ...")
                with zipfile.ZipFile(c) as z:
                    z.extractall(out)
            hits = [out] if (out / "config.json").exists() else [p.parent for p in out.rglob("config.json")]
            if hits:
                return hits[0]
    sys.exit(
        "\n❌ Model not found.\n"
        "   Download 'seatbelt_model.zip' from Google Drive (My Drive > Seatbelt_Detection)\n"
        f"   and put it in: {HERE}\n"
        "   or run:  python app.py --model C:/path/to/seatbelt_model.zip\n"
    )


def load_settings(model_dir, model):
    """Preprocessing + threshold exactly as used for the test set in the notebook."""
    cfg_file = model_dir / "inference_config.json"
    if cfg_file.exists():
        return json.loads(cfg_file.read_text())
    # fallback for a model saved by an older notebook version
    pp = json.loads((model_dir / "preprocessor_config.json").read_text()) if (model_dir / "preprocessor_config.json").exists() else {}
    return {
        "classes": [model.config.id2label[i] for i in sorted(model.config.id2label)],
        "image_size": int(getattr(model.config, "image_size", 224)),
        "image_mean": pp.get("image_mean", [0.5, 0.5, 0.5]),
        "image_std": pp.get("image_std", [0.5, 0.5, 0.5]),
        "no_seatbelt_threshold": float(getattr(model.config, "no_seatbelt_threshold", 0.5)),
    }


# --------------------------------------------------------------------------- inference
class SeatbeltClassifier:
    def __init__(self, model_dir):
        torch.set_num_threads(os.cpu_count() or 1)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir).eval()
        self.cfg = load_settings(model_dir, self.model)
        s = self.cfg["image_size"]
        # identical to eval_tf in the notebook -> identical predictions
        self.tf = T.Compose([T.Resize((s, s)), T.ToTensor(), T.Normalize(self.cfg["image_mean"], self.cfg["image_std"])])
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.no_idx = label2id["no_seatbelt"]
        self.threshold = float(self.cfg["no_seatbelt_threshold"])

    @torch.no_grad()
    def p_no_seatbelt(self, images):
        x = torch.stack([self.tf(img.convert("RGB")) for img in images])
        return torch.softmax(self.model(pixel_values=x).logits.float(), -1)[:, self.no_idx].numpy()


_detector = None
def get_detector():
    """COCO-pretrained YOLOv8n (class 0 = person). Loaded only when Full-scene mode is used."""
    global _detector
    if _detector is None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Full-scene mode needs ultralytics:  pip install ultralytics")
        _detector = YOLO(str(HERE / "yolov8n.pt") if (HERE / "yolov8n.pt").exists() else "yolov8n.pt")
    return _detector


def verdict_html(p_no, thr, title=""):
    bad = p_no >= thr
    colour, bg = ("#b91c1c", "#fee2e2") if bad else ("#15803d", "#dcfce7")
    text = "🚨 NO SEATBELT — violation" if bad else "✅ Seatbelt worn"
    return (f'<div style="padding:18px;border-radius:12px;background:{bg};border:2px solid {colour};">'
            f'<div style="font-size:26px;font-weight:700;color:{colour};">{title}{text}</div>'
            f'<div style="margin-top:6px;color:#333;">P(no_seatbelt) = <b>{p_no:.3f}</b> &nbsp;·&nbsp; '
            f'threshold = {thr:.2f}</div></div>')


def run_crop(clf, img, thr):
    p_no = float(clf.p_no_seatbelt([img])[0])
    return verdict_html(p_no, thr), {"no_seatbelt": p_no, "seatbelt": 1 - p_no}, None


def run_scene(clf, img, thr, conf=0.35):
    res = get_detector()(img, classes=[0], conf=conf, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    if len(boxes) == 0:
        msg = ('<div style="padding:18px;border-radius:12px;background:#fef9c3;border:2px solid #a16207;">'
               '<b>No person found in this image.</b><br>If it is already a close-up of the driver, '
               'switch the mode to <i>Driver crop</i>.</div>')
        return msg, {}, img
    crops = [img.crop(tuple(map(float, b))) for b in boxes]
    probs = clf.p_no_seatbelt(crops)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    lw = max(3, round(min(img.size) / 200))
    try:
        font = ImageFont.load_default(size=max(14, round(min(img.size) / 30)))
    except TypeError:                                    # very old Pillow
        font = ImageFont.load_default()
    rows = []
    for k, (b, p) in enumerate(zip(boxes, probs), 1):
        bad = p >= thr
        col = (220, 38, 38) if bad else (22, 163, 74)
        draw.rectangle(tuple(map(float, b)), outline=col, width=lw)
        tag = f"#{k} {'NO BELT' if bad else 'BELT'} {p:.2f}"
        tb = draw.textbbox((b[0], b[1]), tag, font=font)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=col)
        draw.text((b[0], b[1]), tag, fill="white", font=font)
        rows.append((k, p, bad))
    n_bad = sum(r[2] for r in rows)
    worst = float(max(probs))
    summary = verdict_html(worst, thr, title=f"{len(rows)} person(s), {n_bad} violation(s) — ")
    detail = "".join(f"<li>Person #{k}: {'🚨 no seatbelt' if bad else '✅ seatbelt'} (P(no)={p:.3f})</li>"
                     for k, p, bad in rows)
    return summary + f"<ul style='margin-top:8px'>{detail}</ul>", {f"person #{k}: P(no_seatbelt)": float(p) for k, p, _ in rows}, out


# --------------------------------------------------------------------------- UI
def build_ui(clf):
    import gradio as gr

    m = clf.cfg.get("test_metrics")
    info = (f"Model: fine-tuned **{clf.cfg.get('base_model', 'ViT')}** · tuned threshold **{clf.threshold:.2f}**")
    if m:
        info += (f" · test set ({m['n_test_images']} images): macro-F1 **{m['macro_f1']:.3f}**, "
                 f"no-seatbelt recall **{m['no_seatbelt_recall']:.3f}**, precision **{m['no_seatbelt_precision']:.3f}**")

    def predict(img, mode, thr):
        if img is None:
            return "<b>Please upload an image first.</b>", {}, None
        img = ImageOps.exif_transpose(img).convert("RGB")   # phone photos: respect rotation
        try:
            if mode.startswith("Full"):
                return run_scene(clf, img, thr)
            return run_crop(clf, img, thr)
        except Exception as e:
            return f"<b style='color:#b91c1c'>Error:</b> {type(e).__name__}: {e}", {}, None

    with gr.Blocks(title="Seatbelt Detection") as demo:
        gr.Markdown("# 🚗 Seatbelt Detection\nUpload a photo of a driver / windshield view and click **Check**.")
        gr.Markdown(info)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload image", sources=["upload", "clipboard"], height=380)
                mode = gr.Radio(["Driver crop (most accurate)", "Full scene (find people with YOLOv8 first)"],
                                value="Driver crop (most accurate)", label="Image type")
                thr = gr.Slider(0.05, 0.95, value=round(clf.threshold, 2), step=0.01,
                                label="Decision threshold P(no_seatbelt)  (default = tuned on validation)")
                with gr.Row():
                    btn = gr.Button("Check", variant="primary")
                    gr.ClearButton(components=[inp])
            with gr.Column():
                verdict = gr.HTML(label="Result")
                probs = gr.Label(label="Probabilities", num_top_classes=10)
                boxed = gr.Image(label="Detections (Full-scene mode)", type="pil", height=380)
        btn.click(predict, [inp, mode, thr], [verdict, probs, boxed])
        inp.upload(predict, [inp, mode, thr], [verdict, probs, boxed])     # auto-run on upload
        gr.Markdown("Green box = seatbelt worn, red box = violation. "
                    "For real enforcement a flagged image should still be checked by a person.")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Seatbelt detection local UI")
    ap.add_argument("--model", help="path to seatbelt_model folder or seatbelt_model.zip")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    model_dir = find_model(args.model)
    print("Loading model from:", model_dir)
    clf = SeatbeltClassifier(model_dir)
    print(f"Ready. Threshold P(no_seatbelt) >= {clf.threshold:.2f}  |  running on CPU")
    build_ui(clf).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)
