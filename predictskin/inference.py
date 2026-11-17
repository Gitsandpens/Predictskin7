"""
PredictSkin - inference module
Loads both trained models once and runs the same image through each.

Usage:
    python inference.py path/to/image.jpg
    (or import predict_both() from your FastAPI/Flask backend)

Requires: torch, torchvision, pillow
"""
import json
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# CLASS NAMES  (order MUST match the label order used during training,
# i.e. the class_to_idx / sorted-folder order of your training script)
# ----------------------------------------------------------------------
FITZPATRICK_CLASSES = [
    "Psoriasis", "Squamous Cell Carcinoma", "Lichen Planus",
    "Basal Cell Carcinoma", "Allergic Contact Dermatitis",
    "Lupus Erythematosus", "Neutrophilic Dermatoses", "Sarcoidosis",
    "Photodermatoses", "Folliculitis", "Scabies", "Acne Vulgaris",
    "Scleroderma", "Pityriasis Rubra Pilaris", "Melanoma",
    "Nematode Infection", "Erythema Multiforme", "Granuloma Annulare",
    "Eczema", "Drug Eruption",
]

DERMNET_CLASSES = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]

# ----------------------------------------------------------------------
# MODEL CONFIG - edit paths / image size to match your training setup.
# "candidates" = architectures to try; the loader keeps the first one whose
# weights load cleanly, so a MobileNetV2/EfficientNet mix-up is handled.
# ----------------------------------------------------------------------
MODEL_CONFIG = {
    "fitzpatrick17k": {
        "path": "models/efficientnet_fitzpatrick17k_20class_final.pth",
        "classes": FITZPATRICK_CLASSES,
        "candidates": ["efficientnet_b3", "efficientnet_b0", "mobilenet_v2"],
        "img_size": 300,  # EfficientNet-B3 was trained at 300; also read from the checkpoint
    },
    "dermnet": {
        "path": "models/mobilenetv2_dermnet23_final.pth",
        "classes": DERMNET_CLASSES,
        "candidates": ["mobilenet_v2", "efficientnet_b3"],
        "img_size": 224,
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build(arch: str, num_classes: int) -> nn.Module:
    """Build a torchvision model with a fresh head of `num_classes`."""
    model = getattr(models, arch)(weights=None)
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def _extract_state_dict(ckpt):
    """Handle raw state_dicts and common checkpoint wrappers."""
    if isinstance(ckpt, nn.Module):  # whole model was saved
        return ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return {k.replace("module.", "", 1): v for k, v in ckpt.items()}


def load_model(cfg: dict) -> nn.Module:
    ckpt = torch.load(cfg["path"], map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg["classes"] = ckpt.get("class_names", cfg["classes"])
        cfg["img_size"] = ckpt.get("input_size", cfg["img_size"])
        cfg["mean"] = ckpt.get("mean", IMAGENET_MEAN)
        cfg["std"] = ckpt.get("std", IMAGENET_STD)
        a = str(ckpt.get("architecture", "")).lower().replace("-", "_")
        a = "mobilenet_v2" if a == "mobilenetv2" else a
        if hasattr(models, a):
            cfg["candidates"] = [a] + [c for c in cfg["candidates"] if c != a]
    state = _extract_state_dict(ckpt)
    if isinstance(state, nn.Module):
        return state.to(DEVICE).eval()

    errors = []
    for arch in cfg["candidates"]:
        try:
            model = _build(arch, len(cfg["classes"]))
            model.load_state_dict(state, strict=True)
            print(f"[load] {cfg['path']} -> {arch}")
            return model.to(DEVICE).eval()
        except Exception as e:  # wrong architecture, try the next one
            errors.append(f"{arch}: {str(e)[:120]}")
    raise RuntimeError(f"Could not load {cfg['path']}:\n" + "\n".join(errors))


def _transform(img_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# Load once at startup (important for a web backend)
import os
MODELS = {name: load_model(cfg) for name, cfg in MODEL_CONFIG.items() if os.path.exists(cfg["path"])}
for _n, _c in MODEL_CONFIG.items():
    if _n not in MODELS:
        print(f"[warning] {_c['path']} not found - running without the {_n} model")
if not MODELS:
    raise SystemExit("No model files found in the models/ folder.")


@torch.no_grad()
def predict_one(name: str, image: Image.Image, top_k: int = 3):
    cfg = MODEL_CONFIG[name]
    x = _transform(cfg["img_size"], cfg.get("mean", IMAGENET_MEAN), cfg.get("std", IMAGENET_STD))(image).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(MODELS[name](x), dim=1)[0]
    conf, idx = probs.topk(top_k)
    return [
        {"label": cfg["classes"][i], "confidence": round(c * 100, 2)}
        for c, i in zip(conf.tolist(), idx.tolist())
    ]


def predict_both(image_path: str, top_k: int = 3, low_conf: float = 50.0):
    image = Image.open(image_path).convert("RGB")
    result = {name: predict_one(name, image, top_k) for name in MODELS}
    result["low_confidence"] = {
        name: preds[0]["confidence"] < low_conf for name, preds in result.items()
        if name in MODELS
    }
    result["disclaimer"] = (
        "Educational tool only. Not a medical diagnosis - "
        "please consult a dermatologist."
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python inference.py path/to/image.jpg")
    print(json.dumps(predict_both(sys.argv[1]), indent=2))
