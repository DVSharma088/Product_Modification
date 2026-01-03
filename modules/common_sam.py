import os
import torch
import requests
import importlib.resources as pkg_resources

from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor

# =========================================================
# FORCE CPU (Render-safe)
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

# =========================================================
# BASE PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# GROUNDING DINO CONFIG (PACKAGE SAFE)
# =========================================================
DINO_CONFIG = str(
    pkg_resources.files("groundingdino")
    .joinpath("config", "GroundingDINO_SwinT_OGC.py")
)

DINO_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "groundingdino_swint_ogc.pth"
)

DINO_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

# =========================================================
# SAM
# =========================================================
SAM_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "sam_vit_b_01ec64.pth"
)

SAM_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/"
    "sam_vit_b_01ec64.pth"
)

# =========================================================
# SAFE DOWNLOAD (handles corrupted / partial files)
# =========================================================
def download_if_missing(path: str, url: str, min_size_mb: int):
    """
    Download model if missing or corrupted.
    - GroundingDINO ≈ 700 MB
    - SAM ViT-B ≈ 375 MB
    """
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb >= min_size_mb:
            return
        else:
            print(f"⚠️ Corrupted file detected ({size_mb:.1f} MB), re-downloading {os.path.basename(path)}")
            os.remove(path)

    print(f"⬇️ Downloading {os.path.basename(path)}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"✅ Downloaded {os.path.basename(path)}")

# =========================================================
# ENSURE MODEL FILES
# =========================================================
download_if_missing(DINO_CHECKPOINT, DINO_URL, min_size_mb=600)
download_if_missing(SAM_CHECKPOINT, SAM_URL, min_size_mb=300)

# =========================================================
# LOAD MODELS (CPU ONLY)
# =========================================================
print("🔹 Loading GroundingDINO (CPU)")
dino_model = load_model(
    DINO_CONFIG,
    DINO_CHECKPOINT,
    device="cpu"
)

print("🔹 Loading SAM (CPU)")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)

sam_predictor = SamPredictor(sam)
