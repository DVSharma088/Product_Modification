import os
import torch
import requests

from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor

# =========================================================
# FORCE CPU (Render-safe)
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- GroundingDINO ----------------
DINO_CONFIG = os.path.join(
    os.path.dirname(load_model.__file__),
    "config",
    "GroundingDINO_SwinT_OGC.py"
)

DINO_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "groundingdino_swint_ogc.pth"
)

DINO_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

# ---------------- SAM ----------------
SAM_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "sam_vit_b_01ec64.pth"
)

SAM_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/"
    "sam_vit_b_01ec64.pth"
)

# =========================================================
# UTIL: SAFE DOWNLOAD
# =========================================================
def download_if_missing(path: str, url: str):
    if os.path.exists(path):
        return

    print(f"⬇️ Downloading {os.path.basename(path)}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"✅ Downloaded {os.path.basename(path)}")

# =========================================================
# DOWNLOAD MODELS IF NEEDED
# =========================================================
download_if_missing(DINO_CHECKPOINT, DINO_URL)
download_if_missing(SAM_CHECKPOINT, SAM_URL)

# =========================================================
# LOAD MODELS (CPU)
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
