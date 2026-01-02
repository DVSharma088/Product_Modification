import os
import torch
from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor

# ================= FORCE CPU =================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

# ================= BASE PATH =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------- PATHS ----------------
DINO_CONFIG = os.path.join(
    BASE_DIR,
    "Grounded-Segment-Anything",
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinT_OGC.py"
)

DINO_CHECKPOINT = os.path.join(
    BASE_DIR,
    "Grounded-Segment-Anything",
    "weights",
    "groundingdino_swint_ogc.pth"
)

SAM_CHECKPOINT = os.path.join(
    BASE_DIR,
    "sam_vit_b_01ec64.pth"
)

# ---------------- LOAD MODELS ----------------
print("🔹 Loading GroundingDINO + SAM (CPU)")

dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device="cpu")

sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)

sam_predictor = SamPredictor(sam)
