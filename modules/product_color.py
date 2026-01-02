import os
import time
from PIL import Image
from modules.ai_clients import GEMINI_IMAGE

# ---------------- PATHS ----------------
STATIC_OUTPUT = os.path.join("static", "output")
TEXTURE_DIR = os.path.join("static", "textures")

os.makedirs(STATIC_OUTPUT, exist_ok=True)

# ---------------- COLOR MAP ----------------
COLOR_FABRIC_MAP = {
    "Midnight Green": "Midnight Green New.jpg",
    "Charcoal Drift": "Charcoal Drift.jpg",
    "Burl Wood": "BURL WOOD.jpg",
    "Sage Green": "Sage Green New.jpg",
    "Pacific Blue": "Pacific Blue New.jpg",
    "Angora White": "Angora White New.jpg",
    "Oatmeal": "Oatmeal New.jpg",
    "Misty Lilac": "Misty Lilac New.jpg",
    "Rose Wood": "ROSE WOOD.jpg",
    "Berry Blush": "Berry Blush New.jpg",
    "Sunbrunt Yellow": "Sunbrunt Yellow New.jpg",
    "Olive Mist": "OLIVE MIST .jpg",
    "Muted Lime": "MUTED LIME.jpg",
    "Muted Mocha": "Muted Mocha New.jpg",
    "Brown Bean": "Brown Bean New.jpg",
    "Petal Pink": "Petal Pink New.jpg",
    "Wild Wind": "WILD WIND.jpg"
}

# =========================================================
# ✅ MAIN FUNCTION CALLED FROM app.py
# =========================================================
def product_color(request):
    ts = int(time.time())

    target_file = request.files.get("target_image")
    color_target = request.form.get("color_target", "product")
    selected_colors = request.form.getlist("colors")

    if not target_file or not selected_colors:
        return None, None

    try:
        target_img = Image.open(target_file.stream).convert("RGB")
    except Exception as e:
        print("[ERROR] Invalid target image:", e)
        return None, None

    results = []

    for color in selected_colors:
        color_key = color.strip()

        if color_key not in COLOR_FABRIC_MAP:
            print(f"[WARN] Unsupported color: {color_key}")
            continue

        fabric_path = os.path.join(TEXTURE_DIR, COLOR_FABRIC_MAP[color_key])

        if not os.path.exists(fabric_path):
            print(f"[WARN] Missing fabric image: {fabric_path}")
            continue

        fabric_img = Image.open(fabric_path).convert("RGB")

        prompt = f"""
You are a professional textile color matching expert.

TASK:
- Change ONLY the {color_target} fabric in the FIRST image
- Match it EXACTLY to the fabric in the SECOND image

RULES:
- Preserve weave, texture, folds, lighting
- Do NOT modify non-fabric regions
- No smoothing, no hallucination

IMAGE ORDER:
1) Product Image
2) Fabric Reference
"""

        filename = f"{color_target}_{color_key.replace(' ', '_')}_{ts}.png"
        output_path = os.path.join(STATIC_OUTPUT, filename)

        try:
            response = GEMINI_IMAGE.generate_content(
                [prompt, target_img, fabric_img],
                generation_config={"temperature": 0.2}
            )
        except Exception as e:
            print(f"[ERROR] Gemini failed for {color_key}: {e}")
            continue

        image_written = False

        if not getattr(response, "candidates", None):
            print("[ERROR] Gemini returned no candidates")
            continue

        for part in response.candidates[0].content.parts:
            data = None

            if hasattr(part, "inline_data") and part.inline_data:
                data = part.inline_data.data
            elif hasattr(part, "data"):
                data = part.data

            if data:
                with open(output_path, "wb") as f:
                    f.write(data)
                image_written = True
                break

        if not image_written:
            print(f"[ERROR] No image returned for {color_key}")
            continue

        results.append({
            "label": f"{color_target.capitalize()} → {color_key}",
            "filename": filename
        })

    if not results:
        return None, None

    return results, ts
