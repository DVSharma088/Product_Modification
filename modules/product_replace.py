import base64
import os
import time
import json
from io import BytesIO
from PIL import Image
import google.generativeai as genai

# ================== CONFIG ==================
TMP_DIR = os.path.join("static", "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-image")


# ------------------------------------------------
# STEP 1: ANALYZE IMAGE + SAVE IT SAFELY (JSON MODE)
# ------------------------------------------------
def analyze_image(request):
    if "setup" not in request.files:
        raise ValueError("No setup image provided")

    image = Image.open(request.files["setup"]).convert("RGB")

    ts = int(time.time())
    setup_filename = f"setup_{ts}.png"
    setup_path = os.path.join(TMP_DIR, setup_filename)

    # ✅ SAVE IMAGE
    image.save(setup_path)

    if not os.path.exists(setup_path):
        raise RuntimeError(f"Failed to save setup image at {setup_path}")

    response = model.generate_content([
        """
        Identify all distinct visible objects and fabrics.

        Return STRICT JSON only.
        No markdown.
        No bullet points.
        No extra text.

        Format EXACTLY like this:
        {
          "items": [
            "Table",
            "White tablecloth",
            "Green table runner"
          ]
        }
        """,
        image
    ])

    try:
        data = json.loads(response.text)
        items = data.get("items", [])
    except Exception as e:
        raise RuntimeError(
            f"Invalid JSON returned by Gemini:\n{response.text}"
        ) from e

    return items, setup_path


# ------------------------------------------------
# STEP 2: MULTI OBJECT REPLACEMENT (SAFE + SEQUENTIAL)
# ------------------------------------------------
def replace_product(request):
    if "setup_path" not in request.form:
        raise ValueError("Missing setup_path in form")

    setup_path = request.form["setup_path"]

    if not os.path.exists(setup_path):
        raise FileNotFoundError(f"Setup image not found: {setup_path}")

    current_image = Image.open(setup_path).convert("RGB")
    selected_items = request.form.getlist("selected_items")

    for file_key in request.files:
        if not file_key.startswith("product_"):
            continue

        index = file_key.split("_")[1]
        item_name = request.form.get(f"item_key_{index}")

        if item_name not in selected_items:
            continue

        product_img = Image.open(request.files[file_key]).convert("RGB")

        response = model.generate_content([
            f"""
            Replace ONLY the object named:
            "{item_name}"

            Use the second image as replacement.
            Keep background, lighting, shadows,
            perspective and all other objects unchanged.
            Maintain realism.
            """,
            current_image,
            product_img
        ])

        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break

        if image_bytes:
            buffer = BytesIO(image_bytes)
            buffer.seek(0)
            img = Image.open(buffer)
            img.load()
            current_image = img.convert("RGB")

    # ✅ FINAL OUTPUT
    out = BytesIO()
    current_image.save(out, format="PNG")
    out.seek(0)

    return base64.b64encode(out.read()).decode("utf-8")
