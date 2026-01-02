import os
import time
import base64
from PIL import Image
from modules.ai_clients import openai_client, GEMINI_IMAGE

STATIC_OUTPUT = os.path.join("static", "output")


# ---------- STEP 1: DETECT COLORS ----------
def detect_multicolors(request):
    ts = int(time.time())

    image = request.files["multicolor_image"]
    filename = f"multicolor_{ts}.jpg"
    image_path = os.path.join(STATIC_OUTPUT, filename)

    img = Image.open(image.stream).convert("RGB")
    img.save(image_path, "JPEG", quality=90)

    image_b64 = base64.b64encode(open(image_path, "rb").read()).decode()

    prompt = """
Identify DISTINCT product colors only.
Ignore background, shadows, and highlights.
Return comma-separated color names.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }]
    )

    detected_colors = response.choices[0].message.content.strip()
    color_list = [c.strip() for c in detected_colors.split(",") if c.strip()]

    return color_list, filename, ts


# ---------- STEP 2: MODIFY COLOR ----------
def modify_detected_color(request):
    ts = int(time.time())

    source_color = request.form["source_color"]
    target_color = request.form["target_color"]
    filename = request.form["filename"]

    img_path = os.path.join(STATIC_OUTPUT, filename)
    img = Image.open(img_path).convert("RGB")

    prompt = f"""
Change ONLY regions with color {source_color} to {target_color}.
Preserve all textures, folds, shadows.
Do NOT alter background or other colors.
"""

    out_filename = f"multicolor_{source_color}_to_{target_color}_{ts}.png"
    output_path = os.path.join(STATIC_OUTPUT, out_filename)

    gemini = GEMINI_IMAGE.generate_content([prompt, img])

    for part in gemini.candidates[0].content.parts:
        if part.inline_data:
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)

    return out_filename, ts
