import os
import time
from PIL import Image
from modules.ai_clients import GEMINI_IMAGE

STATIC_OUTPUT = os.path.join("static", "output")

def replace_accessory(request):
    ts = int(time.time())

    image = request.files["accessory_image"]
    new_accessory = request.form["new_accessory"]

    img = Image.open(image.stream).convert("RGB")

    prompt = f"""
You are a professional image editor.

Task:
- Identify any accessory present in the image
- Remove it cleanly
- Replace it with: {new_accessory}

Rules:
- Keep placement realistic
- Match lighting and shadows
- Do NOT change the main product
- Do NOT alter background
"""

    out_filename = f"replaced_accessory_{ts}.png"
    output_path = os.path.join(STATIC_OUTPUT, out_filename)

    gemini = GEMINI_IMAGE.generate_content([prompt, img])

    for part in gemini.candidates[0].content.parts:
        if part.inline_data:
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)

    return out_filename, ts
