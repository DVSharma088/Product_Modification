import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import google.generativeai as genai
from groundingdino.util.inference import predict, load_image
from modules.common_sam import dino_model, sam_predictor

OUTPUT_DIR = "static/output"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash-image")


def replace_wall(request):
    ts = int(time.time())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    temp_path = os.path.join(OUTPUT_DIR, f"temp_{ts}.jpg")
    request.files["target_image"].save(temp_path)

    image_rgb, image_tensor = load_image(temp_path)
    h, w, _ = image_rgb.shape
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    boxes, _, _ = predict(
        model=dino_model,
        image=image_tensor,
        caption="wall",
        box_threshold=0.32,
        text_threshold=0.25,
        device="cpu"
    )

    if len(boxes) == 0:
        raise RuntimeError("No wall detected")

    box = boxes[0] * torch.tensor([w, h, w, h])
    x_c, y_c, bw, bh = box.tolist()

    input_box = np.array([
        int(x_c - bw / 2),
        int(y_c - bh / 2),
        int(x_c + bw / 2),
        int(y_c + bh / 2)
    ])

    sam_predictor.set_image(image_rgb)
    mask = sam_predictor.predict(box=input_box, multimask_output=False)[0][0]

    wall_tex = cv2.imdecode(
        np.frombuffer(request.files["wall_image"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    wall_tex = cv2.resize(wall_tex, (w, h))

    alpha = cv2.GaussianBlur(mask.astype(float), (21, 21), 0)[..., None]
    blended = (wall_tex * alpha + image_bgr * (1 - alpha)).astype(np.uint8)

    sam_out = os.path.join(OUTPUT_DIR, f"sam_wall_{ts}.png")
    cv2.imwrite(sam_out, blended)

    response = gemini_model.generate_content([
        {
            "role": "user",
            "parts": [
                {"text": "Remove highlight line, seamless wall transition."},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": open(sam_out, "rb").read()
                    }
                }
            ]
        }
    ])

    for c in response.candidates:
        for p in c.content.parts:
            if p.inline_data:
                final_path = os.path.join(OUTPUT_DIR, f"final_wall_{ts}.png")
                with open(final_path, "wb") as f:
                    f.write(p.inline_data.data)
                return os.path.basename(final_path)

    raise RuntimeError("Gemini failed")
