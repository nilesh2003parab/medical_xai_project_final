from PIL import Image
import os


def doctor_feedback(label):

    img_path = f"feedback_images/{label.lower()}.png"

    if os.path.exists(img_path):
        return Image.open(img_path), f"Doctor explanation for {label}"
    else:
        raise FileNotFoundError