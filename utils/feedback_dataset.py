from PIL import Image, ImageDraw
import os

FEEDBACK_TEXT = {
    "PNEUMONIA": (
        "PNEUMONIA DETECTED\n\n"
        "Findings suggest pneumonia (bacterial or viral).\n"
        "Recommended actions:\n"
        "  - Chest CT scan for confirmation\n"
        "  - Blood cultures & CBC with differential\n"
        "  - C-Reactive Protein (CRP) test\n"
        "  - Consider antibiotic therapy\n"
        "  - Pulmonology referral if severe\n"
        "  - Monitor O2 saturation closely"
    ),
    "NORMAL": (
        "NO PNEUMONIA DETECTED\n\n"
        "Chest X-ray appears within normal limits.\n"
        "No consolidation or infiltrates identified.\n"
        "Lung fields appear clear bilaterally.\n\n"
        "Recommendation:\n"
        "  - Routine follow-up if symptoms persist\n"
        "  - Re-evaluate if clinical condition changes"
    ),
}

FEEDBACK_COLOR = {
    "PNEUMONIA": (200, 50, 50),
    "NORMAL":    (50, 160, 50),
}


def doctor_feedback(label: str):
    # Normalize label
    label = label.upper()
    if label not in FEEDBACK_TEXT:
        label = "NORMAL"

    text  = FEEDBACK_TEXT[label]
    color = FEEDBACK_COLOR[label]

    # Try loading a real feedback image first
    img_path = f"feedback_images/{label.lower()}.png"
    if os.path.exists(img_path):
        return Image.open(img_path), text

    # Generate a clean card
    img  = Image.new("RGB", (560, 220), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 560, 44], fill=color)
    draw.text((14, 12), f"Doctor's Assessment: {label}", fill=(255, 255, 255))

    y = 56
    for line in text.split("\n"):
        draw.text((14, y), line, fill=(40, 40, 40))
        y += 20

    return img, text
