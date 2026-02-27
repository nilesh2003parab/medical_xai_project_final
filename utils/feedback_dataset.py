from PIL import Image, ImageDraw
import os

FEEDBACK_TEXT = {
    "glioma": (
        "GLIOMA DETECTED\n\n"
        "Gliomas are tumors arising from glial cells.\n"
        "They can be low-grade (slow-growing) or\n"
        "high-grade (aggressive, e.g. glioblastoma).\n\n"
        "Recommended actions:\n"
        "  - Urgent MRI with contrast (gadolinium)\n"
        "  - Neurosurgery consultation\n"
        "  - Biopsy for grading & molecular markers\n"
        "  - Consider surgical resection + radiation\n"
        "  - Oncology referral immediately"
    ),
    "meningioma": (
        "MENINGIOMA DETECTED\n\n"
        "Meningiomas arise from the meninges (brain lining).\n"
        "Most are benign and slow-growing.\n\n"
        "Recommended actions:\n"
        "  - MRI with contrast for full characterization\n"
        "  - Neurosurgery consultation\n"
        "  - Watch-and-wait if small & asymptomatic\n"
        "  - Surgical removal if causing symptoms\n"
        "  - Regular follow-up MRI scans"
    ),
    "notumor": (
        "NO TUMOR DETECTED\n\n"
        "MRI scan appears within normal limits.\n"
        "No significant mass, lesion, or abnormal\n"
        "signal intensity identified.\n\n"
        "Recommendation:\n"
        "  - Correlate with clinical symptoms\n"
        "  - Routine follow-up if symptoms persist\n"
        "  - Repeat imaging if neurological signs develop"
    ),
    "pituitary": (
        "PITUITARY TUMOR DETECTED\n\n"
        "Pituitary adenomas arise from the pituitary gland.\n"
        "Most are benign but may affect hormone levels.\n\n"
        "Recommended actions:\n"
        "  - Endocrinology consultation\n"
        "  - Full hormone panel blood tests\n"
        "  - Ophthalmology for visual field testing\n"
        "  - MRI with dedicated pituitary protocol\n"
        "  - Consider transsphenoidal surgery or medication"
    ),
}

FEEDBACK_COLOR = {
    "glioma":      (180, 40,  40),
    "meningioma":  (200, 100, 20),
    "notumor":     (40,  160, 40),
    "pituitary":   (60,  80,  200),
}


def doctor_feedback(label: str):
    label = label.lower().strip()
    if label not in FEEDBACK_TEXT:
        label = "notumor"

    text  = FEEDBACK_TEXT[label]
    color = FEEDBACK_COLOR[label]

    # Try loading a custom image
    img_path = f"feedback_images/{label}.png"
    if os.path.exists(img_path):
        return Image.open(img_path), text

    # Generate feedback card
    img  = Image.new("RGB", (580, 260), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 580, 48], fill=color)
    draw.text((14, 14), f"Doctor's Assessment: {label.upper()}", fill=(255, 255, 255))

    y = 62
    for line in text.split("\n"):
        draw.text((14, y), line, fill=(40, 40, 40))
        y += 20

    return img, text
