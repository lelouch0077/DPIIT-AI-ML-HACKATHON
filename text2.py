import subprocess
import sys
import json
from PIL import Image

image_path = r"C:\Users\shrey\AIML\DPIIT\input_image.png"
layout_json_path = r"C:\Users\shrey\AIML\DPIIT\layout.json"
output_json_path = r"C:\Users\shrey\AIML\DPIIT\final_output.json"

def setup_environment():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])

    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return pytesseract

def text_gen(bbox, image_path):
    from PIL import Image
    from langdetect import detect
    pytesseract = setup_environment()

    image = Image.open(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))

    text = pytesseract.image_to_string(cropped, lang='eng+hin+ara')
    lang = detect(text) if text.strip() else "unknown"
    return text, lang

# Load image and layout
image = Image.open(image_path)
with open(layout_json_path, "r", encoding="utf-8") as f:
    layout_data = json.load(f)

output_data = []
for block in layout_data:
    bbox = block.get("bbox")
    block_type = block.get("type", "Text")

    if not bbox or len(bbox) != 4:
        continue

    text, lang = text_gen(bbox, image_path)
    print(text)

    output_data.append({
        "bbox": bbox,
        "type": block_type,
        "text": text,
        "language": lang
    })

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("Output saved to:", output_json_path)
