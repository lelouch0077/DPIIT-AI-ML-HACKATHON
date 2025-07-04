import os
import requests
from PIL import Image
from io import BytesIO
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Image source - update with either a URL or local path
image_source = r"https://www.warmoven.in/media/catalog/product/cache/31bd8dbd686c1a4f8d6b9e7414b2f5e1/image/1473c229/doraemon-photo-cake.jpeg"

# Load image
if image_source.startswith("http://") or image_source.startswith("https://"):
    response = requests.get(image_source)
    if response.status_code != 200:
        raise Exception(f"Failed to download image: {response.status_code}")
    if "image" not in response.headers.get("Content-Type", ""):
        raise Exception("The URL did not return an image.")
    image = Image.open(BytesIO(response.content)).convert("RGB")
elif os.path.exists(image_source):
    image = Image.open(image_source).convert("RGB")
else:
    raise FileNotFoundError(f"File not found: {image_source}")

# Generate caption
inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

# Format result as JSON
result = {
    "type": "image",
    "bbox": [],  # Can be populated later with actual bounding box data
    "caption": caption
}

# Print or save the result
print(json.dumps(result, indent=2))
