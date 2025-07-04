import requests
from PIL import Image
from io import BytesIO
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Change this to your image path or URL
image_source = r"https://img.freepik.com/free-photo/young-beautiful-woman-wearing-red-lingerie-bed-morning_158538-10345.jpg?semt=ais_hybrid&w=740"
# image_source = "https://example.com/image.jpg"

# Load the image
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

print("Generated caption:", caption)
