import requests
from PIL import Image
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ------------------------------
# Load BLIP-2 FlanT5 model + processor
# ------------------------------
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# ------------------------------
# Download image from URL
# ------------------------------
image_url = "https://cdn.britannica.com/24/174524-050-A851D3F2/Oranges.jpg?w=300"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert('RGB')

# ------------------------------
# Run caption generation
# ------------------------------
inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

print("Generated caption:", caption)
