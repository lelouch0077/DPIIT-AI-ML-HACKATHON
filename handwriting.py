"""
pip install layoutparser
pip install pillow==9.5.0
"""

import cv2
import numpy as np
from PIL import Image
import layoutparser as lp
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from IPython.display import display
from PIL import ImageDraw

# Load TrOCR for handwriting
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Load and preprocess image
img_path = "/kaggle/input/test-pdf-handwriting4/test pdf handwriting4.png.jpg"


def detect_handwriting(img_path):
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Thresholding for contour detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) # ADJUST THE THRESHOLD

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and analyze contours
    filtered_boxes = []
    height, width = image_rgb.shape[:2]
    width_threshold = width * 0.05
    height_threshold = height * 0.05
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = h / w if w != 0 else 0
        aspect_ratio_inv = w / h if h != 0 else 0

        if area < 1000:
            continue
        if aspect_ratio > 5 and w < width_threshold:  
            continue
        if aspect_ratio_inv > 5 and h < height_threshold:
            continue

        filtered_boxes.append((x, y, w, h))

    # Filter isolated boxes
    centers = np.array([[x + w // 2, y + h // 2] for x, y, w, h in filtered_boxes])
    final_boxes = []
    height, width = image_rgb.shape[:2]
    distance_threshold = int(0.06 * (width + height))
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    area_threshold_keep = image_area * 0.007 
    for i, (cx, cy) in enumerate(centers):
        x, y, w, h = filtered_boxes[i]
        area = w * h

        distances = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
        close_neighbors = np.sum((distances < distance_threshold) & (distances > 0))

        if close_neighbors > 0 or area > area_threshold_keep:
            final_boxes.append((x, y, w, h))

    # Run TrOCR on final valid regions
    handwritten_blocks = []

    for (x, y, w, h) in final_boxes:
        segment = image_rgb[y:y+h, x:x+w]
        pil_segment = Image.fromarray(segment).convert("RGB")

        try:
            pixel_values = processor(images=pil_segment, return_tensors="pt").pixel_values
            generated_ids = trocr_model.generate(pixel_values, max_length=64)
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            continue

        if predicted_text and len(predicted_text.strip()) > 2:
            block = lp.TextBlock(lp.Rectangle(x, y, x+w, y+h), type='Handwritten', text=predicted_text.strip())
            handwritten_blocks.append(block)


    # 1. Extract bounding boxes and their centers from detected blocks
    lines = []
    for block in handwritten_blocks:
        box = block.block  # lp.Rectangle
        lines.append({
            'bbox': [(box.x_1, box.y_1), (box.x_2, box.y_1), (box.x_2, box.y_2), (box.x_1, box.y_2)]
        })

    # 2. Sort lines top to bottom using vertical center
    lines.sort(key=lambda l: (l['bbox'][0][1] + l['bbox'][2][1]) / 2)

    # 3. Estimate average height and define vertical threshold
    heights = [abs(l['bbox'][2][1] - l['bbox'][0][1]) for l in lines]
    avg_height = np.mean(heights)
    threshold = avg_height * 1.8  # adjustable

    # 4. Define vertical distance function
    def vertical_distance(bbox1, bbox2):
        y1 = (bbox1[0][1] + bbox1[2][1]) / 2
        y2 = (bbox2[0][1] + bbox2[2][1]) / 2
        return abs(y1 - y2)

    # 5. Group lines into paragraphs
    paragraphs = []
    current_para = [lines[0]]
    for i in range(1, len(lines)):
        if vertical_distance(lines[i]['bbox'], lines[i - 1]['bbox']) < threshold:
            current_para.append(lines[i])
        else:
            paragraphs.append(current_para)
            current_para = [lines[i]]
    paragraphs.append(current_para)

    # 6. Draw paragraphs with green rectangles
    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    paragraph_bboxes=[]
    for para in paragraphs:
        all_x = [pt[0] for line in para for pt in line['bbox']]
        all_y = [pt[1] for line in para for pt in line['bbox']]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        paragraph_bboxes.append((min_x, min_y, max_x, max_y))

        draw.rectangle([min_x, min_y, max_x, max_y], outline='green', width=3)

    # Show result
    display(canvas)

    return paragraph_bboxes, [], []
# canvas = image_pil.copy()
# drawn = lp.draw_box(canvas, lp.Layout(handwritten_blocks), box_width=3, show_element_type=True)
# # display(drawn)

# from IPython.display import display
# display(drawn)
