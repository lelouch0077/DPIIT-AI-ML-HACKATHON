"""
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
!pip install paddleocr
"""

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')



# img_path = '/kaggle/input/test-pdf-images/pdf test_6.jpg'

def detect_printed_text(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = ocr.predict(img_path) 

    data = result[0]  # get the dictionary inside the list
    boxes = data['rec_polys']
    texts = data['rec_texts']
    scores = data['rec_scores']


    scale = 1 / 1.067
    original_image = Image.open(img_path).convert("RGB")
    W_orig, H_orig = original_image.size
    W_scaled, H_scaled = int(W_orig * scale), int(H_orig * scale)

    pad_x = (W_orig - W_scaled) // 2
    pad_y = (H_orig - H_scaled) // 2


    scaled_boxes = []
    for box in boxes:
        scaled = [((x * scale) + pad_x, (y * scale) + pad_y) for (x, y) in box]
        scaled_boxes.append(scaled)

    def get_center_y(box):
        return (box[0][1] + box[2][1]) / 2  # Assuming 4-pt box in clockwise order

    sorted_lines = sorted(scaled_boxes, key=get_center_y)

    heights = [abs(box[2][1] - box[0][1]) for box in sorted_lines]
    avg_height = np.mean(heights)
    threshold = avg_height * 1.5  # Paragraph grouping threshold

    paragraphs = []
    current_para = [sorted_lines[0]]

    def vertical_distance(b1, b2):
        c1 = get_center_y(b1)
        c2 = get_center_y(b2)
        return abs(c1 - c2)

    for i in range(1, len(sorted_lines)):
        if vertical_distance(sorted_lines[i], sorted_lines[i - 1]) < threshold:
            current_para.append(sorted_lines[i])
        else:
            paragraphs.append(current_para)
            current_para = [sorted_lines[i]]
    paragraphs.append(current_para)

    canvas = original_image.copy()
    draw = ImageDraw.Draw(canvas)

    paragraph_bboxes=[]
    for para in paragraphs:
        all_x = [pt[0] for box in para for pt in box]
        all_y = [pt[1] for box in para for pt in box]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        paragraph_bboxes.append((min_x, min_y, max_x, max_y))
        draw.rectangle([min_x, min_y, max_x, max_y], outline='green', width=3)

    print(paragraph_bboxes)
    plt.figure(figsize=(15, 15))
    plt.imshow(canvas)
    plt.axis("off")
    plt.show()

    return paragraph_bboxes, [], []
