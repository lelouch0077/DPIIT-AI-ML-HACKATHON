import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import json

from transformers import (
    DetrImageProcessor,
    TableTransformerForObjectDetection,
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

# --- Load Models ---
detection_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
detection_model.eval()

structure_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
structure_model.eval()

ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
ocr_model.eval()

# --- OCR Function ---
def ocr_image(image_patch):
    pixel_values = ocr_processor(images=image_patch, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# --- Visualization Function ---
def plot_boxes(pil_img, boxes, labels, scores, model, title="Detected Objects"):
    import matplotlib.pyplot as plt

    COLORS = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100  # repeat colors if more labels

    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        label = int(label)  # 🔑 Convert tensor to int
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, color=c, linewidth=3
            )
        )
        text = f'{model.config.id2label[label]}: {score:.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.title(title)
    plt.show()

# --- Main Pipeline ---
def parse_table(image_path):
    image = Image.open(image_path).convert("RGB")

    # Step 1: Detect table region (outer box)
    inputs = detection_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = detection_model(**inputs)
    results = detection_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]], threshold=0.7)[0]

    table_bbox = None
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        label_name = detection_model.config.id2label[label.item()]
        if label_name == "table":
            table_bbox = list(map(int, box.tolist()))
            break

    if not table_bbox:
        return {"error": "No table found"}

    # Visualize detected table region
    plot_boxes(image, [table_bbox], [0], [1.0], detection_model, title="Detected Table Region")

    # Step 2: Crop and upscale table region
    # Step 2: Crop and upscale the table region
    x1, y1, x2, y2 = table_bbox

    # === Add padding to all four sides ===
    pad = 20  # You can adjust this (10–30 usually works well)
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, image.width)
    y2 = min(y2 + pad, image.height)

    # Crop with padding
    table_img = image.crop((x1, y1, x2, y2))

    # Optional: Upscale for better detection
    upscale_factor = 3
    table_img = table_img.resize(
        (table_img.width * upscale_factor, table_img.height * upscale_factor),
        Image.LANCZOS
    )

    # Step 3: Detect structure in the cropped table
    inputs = structure_processor(images=table_img, return_tensors="pt")
    with torch.no_grad():
        outputs = structure_model(**inputs)
    results = structure_processor.post_process_object_detection(outputs, target_sizes=[table_img.size[::-1]], threshold=0.6)[0]

    labels = results["labels"]
    boxes = results["boxes"]
    scores = results["scores"]

    plot_boxes(table_img, boxes, labels, scores, structure_model, title="Detected Rows, Columns, Headers")

    # Step 4: OCR & Organize
    cells = defaultdict(list)
    for label, score, box in zip(labels, scores, boxes):
        name = structure_model.config.id2label[label.item()]
        if name in ["table column header", "table column", "table row"]:
            cells[name].append(list(map(int, box.tolist())))

    def sort_cells(cells_list):
        return sorted(cells_list, key=lambda b: (b[1], b[0]))

    headers = []
    for box in sort_cells(cells.get("table column header", [])):
        x1, y1, x2, y2 = box
        cropped = table_img.crop((x1, y1, x2, y2))
        headers.append(ocr_image(cropped))

    row_data = []
    row_cells = sort_cells(cells.get("table column", []))
    row_rows = sort_cells(cells.get("table row", []))
    row_height = np.median([b[3] - b[1] for b in row_rows]) if row_rows else 40

    row_groups = defaultdict(list)
    for box in row_cells:
        y_center = (box[1] + box[3]) // 2
        row_index = int(y_center // row_height)
        row_groups[row_index].append(box)

    for i in sorted(row_groups):
        row = []
        for cell_box in sorted(row_groups[i], key=lambda b: b[0]):
            x1, y1, x2, y2 = cell_box
            cropped = table_img.crop((x1, y1, x2, y2))
            row.append(ocr_image(cropped))
        row_data.append(row)

    return {
        "type": "table",
        "bbox": table_bbox,
        "headers": headers,
        "rows": row_data
    }

# --- Run ---
image_path = "scanned.jpg"  # <== replace this with your image file name
result = parse_table(image_path)
print(json.dumps(result, indent=2, ensure_ascii=False))
