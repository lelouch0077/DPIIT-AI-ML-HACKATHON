import threading
from PIL import Image, ImageDraw
from layoutparser.elements import Rectangle, Layout, TextBlock
import matplotlib.pyplot as plt

# === Import model functions ===
from handwriting import detect_handwriting
from printed_text import detect_printed_text
from table_image import detect_text_table_figure


# === Threading Function Wrapper ===
def run_model(model_fn, img_path, out_list1, out_list2=None, out_list3=None):
    result, a, b = model_fn(img_path)
    if out_list1 is not None:
        out_list1.extend(result)
    if out_list2 is not None and a is not None:
        out_list2.extend(a)
    if out_list3 is not None and b is not None:
        out_list3.extend(b)

# === Visualize Layout ===
def draw_layout(image, layout, color_map):
    draw = ImageDraw.Draw(image)
    for block in layout:
        x1, y1, x2, y2 = block.coordinates
        draw.rectangle([x1, y1, x2, y2], outline=color_map.get(block.type, "black"), width=2)
        draw.text((x1, y1 - 10), block.type, fill=color_map.get(block.type, "black"))
    return image

# === Main Pipeline ===
def process_image(img_path):
    # Shared output lists
    handwritten_boxes = []
    printed_boxes = []
    table_boxes = []
    image_boxes = []

    # Threads for each model
    t1 = threading.Thread(target=run_model, args=(detect_handwriting, img_path, handwritten_boxes))
    t2 = threading.Thread(target=run_model, args=(detect_printed_text, img_path, printed_boxes))
    t3 = threading.Thread(target=run_model, args=(detect_text_table_figure, img_path, printed_boxes, table_boxes, image_boxes))

    # Start threads
    t1.start()
    t2.start()
    t3.start()

    # Wait for all to complete
    t1.join()
    t2.join()
    t3.join()

    # Combine into a Layout
    layout = Layout()

    for box in handwritten_boxes:
        layout.append(TextBlock(Rectangle(*box), type="handwritten"))
    for box in printed_boxes:
        layout.append(TextBlock(Rectangle(*box), type="printed"))
    for box in table_boxes:
        layout.append(TextBlock(Rectangle(*box), type="table"))
    for box in image_boxes:
        layout.append(TextBlock(Rectangle(*box), type="image"))

    # Load and draw on image
    image = Image.open(img_path).convert("RGB")
    color_map = {"handwritten": "blue", "printed": "green", "table": "red", "image":"orange"}

    output_image = draw_layout(image, layout, color_map)

    # Show image
    plt.figure(figsize=(12, 10))
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()

    return layout  # Optional: return layout for further processing

# === Run on an image ===
img_path = "path/to/your/image.jpg"
layout = process_image(img_path)
