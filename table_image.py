"""
!pip install -U layoutparser
!pip install pillow==9.5.0 --force-reinstall
!pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2' 
!pip install layoutparser[ocr]     
!git clone https://github.com/Layout-Parser/layout-parser.git
%cd layout-parser/
"""

import cv2
import matplotlib.pyplot as plt
import layoutparser as lp



def detect_text_table_figure(image_path):
    image = cv2.imread(image_path)
    image = image[..., ::-1] 

    # import PIL.Image
    # PIL.Image.LINEAR = PIL.Image.BILINEAR 
    model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                    device='cuda')
    layout = model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
    lp.draw_box(image, layout)


    text_bboxes = []
    table_bboxes = []
    figure_bboxes = []
    for block in layout._blocks:
        if block.type in ("Text","List","Table"):
            rect = block.block
            bbox = [rect.x_1, rect.y_1, rect.x_2, rect.y_2]
            text_bboxes.append(bbox)
        
        elif block.type == "Table":
            rect = block.block
            bbox = [rect.x_1, rect.y_1, rect.x_2, rect.y_2]
            table_bboxes.append(bbox)

        elif block.type == "Figure":
            rect = block.block
            bbox = [rect.x_1, rect.y_1, rect.x_2, rect.y_2]
            figure_bboxes.append(bbox)

    return text_bboxes, table_bboxes, figure_bboxes

# lp.draw_box(image, text_blocks,
#             box_width=3, 
#             show_element_id=True)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis('off')  # Hide axis
# plt.title("Input Image")
# plt.show()