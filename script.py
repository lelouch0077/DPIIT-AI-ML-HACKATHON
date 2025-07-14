import fitz  # PyMuPDF
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# === Unified CRNN Definition ===
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
        )
        self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 1, W]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W, 512]
        x, _ = self.rnn1(x)  # [B, W, 512]
        x, _ = self.rnn2(x)  # [B, W, 512]
        x = self.fc(x)  # [B, W, C]
        return x.permute(1, 0, 2)  # [W, B, C] for CTC

# === Combined Charset (English + Arabic) ===
CHARSET = sorted(set(
    "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإةى"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-+؟.,:()[]{}!\"'،٪؛ "
))

char2idx = {c: i+1 for i, c in enumerate(CHARSET)}  # 0 for CTC blank
idx2char = {i: c for c, i in char2idx.items()}

# === Decode Function ===
def decode_prediction(preds):
    preds = preds.argmax(2).squeeze(1).cpu().numpy().T
    results = []
    for pred in preds:
        filtered = [p for i, p in enumerate(pred) if (i == 0 or p != pred[i-1]) and p != 0]
        results.append(''.join([idx2char.get(c, '') for c in filtered]))
    return results

# === Load Model ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(num_classes=len(char2idx) + 1).to(DEVICE)
model.load_state_dict(torch.load(r"C:\path\to\combined_eng_arabic_crnn.pth", map_location=DEVICE)['model_state_dict'])
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor()
])

# === Main OCR Pipeline ===
def extract_text_from_pdf(pdf_path, annotations_json):
    output = []
    doc = fitz.open(pdf_path)

    with open(annotations_json, 'r', encoding='utf-8') as f:
        regions = json.load(f)

    for region in regions:
        bbox = region["bbox"]
        page_number = region.get("page", 0)
        region_type = region.get("type", "Text")

        page = doc[page_number]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        x1, y1, x2, y2 = bbox
        cropped = img.crop((x1, y1, x2, y2)).convert("L")
        crnn_input = transform(cropped).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(crnn_input)
            text = decode_prediction(pred)[0]

        output.append({
            "type": region_type,
            "bbox": bbox,
            "score": region["score"],
            "text": text
        })

    return output

# === Run ===
pdf_path = r"C:\path\to\document.pdf"
annotations_json = r"C:\path\to\input.json"

results = extract_text_from_pdf(pdf_path, annotations_json)

with open("output_combined.json", "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("✅ OCR extraction complete. Results saved.")
