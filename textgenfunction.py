def text_gen(bbox, image_path):
    import pytesseract
    from PIL import Image
    from langdetect import detect  # ✅ You must import detect

    image = Image.open(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))

    # Run OCR
    text = pytesseract.image_to_string(cropped, lang='eng+hin+ara')
    lang = detect(text) if text.strip() else "unknown"
    
    return text, lang
