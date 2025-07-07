import fitz  # PyMuPDF
import json
import hashlib
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict

class PDFLayoutParser:
    def __init__(self, filter_config=None):
        """
        Initialize the layout parser with filtering configuration.
        
        Args:
            filter_config: Dictionary with filtering parameters
        """
        self.filter_config = filter_config or {
            'min_width': 100,
            'min_height': 100,
            'max_aspect_ratio': 8,
            'remove_duplicates': True,
            'min_area': 10000  # Minimum area in pixels
        }
        self.seen_hashes = set()
        self.stats = defaultdict(int)
    
    def get_image_hash(self, image_bytes):
        """Generate hash for duplicate detection."""
        return hashlib.md5(image_bytes).hexdigest()
    
    def is_meaningful_image(self, bbox, image_bytes):
        """
        Check if image is worth processing based on size and content.
        
        Args:
            bbox: Bounding box coordinates (x0, y0, x1, y1)
            image_bytes: Raw image bytes
            
        Returns:
            tuple: (is_valid, reason)
        """
        x0, y0, x1, y1 = bbox
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        area = width * height
        
        # Check minimum dimensions
        if width < self.filter_config['min_width'] or height < self.filter_config['min_height']:
            return False, f"too_small_{width}x{height}"
        
        # Check minimum area
        if area < self.filter_config['min_area']:
            return False, f"insufficient_area_{area}"
        
        # Check aspect ratio
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.filter_config['max_aspect_ratio']:
                return False, f"extreme_aspect_ratio_{aspect_ratio:.2f}"
        
        # Check for duplicates if enabled
        if self.filter_config['remove_duplicates']:
            img_hash = self.get_image_hash(image_bytes)
            if img_hash in self.seen_hashes:
                return False, "duplicate"
            self.seen_hashes.add(img_hash)
        
        # Additional content-based filtering
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            if self.is_uniform_image(image):
                return False, "uniform_content"
        except Exception:
            return False, "corrupted_image"
        
        return True, "valid"
    
    def is_uniform_image(self, image, threshold=0.15):
        """Check if image has very low entropy (uniform/decorative)."""
        # Convert to grayscale for entropy calculation
        gray = image.convert('L')
        histogram = gray.histogram()
        
        # Calculate entropy
        total_pixels = sum(histogram)
        if total_pixels == 0:
            return True
        
        entropy = 0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * np.log2(p)
        
        # Normalize entropy (0-8 for 8-bit grayscale)
        normalized_entropy = entropy / 8.0
        
        return normalized_entropy < threshold
    
    def extract_image_coordinates(self, pdf_path):
        """
        Extract coordinates of meaningful images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            list: List of image coordinate data
        """
        doc = fitz.open(pdf_path)
        image_coordinates = []
        
        print(f"📄 Processing PDF: {pdf_path}")
        print(f"🔍 Filter config: {self.filter_config}")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            print(f"📄 Page {page_num}: Found {len(image_list)} images")
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # Get image reference and extract
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    
                    # Get image rectangles (bounding boxes) on the page
                    img_rects = page.get_image_rects(xref)
                    
                    if not img_rects:
                        self.stats['no_coordinates'] += 1
                        continue
                    
                    # Process each occurrence of this image on the page
                    for rect in img_rects:
                        bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                        
                        # Check if image is meaningful
                        is_valid, reason = self.is_meaningful_image(bbox, img_bytes)
                        
                        if is_valid:
                            image_data = {
                                "bbox": bbox,
                                "page_number": page_num,
                                "image_index": img_index,
                                "xref": xref,
                                "width": abs(bbox[2] - bbox[0]),
                                "height": abs(bbox[3] - bbox[1]),
                                "area": abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
                            }
                            image_coordinates.append(image_data)
                            self.stats['accepted'] += 1
                        else:
                            self.stats[f'filtered_{reason}'] += 1
                            
                except Exception as e:
                    print(f"⚠️ Error processing image {img_index} on page {page_num}: {e}")
                    self.stats['processing_errors'] += 1
        
        doc.close()
        
        # Sort by page number and then by area (larger images first)
        image_coordinates.sort(key=lambda x: (x['page_number'], -x['area']))
        
        return image_coordinates
    
    def save_layout_data(self, pdf_path, output_path="layout_data.json"):
        """
        Extract and save layout data to JSON file.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Path to save JSON output
            
        Returns:
            dict: Layout data
        """
        image_coordinates = self.extract_image_coordinates(pdf_path)
        
        # Create output structure
        layout_data = {
            "metadata": {
                "pdf_path": pdf_path,
                "total_images_found": sum(self.stats.values()),
                "meaningful_images": len(image_coordinates),
                "filter_config": self.filter_config,
                "processing_stats": dict(self.stats)
            },
            "images": [
                {
                    "bbox": img["bbox"],
                    "page_number": img["page_number"],
                    "dimensions": {
                        "width": img["width"],
                        "height": img["height"],
                        "area": img["area"]
                    },
                    "image_index": img["image_index"],
                    "xref": img["xref"]
                }
                for img in image_coordinates
            ]
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("📊 LAYOUT PARSING STATISTICS")
        print("="*60)
        print(f"📄 Total images found: {sum(self.stats.values())}")
        print(f"✅ Meaningful images: {len(image_coordinates)}")
        print(f"📊 Filtering breakdown:")
        for key, value in self.stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        print("="*60)
        print(f"💾 Layout data saved to: {output_path}")
        
        return layout_data

# ----------- 🎯 Filter Presets -----------

def get_filter_presets():
    """Predefined filter configurations for different use cases."""
    return {
        "strict": {
            'min_width': 200,
            'min_height': 200,
            'max_aspect_ratio': 5,
            'remove_duplicates': True,
            'min_area': 40000
        },
        "moderate": {
            'min_width': 100,
            'min_height': 100,
            'max_aspect_ratio': 8,
            'remove_duplicates': True,
            'min_area': 10000
        },
        "lenient": {
            'min_width': 50,
            'min_height': 50,
            'max_aspect_ratio': 15,
            'remove_duplicates': True,
            'min_area': 2500
        },
        "no_filter": {
            'min_width': 1,
            'min_height': 1,
            'max_aspect_ratio': 1000,
            'remove_duplicates': False,
            'min_area': 1
        }
    }

# ----------- 🚀 Main Execution -----------

def main():
    """Main function to run the layout parser."""
    PDF_PATH = r"C:\Users\RithRajak\OneDrive\Desktop\DPIIT-AI-ML-HACKATHON\bio_repro.pdf"
    OUTPUT_PATH = "layout_data.json"
    
    # Choose filter preset
    filter_presets = get_filter_presets()
    
    # Initialize parser with moderate filtering
    parser = PDFLayoutParser(filter_config=filter_presets["moderate"])
    
    # Extract and save layout data
    layout_data = parser.save_layout_data(PDF_PATH, OUTPUT_PATH)
    
    # Preview first few results
    print(f"\n🔍 Preview of first 3 images:")
    for i, img in enumerate(layout_data["images"][:3]):
        print(f"Image {i+1}: Page {img['page_number']}, BBox: {img['bbox']}")
    
    return layout_data

if __name__ == "__main__":
    main()