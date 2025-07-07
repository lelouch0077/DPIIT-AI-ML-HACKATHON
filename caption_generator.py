import fitz  # PyMuPDF
import json
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from concurrent.futures import ThreadPoolExecutor
import time

class ImageCaptionGenerator:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl"):
        """
        Initialize the caption generator with BLIP2 model.
        
        Args:
            model_name: HuggingFace model name for BLIP2
        """
        print(f"🤖 Loading BLIP2 model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        if torch.cuda.is_available():
            self.model.to(self.device)
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def load_layout_data(self, layout_json_path):
        """
        Load layout data from JSON file.
        
        Args:
            layout_json_path: Path to layout JSON file
            
        Returns:
            dict: Layout data
        """
        with open(layout_json_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        
        print(f"📄 Loaded layout data: {len(layout_data['images'])} images")
        return layout_data
    
    def extract_image_from_pdf(self, pdf_path, bbox, page_number, xref):
        """
        Extract a specific image from PDF using coordinates.
        
        Args:
            pdf_path: Path to PDF file
            bbox: Bounding box coordinates (x0, y0, x1, y1)
            page_number: Page number
            xref: Image reference number
            
        Returns:
            PIL.Image: Extracted image
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Method 1: Try to extract directly using xref
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                doc.close()
                return image
            except:
                pass
            
            # Method 2: Extract from page using coordinates
            page = doc.load_page(page_number)
            
            # Create a rectangle for the image area
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # Get higher resolution
            zoom = 2.0  # Increase resolution
            mat = fitz.Matrix(zoom, zoom)
            
            # Render the specific area
            pix = page.get_pixmap(matrix=mat, clip=rect)
            img_bytes = pix.tobytes("png")
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            doc.close()
            return image
            
        except Exception as e:
            print(f"⚠️ Error extracting image: {e}")
            return None
    
    def resize_image_for_model(self, image, target_size=(512, 512)):
        """
        Resize image optimally for model processing.
        
        Args:
            image: PIL Image
            target_size: Target size tuple (width, height)
            
        Returns:
            PIL.Image: Resized image
        """
        # Calculate aspect ratio preserving resize
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create canvas with target size
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        
        # Center the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        canvas.paste(resized_image, (x_offset, y_offset))
        
        return canvas
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        """
        Generate caption for a single image.
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        try:
            # Resize image for optimal processing
            processed_image = self.resize_image_for_model(image)
            
            # Prepare inputs
            inputs = self.processor(images=processed_image, return_tensors="pt")
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            return "Error generating caption"
    
    def process_single_image(self, pdf_path, image_data, image_index):
        """
        Process a single image and generate caption.
        
        Args:
            pdf_path: Path to PDF file
            image_data: Image data from layout JSON
            image_index: Index of the image
            
        Returns:
            dict: Result with caption
        """
        start_time = time.time()
        
        bbox = image_data["bbox"]
        page_number = image_data["page_number"]
        xref = image_data.get("xref", 0)
        
        print(f"🖼️ Processing image {image_index + 1}: Page {page_number}, BBox: {bbox}")
        
        # Extract image
        image = self.extract_image_from_pdf(pdf_path, bbox, page_number, xref)
        
        if image is None:
            return {
                "type": "image",
                "bbox": bbox,
                "page_number": page_number,
                "caption": "Failed to extract image",
                "status": "error",
                "processing_time": time.time() - start_time
            }
        
        # Generate caption
        caption = self.generate_caption(image)
        
        # Create result
        result = {
            "type": "image",
            "bbox": bbox,
            "page_number": page_number,
            "caption": caption,
            "status": "success",
            "processing_time": time.time() - start_time,
            "image_dimensions": image.size
        }
        
        print(f"✅ Caption: {caption} (Time: {result['processing_time']:.2f}s)")
        
        return result
    
    def process_batch(self, pdf_path, layout_data, batch_size=1, max_workers=2):
        """
        Process images in batches with optional parallel processing.
        
        Args:
            pdf_path: Path to PDF file
            layout_data: Layout data from JSON
            batch_size: Number of images per batch
            max_workers: Number of parallel workers
            
        Returns:
            list: List of caption results
        """
        images = layout_data["images"]
        results = []
        
        print(f"🚀 Processing {len(images)} images with {max_workers} workers")
        
        if max_workers == 1:
            # Sequential processing
            for i, image_data in enumerate(images):
                result = self.process_single_image(pdf_path, image_data, i)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_single_image, pdf_path, image_data, i)
                    for i, image_data in enumerate(images)
                ]
                
                for future in futures:
                    result = future.result()
                    results.append(result)
        
        return results
    
    def process_layout_and_generate_captions(self, pdf_path, layout_json_path, 
                                           output_path="captions_output.json", 
                                           max_workers=1):
        """
        Main function to process layout data and generate captions.
        
        Args:
            pdf_path: Path to PDF file
            layout_json_path: Path to layout JSON file
            output_path: Path to save caption results
            max_workers: Number of parallel workers
            
        Returns:
            dict: Complete results with metadata
        """
        print(f"📋 Starting caption generation process...")
        start_time = time.time()
        
        # Load layout data
        layout_data = self.load_layout_data(layout_json_path)
        
        # Process images
        results = self.process_batch(pdf_path, layout_data, max_workers=max_workers)
        
        # Calculate statistics
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        total_time = time.time() - start_time
        avg_time = total_time / len(results) if results else 0
        
        # Create final output
        final_output = {
            "metadata": {
                "pdf_path": pdf_path,
                "layout_json_path": layout_json_path,
                "total_images": len(results),
                "successful_captions": successful,
                "failed_captions": failed,
                "total_processing_time": total_time,
                "average_time_per_image": avg_time,
                "model_used": "Salesforce/blip2-flan-t5-xl"
            },
            "captions": results
        }
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 CAPTION GENERATION SUMMARY")
        print("="*60)
        print(f"📊 Total images processed: {len(results)}")
        print(f"✅ Successful captions: {successful}")
        print(f"❌ Failed captions: {failed}")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"⚡ Average time per image: {avg_time:.2f}s")
        print(f"💾 Results saved to: {output_path}")
        print("="*60)
        
        return final_output

# ----------- 🚀 Main Execution -----------

def main():
    """Main function to run caption generation."""
    # File paths
    PDF_PATH = r"C:\Users\RithRajak\OneDrive\Desktop\DPIIT-AI-ML-HACKATHON\bio_repro.pdf"
    LAYOUT_JSON_PATH = "layout_data.json"
    OUTPUT_PATH = "captions_output.json"
    
    # Initialize caption generator
    caption_generator = ImageCaptionGenerator()
    
    # Process and generate captions
    results = caption_generator.process_layout_and_generate_captions(
        pdf_path=PDF_PATH,
        layout_json_path=LAYOUT_JSON_PATH,
        output_path=OUTPUT_PATH,
        max_workers=1  # Set to 1 for sequential processing, 2+ for parallel
    )
    
    # Preview first few results
    print(f"\n🔍 Preview of first 3 captions:")
    for i, result in enumerate(results["captions"][:3]):
        print(f"Caption {i+1}: {result['caption']}")
    
    return results

if __name__ == "__main__":
    main()