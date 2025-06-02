from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import base64
import requests
import traceback
from flask_cors import CORS
from models.object_detector import ObjectDetector
from models.translator import ObjectTranslator
from models.sentence_generator import SentenceGenerator
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CROPS_DIR = os.path.join(STATIC_DIR, "crops")

os.makedirs(CROPS_DIR, exist_ok=True)
app = Flask(__name__, static_url_path='/static')
CORS(app)

try:
    detector = ObjectDetector()
    translator = ObjectTranslator("aka_Latn")
    sentence_gen = SentenceGenerator()
    print("All models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {e}")
    traceback.print_exc()

def decode_image(base64_string):
    try:
        if ',' in base64_string:
            image_data = base64.b64decode(base64_string.split(',')[-1])
        else:
            image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def preprocess_image_for_detection(image_path):
    """
    Preprocess image to improve detection results
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # Get original dimensions
        height, width = img.shape[:2]
        print(f"Original image size: {width}x{height}")
        
        # Resize if image is too small (min 640px on shortest side)
        min_side = min(width, height)
        if min_side < 640:
            scale = 640 / min_side
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"Resized to: {new_width}x{new_height}")
        
        # Enhance contrast and brightness
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Save preprocessed image
        processed_path = image_path.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_path, enhanced)
        print(f"Preprocessed image saved: {processed_path}")
        
        return processed_path
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return image_path

def validate_detection_results(result, boxes, classes):
    """
    Validate and filter detection results
    """
    if not result or not boxes or not classes:
        return [], []
    
    # Filter out low confidence detections
    confidence_threshold = 0.3
    filtered_boxes = []
    filtered_classes = []
    
    try:
        # Assuming your detector has confidence scores
        confidences = detector.get_confidence_scores(result) if hasattr(detector, 'get_confidence_scores') else None
        
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            # Check if confidence is available and above threshold
            if confidences is not None:
                if i < len(confidences) and confidences[i] < confidence_threshold:
                    continue
            
            # Validate bounding box
            if len(box) == 4:
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:  # Valid box
                    filtered_boxes.append(box)
                    filtered_classes.append(cls)
    
    except Exception as e:
        print(f"Error filtering results: {e}")
        # Return original if filtering fails
        return boxes, classes
    
    return filtered_boxes, filtered_classes

@app.route("/")
def home():
    return send_from_directory("frontend", "index2.html")

@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    return send_from_directory("frontend", filename)

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        if not data or "images" not in data:
            return jsonify({"error": "No images provided"}), 400

        image_b64_list = data.get("images")
        if not image_b64_list:
            return jsonify({"error": "Image list is empty"}), 400
        target_lang = data.get("language", "aka_Latn")
        supported_languages = ["aka_Latn", "en", "fr", "es", "de", "fra_Latn", "spa_Latn", "swh_Latn", "yor_Latn", "arb_Arab"]
        if target_lang not in supported_languages:
            return jsonify({"error": f"Unsupported language. Supported: {supported_languages}"}), 400

        translator.set_language(target_lang)

        final_output = []
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("static/crops"):
            os.makedirs("static/crops")
        
        for img_index, image_b64 in enumerate(image_b64_list):
            try:
                image = decode_image(image_b64)
                print(f"Processing image {img_index + 1}: {image.size} pixels, mode: {image.mode}")
            except ValueError as e:
                final_output.append({"error": f"Invalid image {img_index + 1}: {str(e)}"})
                continue

            image_path = f"output/uploaded_image_{img_index}.jpg"
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_path, quality=95)

            # Preprocess image for better detection
            processed_path = preprocess_image_for_detection(image_path)

            try:
                print(f"Running detection on image {img_index + 1}...")
                result = detector.detect(processed_path)
                print(f"Detection result type: {type(result)}")
                
                boxes = detector.get_bounding_boxes(result)
                classes = detector.get_class_names(result)
                print(f"Raw detections - Boxes: {len(boxes) if boxes else 0}, Classes: {len(classes) if classes else 0}")
                
                # Validate and filter results
                boxes, classes = validate_detection_results(result, boxes, classes)
                print(f"Filtered detections - Boxes: {len(boxes)}, Classes: {len(classes)}")
                
                objects = list(set(classes)) if classes else []
                print(f"Unique objects found: {objects}")
                
                if not objects:
                    # Try alternative detection strategies
                    print(f"No objects detected in image {img_index + 1}. Trying fallback strategies...")
                    
                    # Strategy 1: Try with original unprocessed image
                    if processed_path != image_path:
                        print("Trying with original image...")
                        result = detector.detect(image_path)
                        boxes = detector.get_bounding_boxes(result)
                        classes = detector.get_class_names(result)
                        objects = list(set(classes)) if classes else []
                        print(f"Original image results: {objects}")
                    
                    # Strategy 2: Lower confidence threshold or try different model settings
                    if not objects and hasattr(detector, 'set_confidence_threshold'):
                        print("Trying with lower confidence threshold...")
                        detector.set_confidence_threshold(0.1)
                        result = detector.detect(image_path)
                        boxes = detector.get_bounding_boxes(result)
                        classes = detector.get_class_names(result)
                        objects = list(set(classes)) if classes else []
                        detector.set_confidence_threshold(0.3)  # Reset
                        print(f"Lower threshold results: {objects}")
                    
                    # Strategy 3: Add generic object if still nothing found
                    if not objects:
                        print("Adding generic 'object' detection...")
                        objects = ["object"]
                        # Create a full image bounding box
                        img_width, img_height = image.size
                        boxes = [[0, 0, img_width, img_height]]
                        classes = ["object"]
                
                if not objects:
                    final_output.append({
                        "error": f"No objects detected in image {img_index + 1}",
                        "debug_info": {
                            "image_size": image.size,
                            "image_mode": image.mode,
                            "processed_path": processed_path,
                            "detection_attempted": True
                        }
                    })
                    continue
                    
            except Exception as e:
                traceback.print_exc()
                final_output.append({"error": f"Detection failed for image {img_index + 1}: {str(e)}"})
                continue

            try:
                gpt_descriptions = []
                sentences, meanings, synonyms_flat = [], [], []
                for obj in objects:
                    desc = sentence_gen.describe_object(obj)
                    gpt_descriptions.append(desc)
                    sentences.append(desc.get("chatgpt_sentence", ""))
                    meanings.append(desc.get("chatgpt_meaning", ""))
                    # Ensure synonyms is always a list
                    syns = desc.get("chatgpt_synonyms", [])
                    if not isinstance(syns, list):
                        syns = [str(syns)] if syns else []
                    synonyms_flat.extend(syns)
            except Exception as e:
                traceback.print_exc()
                final_output.append({"error": f"Description generation failed for image {img_index + 1}: {str(e)}"})
                continue

            try:
                translated_object_names = translator.translate(objects)
                translated_sentences = translator.translate(sentences)
                translated_meanings = translator.translate(meanings)
                translated_synonyms_flat = translator.translate(synonyms_flat)
            except Exception as e:
                traceback.print_exc()
                final_output.append({"error": f"Translation failed for image {img_index + 1}: {str(e)}"})
                continue

            try:
                group_objects = []
                idx = 0
                for i, (obj, desc) in enumerate(zip(objects, gpt_descriptions)):
                    # Ensure synonyms is always a list
                    syns = desc.get("chatgpt_synonyms", [])
                    if not isinstance(syns, list):
                        syns = [str(syns)] if syns else []
                    
                    syn_count = len(syns)
                    translated_syns = translated_synonyms_flat[idx:idx + syn_count]
                    idx += syn_count

                    safe_filename = f"{obj.replace(' ', '_').lower()}_{img_index}.jpg"
                    image_crop_path = os.path.join(CROPS_DIR, safe_filename)

                    # Create crop from bounding box
                    cropped_successfully = False
                    for box, cls in zip(boxes, classes):
                        if cls == obj:
                            try:
                                x1, y1, x2, y2 = map(int, box)
                                # Ensure coordinates are within image bounds
                                x1 = max(0, min(x1, image.width))
                                y1 = max(0, min(y1, image.height))
                                x2 = max(x1, min(x2, image.width))
                                y2 = max(y1, min(y2, image.height))
                                
                                if x2 > x1 and y2 > y1:
                                    cropped_img = image.crop((x1, y1, x2, y2))
                                    cropped_img.save(image_crop_path, format='JPEG')
                                    cropped_successfully = True
                                    break
                            except Exception as crop_error:
                                print(f"Error cropping {obj}: {crop_error}")
                    
                    # If cropping failed, use full image
                    if not cropped_successfully:
                        image.save(image_crop_path, format='JPEG')

                    group_objects.append({
                        "word": obj,
                        "translation": translated_object_names[i] if i < len(translated_object_names) else obj,
                        "meaning_en": desc.get("chatgpt_meaning", ""),
                        "meaning_translated": translated_meanings[i] if i < len(translated_meanings) else "",
                        "sentence_en": desc.get("chatgpt_sentence", ""),
                        "sentence_translated": translated_sentences[i] if i < len(translated_sentences) else "",
                        "synonyms_en": syns,
                        "synonyms_translated": translated_syns if translated_syns else [],
                        "image_crop": f"/static/crops/{safe_filename}"
                    })

                # Convert full image to base64 for group preview
                with open(image_path, "rb") as img_f:
                    encoded_image = base64.b64encode(img_f.read()).decode("utf-8")

                final_output.append({
                    "image": f"data:image/jpeg;base64,{encoded_image}",
                    "objects": group_objects
                })

            except Exception as e:
                traceback.print_exc()
                final_output.append({"error": f"Output failed for image {img_index + 1}: {str(e)}"})

        return jsonify(final_output)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)