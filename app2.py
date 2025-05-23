from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import base64
import requests
import os
import traceback
from flask_cors import CORS
from models.object_detector import ObjectDetector
from models.translator import ObjectTranslator
from models.sentence_generator import SentenceGenerator

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

@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")

@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    return send_from_directory("frontend", filename)

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_b64 = data.get("image")
        target_lang = data.get("language", "aka_Latn")
        supported_languages = ["aka_Latn", "en", "fr", "es", "de"]
        if target_lang not in supported_languages:
            return jsonify({"error": f"Unsupported language. Supported: {supported_languages}"}), 400

        translator.set_language(target_lang)

        try:
            image = decode_image(image_b64)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("static/crops"):
            os.makedirs("static/crops")

        image_path = "output/uploaded_image.jpg"
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path, quality=95)

        try:
            result = detector.detect(image_path)
            boxes = detector.get_boxes(result)
            classes = detector.get_class_names(result)
            objects = list(set(classes))
            if not objects:
                return jsonify({"error": "No objects detected in the image"}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Object detection failed: {str(e)}"}), 500

        try:
            gpt_descriptions = []
            sentences, meanings, synonyms_flat = [], [], []
            for obj in objects:
                desc = sentence_gen.describe_object(obj)
                gpt_descriptions.append(desc)
                sentences.append(desc.get("chatgpt_sentence", ""))
                meanings.append(desc.get("chatgpt_meaning", ""))
                syns = desc.get("chatgpt_synonyms", [])
                synonyms_flat.extend(syns)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Description generation failed: {str(e)}"}), 500

        try:
            translated_object_names = translator.translate(objects)
            translated_sentences = translator.translate(sentences)
            translated_meanings = translator.translate(meanings)
            translated_synonyms_flat = translator.translate(synonyms_flat)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500

        output = []
        try:
            idx = 0
            for i, (obj, desc) in enumerate(zip(objects, gpt_descriptions)):
                syn_count = len(desc.get("chatgpt_synonyms", []))
                translated_syns = translated_synonyms_flat[idx:idx + syn_count]
                idx += syn_count

                safe_filename = obj.replace(" ", "_").lower() + ".jpg"
                image_crop_path = os.path.join("static", "crops", safe_filename)
                for box, cls in zip(boxes, classes):
                    if cls == obj:
                        x1, y1, x2, y2 = map(int, box)
                        cropped_img = image.crop((x1, y1, x2, y2))
                        cropped_img.save(image_crop_path, format='JPEG')
                        break

                output.append({
                    "word": obj,
                    "translation": translated_object_names[i],
                    "meaning_en": desc.get("chatgpt_meaning", ""),
                    "meaning_translated": translated_meanings[i],
                    "sentence_en": desc.get("chatgpt_sentence", ""),
                    "sentence_translated": translated_sentences[i],
                    "synonyms_en": desc.get("chatgpt_synonyms", []),
                    "synonyms_translated": translated_syns,
                    "image_crop": f"/static/crops/{safe_filename}"
                })

            return jsonify(output)

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Output preparation failed: {str(e)}"}), 500

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
    app.run(debug=True, host='0.0.0.0', port=5000)
