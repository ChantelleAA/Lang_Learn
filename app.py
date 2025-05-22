from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import base64
import requests
from models.object_detector import ObjectDetector
from models.translator import ObjectTranslator
from models.sentence_generator import SentenceGenerator
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string.split(',')[-1])
    return Image.open(io.BytesIO(image_data))

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

        image_b64 = data["image"]
        target_lang = data.get("language", "aka_Latn")

        image = decode_image(image_b64)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        if not os.path.exists("output"):
            os.makedirs("output")
        image_path = "output/uploaded_image.jpg"
        image.save(image_path)

        detector = ObjectDetector()
        translator = ObjectTranslator(target_lang)
        sentence_gen = SentenceGenerator()

        result = detector.detect(image_path)
        objects = detector.get_class_names(result)
        bboxes = detector.get_bounding_boxes(result)

        if not objects or objects == ["No object detected"]:
            return jsonify({"error": "No objects detected"}), 400

        translations = translator.translate(objects)

        output = []
        for obj, translated, bbox in zip(objects, translations, bboxes):
            gpt_data = sentence_gen.describe_object(obj)
            sentence = gpt_data["chatgpt_sentence"]
            synonyms = gpt_data["chatgpt_synonyms"]
            definition = gpt_data["chatgpt_meaning"]

            translated_sentence = translator.translate([sentence])[0]
            translated_definition = translator.translate([definition])[0]
            translated_synonyms = [translator.translate([syn])[0] for syn in synonyms]

            # Crop image
            cropped_img = image.crop(bbox)
            buffered = io.BytesIO()
            cropped_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_data_url = f"data:image/jpeg;base64,{img_str}"

            output.append({
                "word": obj,
                "translation": translated,
                "meaning_en": definition,
                "meaning_translated": translated_definition,
                "sentence_en": sentence,
                "sentence_translated": translated_sentence,
                "synonyms_en": synonyms,
                "synonyms_translated": translated_synonyms,
                "image_crop": img_data_url
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
