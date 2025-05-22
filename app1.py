# importation of libraries
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

def fetch_word_info(word):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(url)
        data = response.json()
        meanings = data[0].get("meanings", [])
        definitions = meanings[0]['definitions'][0]['definition'] if meanings else "No definition found"
        synonyms = meanings[0]['definitions'][0].get('synonyms', []) if meanings else []
        return definitions, synonyms
    except Exception as e:
        return "Definition not found", []

# Serve frontend
@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")

@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    return send_from_directory("frontend", filename)

# Process API
@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        print(data)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "Invalid image data"}), 400
        target_lang = data.get("language", "aka_Latn")
        if target_lang not in ["aka_Latn", "en", "fr", "es", "de"]:
            return jsonify({"error": "Unsupported language"}), 400

        # Decode image
        image = decode_image(image_b64)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        image_path = "output/uploaded_image.jpg"
        if not os.path.exists("output"):
            os.makedirs("output")
        image.save(image_path)
        print(f"Image saved at {image_path}")
        # Check if image is valid
        if not os.path.isfile(image_path):
            return jsonify({"error": "Image not found"}), 404

        # Initialize components
        detector = ObjectDetector()
        translator = ObjectTranslator(target_lang)
        sentence_gen = SentenceGenerator()


        # Detect & Translate
        result = detector.detect(image_path)

        # objects = detector.get_class_names(result)
        objects = list(set(detector.get_class_names(result)))


        print(f"Detected objects: {objects}")

        if not objects:
            return jsonify({"error": "No objects detected"}), 400
        print(f"Detected objects: {objects}")
        print(f"Now translating {objects} to {target_lang}")

        translations = translator.translate(objects)

        print(f"Translated objects: {translations}")

        # Sentence + Definitions
        output = []
        # Sentence + Definitions
        output = []
        for obj, translated in zip(objects, translations):
            gpt_data = sentence_gen.describe_object(obj)
            print(f"GPT data: {gpt_data}")
            print(f"type(gpt_data): {type(gpt_data)}")

            sentence = gpt_data["chatgpt_sentence"]
            synonyms = gpt_data["chatgpt_synonyms"]
            definition = gpt_data["chatgpt_meaning"]

            translated_sentence = translator.translate([sentence])[0]
            translated_definition = translator.translate([definition])[0]
            translated_synonyms = [translator.translate([syn])[0] for syn in synonyms]

            output.append({
                "word": obj,
                "translation": translated,
                "meaning_en": definition,
                "meaning_translated": translated_definition,
                "sentence_en": sentence,
                "sentence_translated": translated_sentence,
                "synonyms_en": synonyms,
                "synonyms_translated": translated_synonyms
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)