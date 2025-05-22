from flask import Flask, request, jsonify, send_from_directory, stream_with_context, Response
from PIL import Image
import io
import base64
import requests
import os
from flask_cors import CORS
from models.object_detector import ObjectDetector
from models.translator import ObjectTranslator
from models.sentence_generator import SentenceGenerator

app = Flask(__name__)
CORS(app)

# Initialize components globally
detector = ObjectDetector()
translator = ObjectTranslator("aka_Latn")
sentence_gen = SentenceGenerator()

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string.split(',')[-1])
    return Image.open(io.BytesIO(image_data))

# Serve frontend
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
        if target_lang not in ["aka_Latn", "en", "fr", "es", "de"]:
            return jsonify({"error": "Unsupported language"}), 400

        # Update target language dynamically
        translator.set_language(target_lang)

        # Decode and save image
        image = decode_image(image_b64)
        if not os.path.exists("output"):
            os.makedirs("output")
        image_path = "output/uploaded_image.jpg"
        image.save(image_path)

        # Detect objects
        result = detector.detect(image_path)
        objects = list(set(detector.get_class_names(result)))
        if not objects:
            return jsonify({"error": "No objects detected"}), 400

        # GPT descriptions
        gpt_descriptions = [sentence_gen.describe_object(obj) for obj in objects]

        # Build one big string to translate
        blocks = []
        for obj, desc in zip(objects, gpt_descriptions):
            block = f"{obj}|||{desc['chatgpt_sentence']}|||{desc['chatgpt_meaning']}|||{'|||'.join(desc['chatgpt_synonyms'])}"
            blocks.append(block)
        batched_text = "\n<<>>\n".join(blocks)

        # Translate all at once
        translated_batched = translator.translate([batched_text])[0]
        translated_lines = translated_batched.split("\n<<>>\n")

        # Translate object names separately
        translated_object_names = translator.translate(objects)

        output = []
        for i, (obj, desc, translated_line) in enumerate(zip(objects, gpt_descriptions, translated_lines)):
            parts = translated_line.split("|||")
            if len(parts) < 3:
                continue

            output.append({
                "word": obj,
                "translation": translated_object_names[i],
                "meaning_en": desc["chatgpt_meaning"],
                "meaning_translated": parts[2],
                "sentence_en": desc["chatgpt_sentence"],
                "sentence_translated": parts[1],
                "synonyms_en": desc["chatgpt_synonyms"],
                "synonyms_translated": parts[3:]
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
