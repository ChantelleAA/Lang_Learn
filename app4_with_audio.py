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
import tempfile
import logging
from werkzeug.utils import secure_filename
import subprocess
import platform

# Audio processing imports (lightweight alternatives)
try:
    import torch
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available - audio features will be limited")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("SpeechRecognition not available - installing with: pip install SpeechRecognition")

import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CROPS_DIR = os.path.join(STATIC_DIR, "crops")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Create directories
for directory in [CROPS_DIR, UPLOAD_DIR]:
    os.makedirs(directory, exist_ok=True)

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Set device for audio processing
device = "cuda" if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else "cpu"
logger.info(f"Using device: {device}")

# Initialize visual models
try:
    detector = ObjectDetector()
    translator = ObjectTranslator("aka_Latn")
    sentence_gen = SentenceGenerator()
    logger.info("Visual models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing visual models: {e}")
    detector = translator = sentence_gen = None

# Initialize lightweight audio processing
speech_recognizer = None
if SPEECH_RECOGNITION_AVAILABLE:
    try:
        speech_recognizer = sr.Recognizer()
        logger.info("Speech recognition initialized")
    except Exception as e:
        logger.error(f"Error initializing speech recognition: {e}")

# Lightweight translation models (only load when needed)
AUDIO_TRANSLATION_MODELS = {
    "Twi": {
        "model": "facebook/nllb-200-distilled-600M",  # Smaller model
        "tgt_lang": "aka_Latn",
        "use_bos_token": True
    },
    "Hausa": {
        "model": "facebook/nllb-200-distilled-600M",
        "tgt_lang": "hau_Latn",
        "use_bos_token": True
    },
    "Yoruba": {
        "model": "facebook/nllb-200-distilled-600M",
        "tgt_lang": "yor_Latn",
        "use_bos_token": True
    },
}

def setup_ffmpeg():
    """Setup FFmpeg path for audio processing"""
    system = platform.system().lower()
    
    if system == "windows":
        # Common FFmpeg installation paths on Windows
        possible_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Users\{}\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-*\bin\ffmpeg.exe".format(os.environ.get('USERNAME', '')),
        ]
        
        for path in possible_paths:
            if '*' in path:
                # Handle wildcard paths
                import glob
                matches = glob.glob(path)
                if matches:
                    path = matches[0]
            
            if os.path.exists(path):
                ffmpeg_dir = os.path.dirname(path)
                if ffmpeg_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] += os.pathsep + ffmpeg_dir
                logger.info(f"FFmpeg found at: {path}")
                return True
    
    # Try to find ffmpeg in PATH
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("FFmpeg found in PATH")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("FFmpeg not found. Audio conversion may not work.")
        return False

# Setup FFmpeg
ffmpeg_available = setup_ffmpeg()

def allowed_file(filename, file_type):
    if file_type == 'audio':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS
    elif file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    return False

def convert_audio_to_wav(input_path):
    """Convert audio file to WAV format using FFmpeg"""
    if not ffmpeg_available:
        return input_path
    
    try:
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        cmd = [
            'ffmpeg', '-i', input_path, 
            '-acodec', 'pcm_s16le', 
            '-ar', '16000', 
            '-ac', '1',  # mono
            '-y',  # overwrite
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Audio converted successfully to {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return input_path
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return input_path

def transcribe_audio_basic(audio_path):
    """Basic audio transcription using speech_recognition library"""
    if not speech_recognizer:
        return "Speech recognition not available"
    
    try:
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            audio_path = convert_audio_to_wav(audio_path)
        
        with sr.AudioFile(audio_path) as source:
            audio = speech_recognizer.record(source)
        
        # Try multiple recognition services
        try:
            # Google Speech Recognition (free tier)
            text = speech_recognizer.recognize_google(audio)
            logger.info("Transcription completed with Google Speech Recognition")
            return text
        except sr.RequestError:
            try:
                # Fallback to offline recognition if available
                text = speech_recognizer.recognize_sphinx(audio)
                logger.info("Transcription completed with offline Sphinx")
                return text
            except (sr.RequestError, sr.UnknownValueError):
                return "Could not transcribe audio - please try again"
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Transcription failed: {str(e)}"

def split_by_punctuation(text, max_chars=400):
    """Split text into chunks for translation"""
    if not text:
        return []
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent + " "
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks

def translate_audio_text_lightweight(text, language_choice):
    """Lightweight translation using smaller models"""
    if not TRANSFORMERS_AVAILABLE:
        return f"[Translation not available - transformers library not installed]"
        
    if not text or not text.strip():
        return "[No text to translate]"
    
    try:
        if language_choice not in AUDIO_TRANSLATION_MODELS:
            return f"[Translation for {language_choice} not supported]"
            
        model_info = AUDIO_TRANSLATION_MODELS[language_choice]
        model_name = model_info["model"]

        logger.info(f"Loading lightweight translation model for {language_choice}")
        
        # Load model with reduced memory usage
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Split text into smaller chunks
        chunks = split_by_punctuation(text, max_chars=200)  # Smaller chunks
        logger.info(f"Split transcript into {len(chunks)} chunks")

        if not chunks:
            return "[No translation possible]"

        translations = []
        
        # Process chunks individually to save memory
        for chunk in chunks:
            try:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():  # Reduce memory usage
                    if model_info["use_bos_token"]:
                        bos_token_id = tokenizer.convert_tokens_to_ids(model_info["tgt_lang"])
                        outputs = model.generate(
                            **inputs,
                            forced_bos_token_id=bos_token_id,
                            max_length=256,  # Reduced max length
                            num_beams=2,  # Reduced beam size
                            do_sample=False
                        )
                    else:
                        outputs = model.generate(
                            **inputs, 
                            max_length=256,
                            num_beams=2,
                            do_sample=False
                        )

                translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translations.append(translated)
                
            except Exception as e:
                logger.error(f"Error translating chunk: {e}")
                translations.append(f"[Translation error for chunk]")
        
        # Clean up model from memory
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return " ".join(translations)

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return f"[Translation failed: {str(e)}]"

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
    """Preprocess image to improve detection results"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        height, width = img.shape[:2]
        logger.info(f"Original image size: {width}x{height}")
        
        # Resize if too small
        min_side = min(width, height)
        if min_side < 640:
            scale = 640 / min_side
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized to: {new_width}x{new_height}")
        
        # Enhance contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Save preprocessed image
        processed_path = image_path.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_path, enhanced)
        return processed_path
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return image_path

@app.route("/")
def home():
    return send_from_directory("frontend", "index4.html")

@app.route("/process", methods=["POST"])
def process_images():
    """Process images for object detection and translation"""
    if not detector or not translator or not sentence_gen:
        return jsonify({"error": "Visual processing models not available"}), 500
        
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
        
        for img_index, image_b64 in enumerate(image_b64_list):
            try:
                image = decode_image(image_b64)
                logger.info(f"Processing image {img_index + 1}: {image.size} pixels")
            except ValueError as e:
                final_output.append({"error": f"Invalid image {img_index + 1}: {str(e)}"})
                continue

            image_path = os.path.join(UPLOAD_DIR, f"uploaded_image_{img_index}.jpg")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_path, quality=95)

            # Preprocess and detect objects
            processed_path = preprocess_image_for_detection(image_path)

            try:
                result = detector.detect(processed_path)
                boxes = detector.get_bounding_boxes(result)
                classes = detector.get_class_names(result)
                objects = list(set(classes)) if classes else []
                
                # Fallback strategies if no objects detected
                if not objects:
                    logger.info(f"No objects detected in image {img_index + 1}, trying fallbacks...")
                    
                    # Try with original image
                    if processed_path != image_path:
                        result = detector.detect(image_path)
                        boxes = detector.get_bounding_boxes(result)
                        classes = detector.get_class_names(result)
                        objects = list(set(classes)) if classes else []
                    
                    # Add generic object if still nothing found
                    if not objects:
                        objects = ["object"]
                        boxes = [[0, 0, image.width, image.height]]
                        classes = ["object"]
                
                if not objects:
                    final_output.append({
                        "error": f"No objects detected in image {img_index + 1}",
                        "debug_info": {"image_size": image.size, "image_mode": image.mode}
                    })
                    continue
                    
            except Exception as e:
                logger.error(f"Detection failed for image {img_index + 1}: {e}")
                final_output.append({"error": f"Detection failed for image {img_index + 1}: {str(e)}"})
                continue

            # Generate descriptions and translations
            try:
                group_objects = []
                
                for i, obj in enumerate(objects):
                    # Generate description
                    desc = sentence_gen.describe_object(obj)
                    
                    # Ensure synonyms is a list
                    syns = desc.get("chatgpt_synonyms", [])
                    if not isinstance(syns, list):
                        syns = [str(syns)] if syns else []
                    
                    # Translate content
                    translated_name = translator.translate([obj])[0] if translator.translate([obj]) else obj
                    translated_meaning = translator.translate([desc.get("chatgpt_meaning", "")])[0] if desc.get("chatgpt_meaning") else ""
                    translated_sentence = translator.translate([desc.get("chatgpt_sentence", "")])[0] if desc.get("chatgpt_sentence") else ""
                    translated_synonyms = translator.translate(syns) if syns else []

                    # Create crop
                    safe_filename = f"{obj.replace(' ', '_').lower()}_{img_index}.jpg"
                    image_crop_path = os.path.join(CROPS_DIR, safe_filename)

                    # Find and crop object
                    cropped_successfully = False
                    for box, cls in zip(boxes, classes):
                        if cls == obj:
                            try:
                                x1, y1, x2, y2 = map(int, box)
                                x1 = max(0, min(x1, image.width))
                                y1 = max(0, min(y1, image.height))
                                x2 = max(x1, min(x2, image.width))
                                y2 = max(y1, min(y2, image.height))
                                
                                if x2 > x1 and y2 > y1:
                                    cropped_img = image.crop((x1, y1, x2, y2))
                                    cropped_img.save(image_crop_path, format='JPEG')
                                    cropped_successfully = True
                                    break
                            except Exception:
                                pass
                    
                    if not cropped_successfully:
                        image.save(image_crop_path, format='JPEG')

                    group_objects.append({
                        "word": obj,
                        "translation": translated_name,
                        "meaning_en": desc.get("chatgpt_meaning", ""),
                        "meaning_translated": translated_meaning,
                        "sentence_en": desc.get("chatgpt_sentence", ""),
                        "sentence_translated": translated_sentence,
                        "synonyms_en": syns,
                        "synonyms_translated": translated_synonyms,
                        "image_crop": f"/static/crops/{safe_filename}"
                    })

                # Convert full image to base64
                with open(image_path, "rb") as img_f:
                    encoded_image = base64.b64encode(img_f.read()).decode("utf-8")

                final_output.append({
                    "image": f"data:image/jpeg;base64,{encoded_image}",
                    "objects": group_objects
                })

            except Exception as e:
                logger.error(f"Processing failed for image {img_index + 1}: {e}")
                final_output.append({"error": f"Processing failed for image {img_index + 1}: {str(e)}"})

        return jsonify(final_output)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """Process audio for transcription and translation"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        target_language = request.form.get('language', 'Twi')
        
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
            
        if not allowed_file(audio_file.filename, 'audio'):
            return jsonify({"error": "Invalid audio file format"}), 400

        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(UPLOAD_DIR, filename)
        audio_file.save(audio_path)
        
        logger.info(f"Processing audio file: {filename}")

        # Transcribe audio using lightweight method
        try:
            transcription = transcribe_audio_basic(audio_path)
            logger.info("Transcription completed successfully")
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return jsonify({"error": f"Transcription failed: {e}"}), 500

        # Translate transcription
        try:
            if target_language in AUDIO_TRANSLATION_MODELS:
                translation = translate_audio_text_lightweight(transcription, target_language)
            else:
                translation = f"[Translation for {target_language} not supported]"
            logger.info("Translation completed successfully")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            translation = f"Translation error: {e}"

        # Clean up uploaded file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        return jsonify({
            "transcription": transcription,
            "translation": translation,
            "language": target_language
        })

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

if __name__ == "__main__":
    print("=== System Information ===")
    print(f"FFmpeg available: {ffmpeg_available}")
    print(f"Speech Recognition available: {SPEECH_RECOGNITION_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"Device: {device}")
    print("=========================")
    
    if not SPEECH_RECOGNITION_AVAILABLE:
        print("To enable audio transcription, install: pip install SpeechRecognition")
    
    if not ffmpeg_available:
        print("To enable audio conversion, install FFmpeg from: https://ffmpeg.org/download.html")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)