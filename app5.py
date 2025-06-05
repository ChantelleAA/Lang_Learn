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
from download_model import download_model
import json
import time
from threading import Thread
import queue

download_model()

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
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

# Create directories
for directory in [CROPS_DIR, UPLOAD_DIR, VIDEO_DIR]:
    os.makedirs(directory, exist_ok=True)

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Set device for audio processing
device = "cuda" if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else "cpu"
logger.info(f"Using device: {device}")

# Global variables for real-time processing
realtime_sessions = {}
video_processing_queue = queue.Queue()

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

# Cultural context database - expand this with more cultural information
CULTURAL_CONTEXTS = {
    "aka_Latn": {  # Twi/Akan
        "greetings": {
            "hello": "Cultural Context: In Akan culture, greetings are very important and show respect. The time of day affects the greeting used.",
            "goodbye": "Cultural Context: Akan farewells often include blessings and wishes for safe journey or good health."
        },
        "food": {
            "fufu": "Cultural Context: Fufu is a staple food in Ghana, traditionally eaten with hands and shared from a common bowl, symbolizing unity.",
            "banku": "Cultural Context: Banku is fermented corn and cassava dough, often eaten with soup or stew, representing traditional Ghanaian cuisine."
        },
        "animals": {
            "elephant": "Cultural Context: In Akan tradition, the elephant symbolizes wisdom, strength, and good luck. It's featured in many proverbs.",
            "spider": "Cultural Context: Kwaku Anansi (spider) is a central figure in Akan folklore, representing wisdom and storytelling."
        },
        "objects": {
            "kente": "Cultural Context: Kente cloth is sacred to the Akan people, with each pattern and color having specific meanings and historical significance.",
            "drum": "Cultural Context: Drums are vital in Akan culture for communication, ceremonies, and connecting with ancestors."
        }
    },
    "yor_Latn": {  # Yoruba
        "greetings": {
            "hello": "Cultural Context: Yoruba greetings are elaborate and show respect for age and status in society.",
            "goodbye": "Cultural Context: Yoruba farewells often include prayers and blessings for prosperity."
        },
        "food": {
            "jollof": "Cultural Context: Jollof rice is a communal dish that brings families together, often prepared for celebrations.",
            "plantain": "Cultural Context: Plantain is versatile in Yoruba cuisine, symbolizing abundance and prosperity."
        }
    },
    "swh_Latn": {  # Swahili
        "greetings": {
            "hello": "Cultural Context: Swahili greetings often inquire about family and health, showing community care.",
            "goodbye": "Cultural Context: Swahili farewells emphasize peace (amani) and Allah's protection."
        }
    }
}

# Enhanced language mappings for bidirectional translation
LANGUAGE_MAPPINGS = {
    "aka_Latn": {"name": "Twi (Akan)", "code": "ak", "flag": "🇬🇭"},
    "yor_Latn": {"name": "Yoruba", "code": "yo", "flag": "🇳🇬"},
    "hau_Latn": {"name": "Hausa", "code": "ha", "flag": "🇳🇬"},
    "swh_Latn": {"name": "Swahili", "code": "sw", "flag": "🇰🇪"},
    "fra_Latn": {"name": "French", "code": "fr", "flag": "🇫🇷"},
    "spa_Latn": {"name": "Spanish", "code": "es", "flag": "🇪🇸"},
    "arb_Arab": {"name": "Arabic", "code": "ar", "flag": "🇸🇦"},
    "eng_Latn": {"name": "English", "code": "en", "flag": "🇺🇸"}
}

# Lightweight translation models for bidirectional translation
BIDIRECTIONAL_TRANSLATION_MODELS = {
    "to_english": {
        "model": "facebook/nllb-200-distilled-600M",
        "target": "eng_Latn"
    },
    "from_english": {
        "model": "facebook/nllb-200-distilled-600M", 
        "target_dynamic": True  # Target changes based on user selection
    }
}

def generate_cultural_context(word, language_code, category="general"):
    """
    Generate cultural context for translated words
    
    Args:
        word (str): The word to generate context for
        language_code (str): Target language code
        category (str): Category of the word (food, animals, objects, etc.)
    
    Returns:
        str: Cultural context information
    """
    try:
        # Check if we have specific cultural context for this language
        if language_code in CULTURAL_CONTEXTS:
            lang_contexts = CULTURAL_CONTEXTS[language_code]
            
            # Try to find specific context for the word
            word_lower = word.lower()
            
            # Check in specific categories
            for cat_name, contexts in lang_contexts.items():
                if word_lower in contexts:
                    return contexts[word_lower]
            
            # Generate generic cultural context based on category
            if category == "food":
                return f"Cultural Context: In {LANGUAGE_MAPPINGS.get(language_code, {}).get('name', 'this culture')}, food items like '{word}' often have special preparation methods and cultural significance in family gatherings."
            elif category == "animals":
                return f"Cultural Context: Animals like '{word}' may have symbolic meanings or appear in traditional stories and proverbs in {LANGUAGE_MAPPINGS.get(language_code, {}).get('name', 'this culture')}."
            elif category == "objects":
                return f"Cultural Context: Objects like '{word}' may have traditional uses or cultural importance in {LANGUAGE_MAPPINGS.get(language_code, {}).get('name', 'this culture')}."
            else:
                return f"Cultural Context: This word has cultural significance in {LANGUAGE_MAPPINGS.get(language_code, {}).get('name', 'this culture')}. Understanding its cultural usage enriches communication."
        
        # Fallback for languages without specific contexts
        return f"Cultural Context: Learning about '{word}' in different cultures helps build cross-cultural understanding and appreciation."
    
    except Exception as e:
        logger.error(f"Error generating cultural context: {e}")
        return "Cultural Context: Understanding cultural context enhances language learning and communication."

def translate_bidirectional(text, source_lang, target_lang):
    """
    Translate text between any two supported languages
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code
        target_lang (str): Target language code
    
    Returns:
        str: Translated text
    """
    if not TRANSFORMERS_AVAILABLE:
        return f"[Translation not available - transformers library not installed]"
        
    if not text or not text.strip():
        return "[No text to translate]"
    
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        
        logger.info(f"Translating from {source_lang} to {target_lang}")
        
        # Load model with reduced memory usage
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate translation
        with torch.no_grad():
            target_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=256,
                num_beams=2,
                do_sample=False
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up model from memory
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return translated

    except Exception as e:
        logger.error(f"Bidirectional translation failed: {e}")
        return f"[Translation failed: {str(e)}]"

def process_video_frame(frame, session_id, target_lang):
    """
    Process a single video frame for object detection
    
    Args:
        frame: OpenCV frame
        session_id: Unique session identifier
        target_lang: Target language for translation
    
    Returns:
        dict: Detection results with translations
    """
    try:
        # Save frame temporarily
        temp_path = os.path.join(UPLOAD_DIR, f"frame_{session_id}_{int(time.time())}.jpg")
        cv2.imwrite(temp_path, frame)
        
        # Detect objects
        result = detector.detect(temp_path)
        boxes = detector.get_bounding_boxes(result)
        classes = detector.get_class_names(result)
        
        if not classes:
            return {"objects": [], "timestamp": time.time()}
        
        # Process detected objects
        objects = []
        unique_classes = list(set(classes))
        
        for obj in unique_classes:
            # Generate description and cultural context
            desc = sentence_gen.describe_object(obj)
            cultural_context = generate_cultural_context(obj, target_lang, "general")
            
            # Translate content
            translated_name = translator.translate([obj])[0] if translator.translate([obj]) else obj
            translated_meaning = translator.translate([desc.get("chatgpt_meaning", "")])[0] if desc.get("chatgpt_meaning") else ""
            
            objects.append({
                "word": obj,
                "translation": translated_name,
                "meaning_en": desc.get("chatgpt_meaning", ""),
                "meaning_translated": translated_meaning,
                "cultural_context": cultural_context,
                "confidence": 0.8,  # You can get actual confidence from detector
                "bbox": boxes[classes.index(obj)] if obj in classes else None
            })
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "objects": objects,
            "timestamp": time.time(),
            "frame_size": frame.shape[:2]
        }
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return {"objects": [], "error": str(e), "timestamp": time.time()}

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
    elif file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
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

def extract_audio_from_video(video_path):
    """Extract audio track from video file"""
    if not ffmpeg_available:
        return None
    
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # no video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Audio extracted to {audio_path}")
            return audio_path
        else:
            logger.error(f"Audio extraction failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return None

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
        # Map language choice to NLLB code
        lang_map = {
            "Twi": "aka_Latn",
            "Hausa": "hau_Latn", 
            "Yoruba": "yor_Latn",
            "Tiv": "tiv_Latn"  # Add more as needed
        }
        
        target_lang = lang_map.get(language_choice, "aka_Latn")
        return translate_bidirectional(text, "eng_Latn", target_lang)
            
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
    return send_from_directory("frontend", "index5.html")

@app.route("/get_languages", methods=["GET"])
def get_languages():
    """Get available languages for translation"""
    return jsonify(LANGUAGE_MAPPINGS)

@app.route("/translate_text", methods=["POST"])
def translate_text_route():
    """
    Bidirectional text translation endpoint
    
    Expected JSON:
    {
        "text": "text to translate",
        "source_lang": "eng_Latn", 
        "target_lang": "aka_Latn"
    }
    """
    try:
        data = request.get_json()
        text = data.get("text", "")
        source_lang = data.get("source_lang", "eng_Latn")
        target_lang = data.get("target_lang", "aka_Latn")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
            
        # Perform bidirectional translation
        translated_text = translate_bidirectional(text, source_lang, target_lang)
        
        # Generate cultural context
        cultural_context = generate_cultural_context(text, target_lang)
        
        return jsonify({
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "cultural_context": cultural_context
        })
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route("/start_realtime", methods=["POST"])
def start_realtime():
    """
    Start real-time object detection session
    
    Expected JSON:
    {
        "target_lang": "aka_Latn",
        "detection_interval": 2  // seconds between detections
    }
    """
    try:
        data = request.get_json()
        target_lang = data.get("target_lang", "aka_Latn")
        detection_interval = data.get("detection_interval", 2)
        
        # Generate unique session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # Store session configuration
        realtime_sessions[session_id] = {
            "target_lang": target_lang,
            "detection_interval": detection_interval,
            "active": True,
            "last_detection": 0,
            "objects_history": []
        }
        
        return jsonify({
            "session_id": session_id,
            "status": "Real-time detection started",
            "target_language": target_lang
        })
        
    except Exception as e:
        logger.error(f"Error starting real-time detection: {e}")
        return jsonify({"error": f"Failed to start real-time detection: {str(e)}"}), 500

@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Process a single frame from real-time camera feed
    
    Expected form data:
    - session_id: Session identifier
    - frame: Base64 encoded image frame
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        frame_data = data.get("frame")
        
        if session_id not in realtime_sessions:
            return jsonify({"error": "Invalid session ID"}), 400
            
        session = realtime_sessions[session_id]
        
        # Check if enough time has passed since last detection
        current_time = time.time()
        if current_time - session["last_detection"] < session["detection_interval"]:
            return jsonify({"status": "skipped", "reason": "too_frequent"})
        
        # Decode frame
        try:
            image = decode_image(frame_data)
            frame_array = np.array(image)
            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                frame_cv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            else:
                frame_cv = frame_array
        except Exception as e:
            return jsonify({"error": f"Invalid frame data: {e}"}), 400
        
        # Process frame
        results = process_video_frame(frame_cv, session_id, session["target_lang"])
        
        # Update session
        session["last_detection"] = current_time
        session["objects_history"].append(results)
        
        # Keep only last 10 detections in memory
        if len(session["objects_history"]) > 10:
            session["objects_history"] = session["objects_history"][-10:]
        
        return jsonify({
            "session_id": session_id,
            "results": results,
            "timestamp": current_time
        })
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return jsonify({"error": f"Frame processing failed: {str(e)}"}), 500

@app.route("/stop_realtime/<session_id>", methods=["POST"])
def stop_realtime(session_id):
    """Stop real-time detection session"""
    try:
        if session_id in realtime_sessions:
            session = realtime_sessions[session_id]
            session["active"] = False
            
            # Get session summary
            total_detections = len(session["objects_history"])
            unique_objects = set()
            for detection in session["objects_history"]:
                for obj in detection.get("objects", []):
                    unique_objects.add(obj["word"])
            
            summary = {
                "session_id": session_id,
                "total_detections": total_detections,
                "unique_objects": len(unique_objects),
                "objects_detected": list(unique_objects),
                "status": "Session ended"
            }
            
            # Clean up session after a delay (optional)
            # del realtime_sessions[session_id]
            
            return jsonify(summary)
        else:
            return jsonify({"error": "Session not found"}), 404
            
    except Exception as e:
        logger.error(f"Error stopping real-time detection: {e}")
        return jsonify({"error": f"Failed to stop session: {str(e)}"}), 500

@app.route("/process_video", methods=["POST"])
def process_video():
    """
    Process uploaded video file for object detection and translation
    
    Expected form data:
    - video: Video file
    - target_lang: Target language for translation
    - extract_audio: Whether to extract and translate audio (optional)
    """
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        target_lang = request.form.get('target_lang', 'aka_Latn')
        extract_audio = request.form.get('extract_audio', 'false').lower() == 'true'
        
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
            
        if not allowed_file(video_file.filename, 'video'):
            return jsonify({"error": "Invalid video file format"}), 400

        # Save uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(VIDEO_DIR, filename)
        video_file.save(video_path)
        
        logger.info(f"Processing video file: {filename}")

        # Initialize results
        results = {
            "video_info": {},
            "visual_detections": [],
            "audio_transcription": None,
            "audio_translation": None
        }
        
        # Process video frames
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        results["video_info"] = {
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
            "filename": filename
        }
        
        # Process every nth frame to avoid overwhelming
        frame_skip = max(1, fps // 2)  # Process 2 frames per second
        frame_count = 0
        processed_frames = 0
        
        logger.info(f"Processing video: {fps} FPS, {total_frames} frames, processing every {frame_skip} frames")
        
        while cap.isOpened() and processed_frames < 20:  # Limit to 20 frames for demo
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                timestamp = frame_count / fps
                frame_results = process_video_frame(frame, f"video_{int(timestamp)}", target_lang)
                frame_results["timestamp"] = timestamp
                
                if frame_results.get("objects"):
                    results["visual_detections"].append(frame_results)
                    processed_frames += 1
                    
            frame_count += 1
        
        cap.release()
        
        # Process audio if requested
        if extract_audio:
            try:
                audio_path = extract_audio_from_video(video_path)
                if audio_path:
                    # Transcribe audio
                    transcription = transcribe_audio_basic(audio_path)
                    results["audio_transcription"] = transcription
                    
                    # Translate audio
                    if transcription and not transcription.startswith("Could not"):
                        translation = translate_audio_text_lightweight(transcription, target_lang)
                        results["audio_translation"] = translation
                    
                    # Clean up audio file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        
            except Exception as e:
                logger.error(f"Audio processing failed: {e}")
                results["audio_error"] = str(e)
        
        # Clean up video file
        if os.path.exists(video_path):
            os.remove(video_path)
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

 
@app.route("/process_audio", methods=["POST"])
def process_audio():
    """Process audio for transcription and translation - Enhanced with bidirectional translation"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        target_language = request.form.get('language', 'Twi')
        source_language = request.form.get('source_language', 'English')  # New parameter
        
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

        # Translate transcription with cultural context
        try:
            if target_language in ["Twi", "Hausa", "Yoruba", "Tiv"]:
                translation = translate_audio_text_lightweight(transcription, target_language)
                
                # Generate cultural context for the transcribed content
                cultural_context = generate_cultural_context(
                    transcription[:50],  # First 50 chars for context
                    target_language.lower() + "_Latn" if target_language != "Tiv" else "tiv_Latn",
                    "general"
                )
            else:
                translation = f"[Translation for {target_language} not supported]"
                cultural_context = ""
                
            logger.info("Translation completed successfully")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            translation = f"Translation error: {e}"
            cultural_context = ""

        # Clean up uploaded file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        return jsonify({
            "transcription": transcription,
            "translation": translation,
            "language": target_language,
            "source_language": source_language,
            "cultural_context": cultural_context  # New field
        })

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

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

            processed_path = preprocess_image_for_detection(image_path)

            try:
                result = detector.detect(processed_path)
                boxes = detector.get_bounding_boxes(result)
                classes = detector.get_class_names(result)
                objects = list(set(classes)) if classes else []
                
                if not objects:
                    objects = ["object"]
                    boxes = [[0, 0, image.width, image.height]]
                    classes = ["object"]
                    
            except Exception as e:
                logger.error(f"Detection failed for image {img_index + 1}: {e}")
                final_output.append({"error": f"Detection failed for image {img_index + 1}: {str(e)}"})
                continue

            try:
                group_objects = []
                
                for obj in objects:
                    desc = sentence_gen.describe_object(obj)
                    syns = desc.get("chatgpt_synonyms", [])
                    if not isinstance(syns, list):
                        syns = [str(syns)] if syns else []
                    
                    translated_name = translator.translate([obj])[0] if translator.translate([obj]) else obj
                    translated_meaning = translator.translate([desc.get("chatgpt_meaning", "")])[0] if desc.get("chatgpt_meaning") else ""
                    translated_sentence = translator.translate([desc.get("chatgpt_sentence", "")])[0] if desc.get("chatgpt_sentence") else ""
                    translated_synonyms = translator.translate(syns) if syns else []

                    safe_filename = f"{obj.replace(' ', '_').lower()}_{img_index}.jpg"
                    image_crop_path = os.path.join(CROPS_DIR, safe_filename)

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
    print("=== Enhanced System Information ===")
    print(f"FFmpeg available: {ffmpeg_available}")
    print(f"Speech Recognition available: {SPEECH_RECOGNITION_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"Device: {device}")
    print(f"Supported languages: {list(LANGUAGE_MAPPINGS.keys())}")
    print("=== New Features ===")
    print("✓ Real-time object detection")
    print("✓ Video processing with audio extraction")
    print("✓ Bidirectional translation")
    print("✓ Cultural context generation")
    print("✓ Enhanced language support")
    print("========================")
    
    if not SPEECH_RECOGNITION_AVAILABLE:
        print("To enable audio transcription, install: pip install SpeechRecognition")
    
    if not ffmpeg_available:
        print("To enable audio/video conversion, install FFmpeg from: https://ffmpeg.org/download.html")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)