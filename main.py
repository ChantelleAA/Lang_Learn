import os
from PIL import Image
from models.object_detector import ObjectDetector
from models.translator import ObjectTranslator
from models.sentence_generator import SentenceGenerator
from utils.io import save_image, print_summary

def main(image: Image.Image, target_lang_code="aka_Latn"):
    image_path = save_image(image)
    
    detector = ObjectDetector()
    result = detector.detect(image_path)
    objects = detector.get_class_names(result)
    
    translator = ObjectTranslator(target_lang=target_lang_code)
    translations = translator.translate(objects)
    
    sentence_gen = SentenceGenerator()
    results = []
    for obj in objects:
        sent = sentence_gen.generate(obj)
        valid = sentence_gen.validate(sent)
        score = sentence_gen.score(sent)
        results.append({
            "object": obj,
            "sentence": sent,
            "valid": valid,
            "complexity": score
        })
    
    print_summary(results)
    sentence_gen.close()

if __name__ == "__main__":
    sample_image = Image.open("test_images/apple.jpg")  # Replace with actual image
    main(sample_image, target_lang_code="fra_Latn")
