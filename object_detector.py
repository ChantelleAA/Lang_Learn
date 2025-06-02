import os
from ultralytics import YOLO
from PIL import Image
from config import YOLO_MODEL_PATH

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

    def detect(self, image_path):
        results = self.model.predict(image_path, save=True)
        return results[0]  # Return the first result

    def get_class_names(self, result):
        if result.boxes.cls.numel() > 0:
            names = [self.model.names[int(cls)] for cls in result.boxes.cls]
        else:
            names = []
        return names

    def get_bounding_boxes(self, result):
        """Return a list of bounding boxes (x1, y1, x2, y2)"""
        return [tuple(map(int, box)) for box in result.boxes.xyxy.tolist()]

    def get_output_image_path(self):
        save_dir = self.model.predictor.save_dir
        images = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.png'))]
        return os.path.join(save_dir, images[0]) if images else None
