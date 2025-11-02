import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, weights, conf=0.6, imgsz=1280, device=None):
        """
        Initialize YOLO model and configs.
        Args:
            weights (str): Path to YOLO weights
            conf (float): Confidence threshold
            imgsz (int): Image size for YOLO
            device (int/str, optional): GPU index or 'cpu'
        """
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def get_boxes(self, image: np.ndarray):
        """
        Run YOLO inference on an image array.
        Returns: list of (x1, y1, x2, y2, conf)
        """
        res = self.model.predict(
            image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )

        b = getattr(res[0], "boxes", None)
        if b is None or len(b) == 0:
            return []

        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        return [tuple(map(int, xy)) + (float(c),) for xy, c in zip(xyxy, confs)]

    def crop(self, image: np.ndarray, original_name="image", save_dir=None):
        """
        Crop detected plates from an image.

        Returns: list of dict [{"crop": np.ndarray, "bbox": (x1,y1,x2,y2,conf)}]
        """
        boxes = self.get_boxes(image)
        results = []

        for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
            crop = image[y1:y2, x1:x2]
            results.append({"crop": crop, "bbox": (x1, y1, x2, y2, conf)})

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                name, _ = os.path.splitext(os.path.basename(original_name))
                save_path = os.path.join(save_dir, f"{name}_{i}.jpg")
                cv2.imwrite(save_path, crop)

        if save_dir:
            print(f"Saved {len(results)} crops from '{original_name}' to '{save_dir}'")

        return results

    def close(self):
        """Release YOLO model and clear GPU memory."""
        if self.model:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    weights = "../v3_best.pt"
    image_path = "../images/10.jpg"
    save_dir = "temp/"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector = PlateDetector(weights, conf=0.6)
    detector.crop(img, original_name=image_path, save_dir=save_dir)
    detector.close()
