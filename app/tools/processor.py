import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self, min_width=300, min_height=200, ocr_langs=['en']):
        self.min_width = min_width
        self.min_height = min_height
        self.reader = easyocr.Reader(ocr_langs)

    @staticmethod
    def display(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def resize(self, img):
        h, w = img.shape[:2]
        if w < self.min_width or h < self.min_height:
            scale = max(self.min_width/w, self.min_height/h)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def full_preprocess(img):
        img = cv2.bitwise_not(img)
        kernel = np.ones((3,3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel_sharp)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv2.erode(img, kernel, iterations=1)
        return img

    def ocr_conf(self, img):
        results = self.reader.readtext(img)
        if not results: 
            return 0.0, []
        confs = [r[2] for r in results]
        return sum(confs)/len(confs), results

    def adaptive_ocr(self, path, display_steps=False, return_image=False, high=0.9, mid=0.7):
        img = cv2.imread(path)
        img = self.resize(img)
        gray = self.to_gray(img)

        conf1, res1 = self.ocr_conf(gray)
        best_conf, best_res, best_img = conf1, res1, gray

        if conf1 >= high:
            # High confidence, return immediately
            pass
        elif conf1 >= mid:
            # Medium confidence: light preprocessing (just gray)
            conf2, res2 = self.ocr_conf(gray)
            if conf2 > best_conf:
                best_conf, best_res, best_img = conf2, res2, gray
        else:
            # Low confidence: full preprocessing
            proc = self.full_preprocess(gray)
            if display_steps: self.display(proc)
            conf2, res2 = self.ocr_conf(proc)
            if conf2 > best_conf:
                best_conf, best_res, best_img = conf2, res2, proc

        if return_image:
            return best_res, best_img
        else:
            return best_res

processor = ImagePreprocessor(min_width=300, min_height=200, ocr_langs=['en'])

# Get both OCR results and the processed image
ocr_results, processed_image = processor.adaptive_ocr(
    "Afternoon_1.jpg",
    display_steps=True,
    return_image=True,
    high=0.9,
    mid=0.7
)

# Print detected text
for bbox, text, conf in ocr_results:
    print(f"{text} ({conf:.2f})")

# Optionally save the processed image
cv2.imwrite("processed_output.jpg", processed_image)
