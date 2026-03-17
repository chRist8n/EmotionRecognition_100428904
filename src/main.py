from PIL import Image
from PIL import ImageDraw
import numpy as np

from detection.face_detector import detect_faces

def main():
    #test face detector:

    img = Image.open("data/debugging_images/test_normalized.npy")

    detections = detect_faces(img)
    print("Detections: ", detections)

    # convert back to PIL for drawing
    img_draw = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_draw)

    for (x, y, w, h) in detections:
        draw.rectangle([x, y, x + w, y + h], outline=255)

    img_draw.show()


if __name__ == "__main__":
   main()