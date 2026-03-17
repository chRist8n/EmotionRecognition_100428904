from detection.integral_image import compute_integral_image
from detection.haar_features import haar_feature_horizontal
from detection.sliding_window import sliding_window

#starting parameters:
# size = 32
# step = 8
# threshold = -1
def detect_faces(image, window_size=32, step=8, threshold=-1):
    integral = compute_integral_image(image)

    detections = []

    for (x, y, w, h) in sliding_window(image, window_size, step):
        feature = haar_feature_horizontal(integral, x, y, w, h)

        if feature < threshold:
            detections.append((x, y, w, h))

    return detections

#if __name__ == "__main__":
    #detections = detect_faces(image)

    #print(detections)