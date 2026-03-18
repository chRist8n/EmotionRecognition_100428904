from integral_image import compute_integral_image
from haar_features import evaluate_window
from sliding_window import sliding_window
import numpy as np
import sys
import os

#HELPERS ETC
def normalize_window(image, x, y, w, h):
    patch = image[y:y+h, x:x+w].astype(np.float32)
    mean = np.mean(patch)
    std = np.std(patch)
    if std > 0:
        patch = (patch - mean) / std
    else:
        patch = patch - mean
    return patch

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Non-maximum suppression (NMS) helper function.
    Performs NMS on the list of boxes with their scores to filter out
    redundant boxes - only one is wanted per face.

    Parameters:
        boxes : list of (x, y, w, h)
        scores : list of scores corresponding to boxes
        iou_threshold : overlap threshold above which boxes are suppressed

    Returns:
        boxes[keep].tolist() : a list of boxes with overlapping ones filtered out
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # sort descending by score

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep].tolist()


#DETECT FACES
#starting parameters:
# size = 32
# step = 8
# threshold = -1
def detect_faces(image, scales=[32, 48, 64], step=16, threshold=0.31):
    #approximate normalisation for contrast
    image_mean = np.mean(image)
    image_std = np.std(image) if np.std(image) > 0 else 1
    norm_image = (image - image_mean) / image_std  # entire frame normalized

    # compute integral image once for whole frame
    integral = compute_integral_image(norm_image)

    detections = []
    scores = []

    for scale in scales:
        for (x, y, w, h) in sliding_window(norm_image, scale, step):
            # evaluate window on integral image
            score = evaluate_window(integral, x, y, w, h)

            if score > threshold:
                detections.append((x, y, w, h))
                scores.append(score)

    detections = non_max_suppression(detections, scores, iou_threshold=0.3)
    return detections

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()