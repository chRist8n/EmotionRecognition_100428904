try:
    from integral_image import compute_integral_image
    from haar_features import evaluate_window
    from sliding_window import sliding_window
except ImportError:
    from detection.integral_image import compute_integral_image
    from detection.haar_features import evaluate_window
    from detection.sliding_window import sliding_window
from sklearn.cluster import DBSCAN
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

# initial paramters - eps=15, min_samples=3
def cluster_boxes(boxes, eps=10, min_samples=2):
    """
    Clusters small candidate boxes into a larger bounding boxes using Density-based spatial clustering of applications with noise (DBSCAN).

    Parameters:
        boxes(array) : list of (x, y, w, h) small candidate boxes
        eps(float) : maximum distance between points in a cluster
        min_samples(int) : minimum number of boxes to form a cluster

    Returns:
        clustered_boxes(array) : list of (x_min, y_min, w, h) merged boxes
    """
    if len(boxes) == 0:
        return []

    # compute centers of each box
    centers = np.array([[x + w/2, y + h/2] for (x, y, w, h) in boxes])

    # run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_

    clustered_boxes = []

    for label in set(labels):
        if label == -1:
            continue  # noise, ignore
        cluster_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == label]

        # merge boxes into one bounding box
        x_min = min([b[0] for b in cluster_boxes])
        y_min = min([b[1] for b in cluster_boxes])
        x_max = max([b[0] + b[2] for b in cluster_boxes])
        y_max = max([b[1] + b[3] for b in cluster_boxes])

        clustered_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return clustered_boxes

def verify_detections(image, detections):
    if len(detections) == 0:
        return None
    
    best_box = None
    best_score = float("-inf")

    for x, y, w, h in detections:
        patch = image[y:y+h, x:x+w]
        integral = compute_integral_image(patch)
        feature_score = evaluate_window(integral, 0, 0, w, h)

        if feature_score > best_score:
            best_score = feature_score
            best_box = (x, y, w, h)

    return best_box


def get_box_center(box):
    x, y, w, h = box

    center_x = x + w // 2
    center_y = y + h // 2

    return (center_x, center_y)

## NOTE: width=100, height=160 provides a tight crop at >= medium distances but cuts part
#       of the face off at close range (when leaning forwards). for now it gives a
#       generous amount of room but the area will be tightened later in the pipeline.
def build_face_box_from_center(box, image_shape, face_width=100, face_height=150): #center_x, center_y, box_size, image_shape):
    # find the centre of detected feature as well as its width and height (for depth estimation)
    center_x, center_y = get_box_center(box)
    x, y, w, h = box

    # build  a fixed-size box around centre 
    # (and adjust x and y to account for feature sitting at top of head)
    face_x = (center_x - face_width // 2) - 5
    face_y = (center_y - face_height // 2) + 60

    # keep inside image bounds
    img_h, img_w = image_shape
    face_x = max(0, face_x)
    face_y = max(0, face_y)
    face_width = min(face_width, img_w - face_x)
    face_height = min(face_height, img_h - face_y)

    return (face_x, face_y, face_width, face_height)

## Dynamically-sized box
def build_face_box_from_detection(box, image_shape):
    x, y, w, h = box

    # Expand box (tuned ratios, not constants)
    new_x = x - int(0.2 * w)
    new_y = y - int(0.3 * h)

    new_w = int(1.4 * w)
    new_h = int(1.8 * h)  # extend downward more for chin

    # Clamp to image bounds
    img_h, img_w = image_shape
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(new_w, img_w - new_x)
    new_h = min(new_h, img_h - new_y)

    return (new_x, new_y, new_w, new_h)

#DETECT FACES
def detect_faces(image, scales=[32, 48, 64, 96], step=12, threshold=0.9):
    #approximate normalisation for contrast
    image_mean = np.mean(image)
    image_std = np.std(image) if np.std(image) > 0 else 1
    norm_image = (image - image_mean) / image_std  # entire frame normalized

    # compute integral image once for whole frame
    integral = compute_integral_image(norm_image)

    # init detections and scores
    detections = []
    scores = []

    # init threshold for ignoring bottom of screen 
    # (to help prevent distraction by clothes/phones/pets etc)
    img_h, img_w = norm_image.shape
    bottom_ignore_y = int(img_h * 0.8)  # bottom 1/5th

    for scale in scales:
        for (x, y, w, h) in sliding_window(norm_image, scale, step):
            if y + h >= bottom_ignore_y:
                continue    # if any of the box overlaps into deadzone, next.

            # evaluate window on integral image
            score = evaluate_window(integral, x, y, w, h)

            if score > threshold:
                detections.append((x, y, w, h))
                scores.append(score)

    # check for empty detections
    if len(detections) == 0:
        return None

    # apply NMS
    detections = non_max_suppression(detections, scores, iou_threshold=0.3)
    
    # verify detections
    best_box = verify_detections(norm_image, detections)
    if best_box is None:
        return None
    # final_pred = build_face_box_from_center(best_box, image.shape)
    final_pred = build_face_box_from_detection(best_box, image.shape)

    return final_pred

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()