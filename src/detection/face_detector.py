from integral_image import compute_integral_image
from haar_features import evaluate_window
from sliding_window import sliding_window
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


#DETECT FACES
def detect_faces(image, scales=[32, 48, 64, 96], step=12, threshold=0.9):
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

    # apply NMS and clustering
    detections = non_max_suppression(detections, scores, iou_threshold=0.3)
    return detections
    # predZone = cluster_boxes(detections, eps=15, min_samples=3)
    # return predZone

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()