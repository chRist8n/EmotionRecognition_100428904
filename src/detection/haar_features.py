import numpy as np
import sys
import os

#HELPERS ETC
def rect_sum(integral, x, y, w, h):
    A = integral[y-1, x-1] if x > 0 and y > 0 else 0
    B = integral[y-1, x+w-1] if y > 0 else 0
    C = integral[y+h-1, x-1] if x > 0 else 0
    D = integral[y+h-1, x+w-1]

    return D - B - C + A


#VERTICAL FEATURES:
def haar_feature_vertical(integral, x, y, w, h):
    half_w = w // 2

    left_sum = rect_sum(integral, x, y, half_w, h)
    right_sum = rect_sum(integral, x + half_w, y, half_w, h)

    return left_sum - right_sum
def haar_feature_vertical_2(integral, x, y, w, h):
    third_h = h // 3
    top = rect_sum(integral, x, y, w, third_h)
    middle = rect_sum(integral, x, y + third_h, w, third_h)
    bottom = rect_sum(integral, x, y + 2*third_h, w, third_h)
    return top - middle + bottom
def haar_feature_vertical_head(integral, x, y, w, h):
    """
    Vertical feature biased to the top of the window.
    Top third is hair/forehead, bottom two-thirds is face.
    """
    top_h = h // 3
    bottom_h = h - top_h

    top_sum = rect_sum(integral, x, y, w, top_h)
    bottom_sum = rect_sum(integral, x, y + top_h, w, bottom_h)

    return top_sum - bottom_sum


#HORIZONTAL FEATURES
def haar_feature_horizontal(integral, x, y, w, h):
    half_h = h // 2

    top_sum = rect_sum(integral, x, y, w, half_h)
    bottom_sum = rect_sum(integral, x, y + half_h, w, half_h)

    return top_sum - bottom_sum
def haar_feature_horizontal_2(integral, x, y, w, h):
    third_w = w // 3
    left = rect_sum(integral, x, y, third_w, h)
    middle = rect_sum(integral, x + third_w, y, third_w, h)
    right = rect_sum(integral, x + 2*third_w, y, third_w, h)
    return left - middle + right
def haar_feature_horizontal_head(integral, x, y, w, h):
    """
    Horizontal feature emphasizing top region.
    """
    half_h = h // 2
    top_sum = rect_sum(integral, x, y, w, half_h)
    bottom_sum = rect_sum(integral, x, y + half_h, w, half_h)
    return top_sum - bottom_sum

def haar_feature_4(integral, x, y, w, h):
    half_w = w // 2
    half_h = h // 2
    A = rect_sum(integral, x, y, half_w, half_h)
    B = rect_sum(integral, x + half_w, y, half_w, half_h)
    C = rect_sum(integral, x, y + half_h, half_w, half_h)
    D = rect_sum(integral, x + half_w, y + half_h, half_w, half_h)
    return (A - B - C + D)


#EVALUATE WINDOW
def evaluate_window(integral, x, y, w, h):
    """
    Returns a normalized score for a window using multiple Haar features.
    Higher = more likely a face.

    Parameters:
        integral(array) : Integral of an image
        x, y (float) : x and y coordinates
        w, h (float) : width and height

    Returns:
        score (float) : evaluated score for window
    """
    score = 0.0

    features = [
        #(haar_feature_vertical, 1.0),           #<-- stage right arm
        (haar_feature_horizontal, 1.475),         #<-- tip of the head    (tends to catch background items)
        #(haar_feature_vertical_2, 1.0),         #<-- most of the body/torso
        #(haar_feature_horizontal_2, 1.75),       #<-- most of the body/torso        ^ might overlap but similar performance
        #(haar_feature_4, 1.0),                  #<-- nearly/ocassionally gets the face but gets distracted easily
        #(haar_feature_vertical_head, 0.75),     #<-- top of head
        #(haar_feature_horizontal_head, 1.5)     #<-- top of head (more accurate)
    ]
    for f, weight in features:
        score += weight * f(integral, x, y, w, h) / (w * h)

    return score


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()