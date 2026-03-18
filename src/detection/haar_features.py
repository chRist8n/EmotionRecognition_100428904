import sys
import os

def rect_sum(integral, x, y, w, h):
    A = integral[y-1, x-1] if x > 0 and y > 0 else 0
    B = integral[y-1, x+w-1] if y > 0 else 0
    C = integral[y+h-1, x-1] if x > 0 else 0
    D = integral[y+h-1, x+w-1]

    return D - B - C + A

def haar_feature_vertical(integral, x, y, w, h):
    half_w = w // 2

    left_sum = rect_sum(integral, x, y, half_w, h)
    right_sum = rect_sum(integral, x + half_w, y, half_w, h)

    #return left_sum - right_sum
    return (left_sum - right_sum) / (w * h)

def haar_feature_horizontal(integral, x, y, w, h):
    half_h = h // 2

    top_sum = rect_sum(integral, x, y, w, half_h)
    bottom_sum = rect_sum(integral, x, y + half_h, w, half_h)

    #return top_sum - bottom_sum
    return (top_sum - bottom_sum) / (w * h)

def evaluate_window(integral, x, y, w, h):
    score = 0

    score += haar_feature_vertical(integral, x, y, w, h)
    score += haar_feature_horizontal(integral, x, y, w, h)

    return score

# def is_face(feature_value, threshold = -1):
#     return feature_value < threshold

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()