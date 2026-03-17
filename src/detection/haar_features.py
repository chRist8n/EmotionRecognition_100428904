import numpy as np
from integral_image import compute_integral_image

def rect_sum(integral, x, y, w, h):
    A = integral[y-1, x-1] if x > 0 and y > 0 else 0
    B = integral[y-1, x+w-1] if y > 0 else 0
    C = integral[y+h-1, x-1] if x > 0 else 0
    D = integral[y+h-1, x+w-1]

    return D - B - C + A

def haar_feature_horizontal(integral, x, y, w, h):
    half_w = w // 2

    left_sum = rect_sum(integral, x, y, half_w, h)
    right_sum = rect_sum(integral, x + half_w, y, half_w, h)

    return left_sum - right_sum

def is_face(feature_value, threshold = -1):
    return feature_value < threshold

if __name__ == "__main__":
    img = np.array([
    [1, 2],
    [3, 4]
    ], dtype=np.float32)

    integral = compute_integral_image(img)

    print(haar_feature_horizontal(integral, 0, 0, 2, 2))