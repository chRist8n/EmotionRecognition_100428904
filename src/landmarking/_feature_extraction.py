import math
try:
    from cropping import compute_centre
except ImportError:
    from landmarking.cropping import compute_centre


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def slope(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) / math.pi


def clamp(x, min_val=-3.0, max_val=3.0):
    return max(min(x, max_val), min_val)


def extract_features(points):
    features = []

    face_width = max(dist(points[234], points[454]), 1e-6)
    face_height = max(dist(points[10], points[152]), 1e-6)

    # ------------------ NORMALISED LANDMARK CORE ------------------ #
    # (replaces a lot of redundant ratios)

    key_points = [
        33, 133, 362, 263,     # eyes corners
        159, 145, 386, 374,     # eyelids
        70, 63, 300, 293,       # brows
        13, 14, 78, 308,        # mouth
        1, 152                  # nose, chin
    ]

    for i in key_points:
        x = (points[i][0] - points[1][0]) / face_width
        y = (points[i][1] - points[1][1]) / face_height
        features.extend([x, y])

    # ------------------ EYES ------------------ #
    left_eye_open = dist(points[159], points[145]) / face_height
    right_eye_open = dist(points[386], points[374]) / face_height

    eye_asymmetry = abs(left_eye_open - right_eye_open)
    avg_eye_open = (left_eye_open + right_eye_open) / 2

    # ------------------ EYEBROWS ------------------ #
    left_brow_height = dist(points[70], points[33]) / face_height
    right_brow_height = dist(points[300], points[263]) / face_height

    brow_asymmetry = abs(left_brow_height - right_brow_height)

    left_brow_tilt = slope(points[70], points[63])
    right_brow_tilt = slope(points[300], points[293])

    avg_brow_tilt = (left_brow_tilt + right_brow_tilt) / 2

    brow_raise = (left_brow_height + right_brow_height) / 2

    # ------------------ MOUTH ------------------ #
    mouth_open = dist(points[13], points[14]) / face_height
    mouth_width = dist(points[78], points[308]) / face_width

    left_corner = points[78]
    right_corner = points[308]
    top_lip = points[13]

    mouth_curve = (left_corner[1] + right_corner[1]) / 2 - top_lip[1]
    mouth_curve /= face_height

    mouth_asymmetry = abs(left_corner[1] - right_corner[1]) / face_height

    smile_intensity = mouth_curve * mouth_width

    # ------------------ GLOBAL GEOMETRY ------------------ #
    eye_distance = dist(points[33], points[263]) / face_width
    nose_to_chin = dist(points[1], points[152]) / face_height

    # ------------------ FEATURE VECTOR ------------------ #
    features.extend([

        # eyes
        left_eye_open,
        right_eye_open,
        eye_asymmetry,
        avg_eye_open,

        # brows
        left_brow_height,
        right_brow_height,
        brow_asymmetry,
        avg_brow_tilt,
        brow_raise,

        # mouth
        mouth_open,
        mouth_width,
        mouth_curve,
        mouth_asymmetry,
        smile_intensity,

        # global geometry
        eye_distance,
        nose_to_chin
    ])

    return features