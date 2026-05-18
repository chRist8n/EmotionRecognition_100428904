import math
try:
    from landmarking.cropping import compute_centre
except ImportError:
    from cropping import compute_centre


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def slope(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) / math.pi


def clamp(x, min_val=-3.0, max_val=3.0):
    return max(min(x, max_val), min_val)


def extract_features(points):
    """
    Extracts a fixed-length feature vector from MediaPipe facial landmarks.

    The function converts raw landmark coordinates into a set of 15 engineered
    geometric features representing key facial expression signals.

    These features capture variations in:
    - eye openness and asymmetry
    - eyebrow height, shape, and symmetry
    - mouth opening, width, and curvature
    - global facial structure relationships
    - combined expressive interaction signals

    All features are normalised using face-width scaling to reduce sensitivity
    to scale and distance from the camera.

    Returns:
        list[float]: A 15-dimensional feature vector used as input to the MLP classifier.
    """

    features = []

    face_width = max(dist(points[234], points[454]), 1e-6)

    # ---------------- EYES ---------------- #

    left_eye_open = dist(points[159], points[145]) / (dist(points[33], points[133]) + 1e-6)
    right_eye_open = dist(points[386], points[374]) / (dist(points[362], points[263]) + 1e-6)

    eye_open_mean = (left_eye_open + right_eye_open) / 2
    eye_open_diff = abs(left_eye_open - right_eye_open)

    # ---------------- BROWS ---------------- #

    left_eye_centre = compute_centre(points, [33, 133, 159, 145])
    right_eye_centre = compute_centre(points, [362, 263, 386, 374])

    left_brow = compute_centre(points, [70, 63])
    right_brow = compute_centre(points, [300, 293])

    left_brow_height = (left_eye_centre[1] - left_brow[1]) / face_width
    right_brow_height = (right_eye_centre[1] - right_brow[1]) / face_width

    brow_height_mean = (left_brow_height + right_brow_height) / 2
    brow_asymmetry = abs(left_brow_height - right_brow_height)

    # inner brow shape
    inner_brow = compute_centre(points, [70, 300])
    outer_brow = compute_centre(points, [63, 293])

    inner_brow_height = (left_eye_centre[1] - inner_brow[1]) / face_width
    outer_brow_height = (left_eye_centre[1] - outer_brow[1]) / face_width

    brow_shape = inner_brow_height - outer_brow_height

    # ---------------- MOUTH ---------------- #

    mouth_top = points[13]
    mouth_bottom = points[14]
    mouth_left = points[78]
    mouth_right = points[308]

    mouth_open = dist(mouth_top, mouth_bottom) / face_width
    mouth_width = dist(mouth_left, mouth_right) / face_width

    mouth_curve = (mouth_left[1] + mouth_right[1]) / 2 - mouth_top[1]
    mouth_curve /= (face_width + 1e-6)

    mouth_asymmetry = abs(mouth_left[1] - mouth_right[1]) / face_width

    mouth_corner_drop = (
        ((mouth_left[1] + mouth_right[1]) / 2) - mouth_top[1]
    ) / face_width

    # ---------------- GLOBAL FACE ---------------- #

    mouth_centre = compute_centre(points, [13, 14, 78, 308])
    eye_centre = compute_centre(points, [33, 133, 362, 263])

    eye_mouth_dist = dist(eye_centre, mouth_centre) / face_width

    # ---------------- EXPRESSIVE SIGNALS ---------------- #

    smile_strength = mouth_width * mouth_open

    eye_brow_ratio = eye_open_mean / (brow_height_mean + 1e-6)  #how open are eyes relative to brow height
    mouth_eye_balance = mouth_open - eye_open_mean              #eye vs mouth openness

    face_asymmetry = eye_open_diff + brow_asymmetry + mouth_asymmetry

    # ---------------- FINAL FEATURE VECTOR ---------------- #

    features.extend([
        ## eyes
        eye_open_mean,
        eye_open_diff,

        ## brows
        brow_height_mean,
        brow_asymmetry,
        brow_shape,

        ## mouth
        mouth_open,
        mouth_width,
        mouth_curve,
        mouth_asymmetry,
        mouth_corner_drop,

        # global structure
        eye_mouth_dist,

        ## expressive summary signals
        smile_strength,
        eye_brow_ratio,
        mouth_eye_balance,
        face_asymmetry
    ])

    return features