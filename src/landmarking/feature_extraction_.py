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

    #------------------eye openness------------------#
    # left eye open
    left_eye_vert = dist(points[159], points[145])
    left_eye_horiz = dist(points[33], points[133])
    left_eye_open = clamp(left_eye_vert / (left_eye_horiz + 1e-6))

    # right eye open
    right_eye_vert = dist(points[386], points[374])
    right_eye_horiz = dist(points[362], points[263])
    right_eye_open = clamp(right_eye_vert / (right_eye_horiz + 1e-6))

    # eye squinting
    eye_squint = (left_eye_vert + right_eye_vert) / (left_eye_horiz + right_eye_horiz)
    #------------------------------------------------#

    #---------------------cheeks---------------------#
    # cheek_left = dist(points[33], points[50])
    # cheek_right = dist(points[263], points[280])
    # cheek_raise = -(cheek_left + cheek_right) / 2
    #------------------------------------------------#

    #---------------------mouth----------------------#
    eye_centre = compute_centre(points, [33,133,362,263])
    mouth_centre = compute_centre(points, [13,14,78,308])

    eye_mouth_dist = dist(eye_centre, mouth_centre) / (face_width + 1e-6)

    mouth_vert = dist(points[13], points[14])
    mouth_horiz = dist(points[78], points[308])
    mouth_open = mouth_vert / (mouth_horiz + 1e-6)
    mouth_width = mouth_horiz / (face_width + 1e-6)

    mouth_area_proxy = mouth_vert * mouth_horiz / (face_width**2 + 1e-6)
    lip_compression = 1.0 / (mouth_open + 1e-6)

    mouth_eye_ratio = mouth_open / (eye_mouth_dist + 1e-6)


    
    left_corner = points[78]
    right_corner = points[308]
    top_lip = points[13]

    mouth_corner_avg_y = (left_corner[1] + right_corner[1]) / 2
    mouth_curve = (mouth_corner_avg_y - top_lip[1]) / (mouth_vert + 1e-6)

    mouth_stretch = dist(left_corner, right_corner) / (face_width + 1e-6)

    mouth_corner_diff = abs(left_corner[1] - right_corner[1]) / face_width

    smile_asymmetry = (left_corner[1] - right_corner[1]) / face_width
    smile_intensity = mouth_curve * mouth_width

    smile_activation = (mouth_width - mouth_open)
    #------------------------------------------------#

    #--------------------eyebrows--------------------#
    left_eye_centre = compute_centre(points, [33,133,159,145])
    right_eye_centre = compute_centre(points, [362,263,386,374])

    left_eyebrow = compute_centre(points, [70,63])
    right_eyebrow = compute_centre(points, [300,293])

    # brow height
    left_brow_height = clamp((left_eye_centre[1] - left_eyebrow[1]) / (face_width + 1e-6))
    right_brow_height = clamp((right_eye_centre[1] - right_eyebrow[1]) / (face_width + 1e-6))

    # brow tilt
    left_brow_tilt = clamp(slope(points[70], points[63]))
    right_brow_tilt = clamp(slope(points[300], points[293]))

    avg_eye_openness = (left_eye_open + right_eye_open) / 2
    avg_brow_height = (left_brow_height + right_brow_height) / 2
    brow_eye_ratio = clamp(avg_eye_openness / (avg_brow_height + 1e-6))

    brow_raise = (left_brow_height + right_brow_height) / 2
    brow_asymmetry = abs(left_brow_height - right_brow_height)
    #------------------------------------------------#

    #--------------------symmetry--------------------#
    # upper vs lower lip openness
    upper_lip = dist(points[13], points[0])   # top lip to nose area
    lower_lip = dist(points[14], points[17])  # bottom lip to chin area

    lip_ratio = mouth_open / (mouth_width + 1e-6)

    # neutral score
    neutral_score = (
        abs(mouth_curve) +
        abs(mouth_open) +
        abs(left_brow_height) +
        abs(right_brow_height)
    )

    face_asymmetry = (
        abs(left_eye_open - right_eye_open) +           # <== new version
        abs(left_brow_height - right_brow_height) +
        abs((points[78][1] - points[308][1]) / face_width)
    )
    #------------------------------------------------#

    #------------------------------------------------#
    features.extend([
        left_eye_open,
        right_eye_open,
        eye_squint,
        #smile_activation,
        eye_mouth_dist,
        mouth_open,
        mouth_width,
        #mouth_area_proxy,
        #lip_compression,   #smile asymmetry, smile intensity
        #mouth_eye_ratio,
        mouth_curve,
        mouth_stretch,
        mouth_corner_diff,
        smile_asymmetry,
        smile_intensity,
        left_brow_height,
        right_brow_height,
        left_brow_tilt,
        right_brow_tilt,
        brow_raise,
        brow_asymmetry,
        lip_ratio,
        #eye_diff,
        #eye_y_diff,
        #eye_inner_y_diff,
        #neutral_score,  #maybe remove
        face_asymmetry
    ])
    #------------------------------------------------#


    return features