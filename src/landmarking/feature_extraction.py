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
    left_eye_vert = dist(points[159], points[145])
    left_eye_horiz = dist(points[33], points[133])
    left_eye_open = clamp(left_eye_vert / (left_eye_horiz + 1e-6))

    right_eye_vert = dist(points[386], points[374])
    right_eye_horiz = dist(points[362], points[263])
    right_eye_open = clamp(right_eye_vert / (right_eye_horiz + 1e-6))
    #------------------------------------------------#

    #---------------eye to mouth dist----------------#
    eye_centre = compute_centre(points, [33,133,362,263])
    mouth_centre = compute_centre(points, [13,14,78,308])

    eye_mouth_dist = dist(eye_centre, mouth_centre) / (face_width + 1e-6)
    #------------------------------------------------#

    #---------------------mouth----------------------#
    mouth_vert = dist(points[13], points[14])
    mouth_horiz = dist(points[78], points[308])
    mouth_open = mouth_vert / (mouth_horiz + 1e-6)
    mouth_width = mouth_horiz / (face_width + 1e-6)
    #------------------------------------------------#

    #------------------mouth shape-------------------#
    left_corner = points[78]
    right_corner = points[308]
    top_lip = points[13]

    mouth_corner_avg_y = (left_corner[1] + right_corner[1]) / 2
    mouth_curve = (mouth_corner_avg_y - top_lip[1]) / (face_width + 1e-6)
    #------------------------------------------------#

    #--------------------eyebrows--------------------#
    left_eye_centre = compute_centre(points, [33,133,159,145])
    right_eye_centre = compute_centre(points, [362,263,386,374])

    left_eyebrow = compute_centre(points, [70,63])
    right_eyebrow = compute_centre(points, [300,293])

    left_brow_height = clamp((left_eye_centre[1] - left_eyebrow[1]) / (face_width + 1e-6))
    right_brow_height = clamp((right_eye_centre[1] - right_eyebrow[1]) / (face_width + 1e-6))

    left_brow_tilt = clamp(slope(points[70], points[63]))
    right_brow_tilt = clamp(slope(points[300], points[293]))
    #------------------------------------------------#

    #--------------------symmetry--------------------#
    #eye openness
    eye_diff = abs(left_eye_open - right_eye_open)

    #eye positions
    eye_y_diff = abs(left_eye_centre[1] - right_eye_centre[1])
    eye_inner_y_diff = abs((points[159][1] - points[386][1]) / face_width)
    eye_x_diff = abs((points[33][0] - points[362][0]) / face_width)

    #mouth
    mouth_asym = abs(points[78][1] - points[308][1]) / (face_width + 1e-6)

    #brows
    brow_diff = abs(left_brow_height - right_brow_height)
    #------------------------------------------------#

    #------------------------------------------------#
    features.extend([
        left_eye_open,
        right_eye_open,
        eye_mouth_dist,
        mouth_open,
        mouth_width,
        mouth_curve,
        left_brow_height,
        right_brow_height,
        left_brow_tilt,
        right_brow_tilt,
        eye_diff,
        eye_y_diff,
        eye_inner_y_diff,
        eye_x_diff,
        mouth_asym,
        brow_diff
    ])
    #------------------------------------------------#

    #------------------key indices-------------------#
    key_indices = [33, 133, 362, 263, 13, 14, 78, 308]

    for i in key_indices:
        x = points[i][0] / face_width
        y = points[i][1] / face_width
        features.extend([x, y])

    # eye vertical offset -> features.append((points[133][1] - points[263][1]) / face_width)
    #------------------------------------------------#

    return features