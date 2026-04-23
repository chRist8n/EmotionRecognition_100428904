import math
try:
    from cropping import compute_centre
except ImportError:
    from landmarking.cropping import compute_centre


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)


def extract_features(points):
    features = []


    face_width = dist(points[234], points[454])  # between cheeks

    #------------------eye openness------------------#
    left_eye_vert = dist(points[159], points[145])
    left_eye_horiz = dist(points[33], points[133])
    left_eye_open = left_eye_vert / (left_eye_horiz + 1e-6)

    right_eye_vert = dist(points[386], points[374])
    right_eye_horiz = dist(points[362], points[263])
    right_eye_open = right_eye_vert / (right_eye_horiz + 1e-6)
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

    left_brow_height = (left_eye_centre[1] - left_eyebrow[1]) / (face_width + 1e-6)
    right_brow_height = (right_eye_centre[1] - right_eyebrow[1]) / (face_width + 1e-6)

    left_brow_tilt = slope(points[70], points[63])
    right_brow_tilt = slope(points[300], points[293])
    #------------------------------------------------#

    #--------------------symmetry--------------------#
    #openness
    eye_diff = abs(left_eye_open - right_eye_open)

    #positional
    eye_y_diff = abs(left_eye_centre[1] - right_eye_centre[1])
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
        eye_y_diff
    ])
    #------------------------------------------------#

    #------------------key indices-------------------#
    key_indices = [33, 133, 362, 263, 13, 14, 78, 308]

    for i in key_indices:
        features.extend([
            points[i][0] / (face_width + 1e-6),
            points[i][1] / (face_width + 1e-6)
            ])
    #------------------------------------------------#

    return features