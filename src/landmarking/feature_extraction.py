import math
try:
    from cropping import compute_centre
except ImportError:
    from landmarking.cropping import compute_centre

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_features(points):
    features = []

    #------------------eye openness------------------#
    left_eye_vert = dist(points[159], points[145])
    left_eye_horiz = dist(points[33], points[133])
    left_eye_open = left_eye_vert / (left_eye_horiz + 1e-6)

    right_eye_vert = dist(points[386], points[374])
    right_eye_horiz = dist(points[362], points[263])
    right_eye_open = right_eye_vert / (right_eye_horiz + 1e-6)
    #------------------------------------------------#

    #---------------------mouth----------------------#
    mouth_vert = dist(points[13], points[14])
    mouth_horiz = dist(points[78], points[308])
    mouth_open = mouth_vert / (mouth_horiz + 1e-6)
    mouth_width = mouth_horiz
    #------------------------------------------------#

    #--------------------eyebrows--------------------#
    left_eye_centre = compute_centre(points, [33,133,159,145])
    right_eye_centre = compute_centre(points, [362,263,386,374])

    left_eyebrow = compute_centre(points, [70,63])
    right_eyebrow = compute_centre(points, [300,293])

    left_brow_height = left_eye_centre[1] - left_eyebrow[1]
    right_brow_height = right_eye_centre[1] - right_eyebrow[1]
    #------------------------------------------------#

    #--------------------symmetry--------------------#
    eye_diff = abs(left_eye_open - right_eye_open)
    #------------------------------------------------#

    features.extend([
        left_eye_open,
        right_eye_open,
        mouth_open,
        mouth_width,
        left_brow_height,
        right_brow_height,
        eye_diff
    ])

    return features