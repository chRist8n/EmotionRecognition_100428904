def build_face_box(frame, points):
    #get all x and y coords and find mesh's edges in each direction
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    #define face width and height
    face_w = x_max - x_min
    face_h = y_max - y_min

    #allow a small margin
    margin_x = int(face_w * 0.15)
    margin_y = int(face_h * 0.15)

    #define a bounding box
    x = x_min - margin_x
    y = y_min - margin_y
    w = face_w + 2 * margin_x
    h = face_h + 2 * margin_y

    #clamp box to image bounds
    h_img, w_img = frame.shape[:2]

    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    return (x, y, w, h)

def compute_centre(points, indices):
    """
    Finds a centre point of a set of 2D points by finding
    mean average of the x coordinates and y coordinates.

    Parameters:
        points (list of (float, float)) : a list of (x, y) points

    Returns:
        (float, float) : the centre of the given points
    """
    xs = [points[i][0] for i in indices]
    ys = [points[i][1] for i in indices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))