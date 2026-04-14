import sys
import os
import cv2
import numpy as np



def get_cluster_bounds(cluster):
    # # debug:
    # if not isinstance(cluster, dict):
    #     raise TypeError(f"Expected cluster dict, got {type(cluster)}")

    # if "contours" not in cluster:
    #     raise KeyError(f"Malformed cluster keys: {cluster.keys()}")

    all_points = np.vstack(cluster["contours"])
    return cv2.boundingRect(all_points)


def find_contours(face_crop):
    # set threshold for contour detection
    _, thresh = cv2.threshold(face_crop, 65, 255, cv2.THRESH_BINARY_INV)

    # detect contours from thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def filter_contours(face_crop, raw_contours):
    # filter out noise
    contours = []
    w, h = face_crop.shape
    for contour in raw_contours:
        # immediately disregard any features above/below size threshold
        if cv2.contourArea(contour) > 1000:   
            continue    # too big likely means clothes/hair
        elif cv2.contourArea(contour) < 15:
            continue    # too small likely means random noise
            
        # get contour info
        cx, cy, cw, ch = cv2.boundingRect(contour)

        # filter out top of image to reduce noise from hair/ hats
        if cy < (0.25 * h):
            continue

        # if filters are passed, add to valid contour list
        contours.append(contour)

    return contours


def cluster_contours(contours, distance_threshold=25):
    clusters = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centre = (x + w//2, y + h//2)
        added = False

        for cluster in clusters:
            cx, cy = cluster["centre"]

            # euclidean distance
            dist = ((centre[0]-cx)**2 + (centre[1]-cy)**2) ** 0.5

            if dist < distance_threshold:
                cluster["contours"].append(contour)

                # recompute centre
                all_centres = []
                for c in cluster["contours"]:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    all_centres.append((bx+bw//2, by+bh//2))

                avg_x = sum(p[0] for p in all_centres)//len(all_centres)
                avg_y = sum(p[1] for p in all_centres)//len(all_centres)

                cluster["centre"] = (avg_x, avg_y)

                added = True
                break

        if not added:
            clusters.append({
                "centre": centre, 
                "contours": [contour]
                })

    return clusters


def symmetry_bonus(a, b):
    ax, ay, aw, ah = get_cluster_bounds(a["cluster"])
    bx, by, bw, bh = get_cluster_bounds(b["cluster"])

    a_cx, a_cy = ax + aw/2, ay + ah/2
    b_cx, b_cy = bx + bw/2, by + bh/2

    # check for similar y level (should be fairly level)
    dy = abs(a_cy - b_cy)
    # check for horizontal seperation (should not overlap)
    dx = abs(a_cx - b_cx)

    # check for size difference (should be roughly the same size)
    size_diff = abs(aw - bw) + abs(ah - bh)

    return -((0.5 * dy) + (0.2 * size_diff)) + (0.1 * dx)

def score_clusters(clusters, image_shape):
    h, w = image_shape
    scores = []

    for cluster in clusters:
        x, y, cw, ch = get_cluster_bounds(cluster)

        cx = x + cw / 2
        cy = y + ch / 2

        # normalise values (0-1)
        norm_y = cy / h
        norm_x = cx / w
        aspect = cw / (ch + 1e-6)
        # area = cw * ch

        ##--------------------eye score--------------------##
        eye_score = 0                           # score higher if:
        
        eye_score += (1 - norm_y) * 3.0         #  - higher in frame
        eye_score += (1 - abs(norm_x - 0.5))    #  - closer to centre
        eye_score += min(aspect, 3.0)           #  - horizontally shaped
        ##-------------------------------------------------##

        ##-------------------mouth score-------------------##
        mouth_score = 0                         # score higher if:

        mouth_score += norm_y * 2.0             #  - lower in frame
        mouth_score += (1 - abs(norm_x - 0.5))  #  - closer to centre
        mouth_score += min(aspect, 4.0)         #  - horizontally shaped
        ##-------------------------------------------------##

        scores.append({
            "cluster": cluster,
            "eye_score": eye_score,
            "mouth_score": mouth_score
        })

    # check for symmetry between top eye candidates and adjust eye_score 
    # NOTE: use scores pointer, not a .copy() - otherwise scores won't update
    top_eye_candidates = sorted(scores, key=lambda c: c["eye_score"], reverse=True)[:5]
    if len(top_eye_candidates) >= 2:
        for i in range(len(top_eye_candidates)):
            for j in range(i+1, len(top_eye_candidates)):
                bonus = symmetry_bonus(top_eye_candidates[i], top_eye_candidates[j])

                top_eye_candidates[i]["eye_score"] += bonus
                top_eye_candidates[j]["eye_score"] += bonus

    return scores




def detect_landmarks(face_crop):

    # ensure uint8 type for cv2 functions
    face_crop = (face_crop).astype('uint8')

    # find contours
    raw_contours = find_contours(face_crop)

    # clean contours
    contours = filter_contours(face_crop, raw_contours)

    # cluster contours
    clusters = cluster_contours(contours)

    # score clusters
    scores = score_clusters(clusters, face_crop.shape)

    # select best candidates based on scores:
    
    # NOTE: currently, an error can occur if scores, or best mouth/eyes are empty
    if scores:
        best_mouth = max(scores, key=lambda c: c["mouth_score"])
        best_eyes = sorted(scores, key=lambda c: c["eye_score"], reverse=True)[:2]
    



    ##*********************************************************##
    #Debugging:
    # create debugging output
    output = cv2.cvtColor(face_crop.copy(), cv2.COLOR_GRAY2BGR)

    # draw all contours to debug version (in green)
    cv2.drawContours(output, contours, -1, (0,255,0), 2)

    # draw best eyes (in blue)
    if best_eyes is not None:
        for cluster in best_eyes:
            x,y,w,h = get_cluster_bounds(cluster["cluster"])
            cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)

    # draw best mouth (in red)
    if best_mouth is not None:
        x,y,w,h = get_cluster_bounds(best_mouth["cluster"])
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,0,255),2)

    #cv2.imwrite("data/debugging_images/test_landmarked.jpg", output)
    ##*********************************************************##

    # NOTE: In a later stage, a return value will be added to provide feedback to
    #       previous stages - if anything needs adjusting in next frame (mouth clipped
    #       out of bottom of frame - looser crop etc), this will allow for that to 
    #       happen for a more accurate end result.

    # return contours and debugging image
    return output

    

    

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()
    
    # img_raw = cv2.imread("data/debugging_images/test_normalized.jpg")

    # # apply some processing
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # landmarks = detect_landmarks(img)

    