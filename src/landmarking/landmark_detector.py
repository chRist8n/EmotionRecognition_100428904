import cv2

def detect_landmarks(face_crop):
    _, thresh = cv2.threshold(face_crop, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = face_crop.copy()
    cv2.drawContours(debug, contours, -1, (0,255,0), 2)
    #cv2.imwrite("data/debugging_images/test_landmarked.jpg", debug)
    return debug

    

if __name__ == "__main__":
    img_raw = cv2.imread("data/debugging_images/test_normalized.jpg")

    # apply some processing
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)

    landmarks = detect_landmarks(img)

    