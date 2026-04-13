import sys
import os
import cv2




def find_contours(input):
    # set threshold for contour detection
    _, thresh = cv2.threshold(input, 65, 255, cv2.THRESH_BINARY_INV)

    # detect contours from thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def filter_contours(input, raw_contours):
    # filter out noise
    contours = []
    w, h = input.shape
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


def detect_landmarks(input_raw):

    # parse to uint8 for cv2 functions
    if input_raw.dtype != 'uint8':
        input_raw = (input_raw).astype('uint8')


    # apply Gaussian blur
    input = cv2.GaussianBlur(input_raw, (3,3), 0)

    # find contours
    raw_contours = find_contours(input)

    # clean contours
    contours = filter_contours(input, raw_contours)



    ##*********************************************************##
    #Debugging:
    # create debugging output
    output = cv2.cvtColor(input.copy(), cv2.COLOR_GRAY2BGR)
    # draw contours to debug version
    cv2.drawContours(output, contours, -1, (0,255,0), 2)
    #cv2.imwrite("data/debugging_images/test_landmarked.jpg", output)
    ##*********************************************************##

    # NOTE: In a later stage, a return value will be added to provide feedback to
    #       previous stages - if anything needs adjusting in next frame (mouth clipped
    #       out of bottom of frame - looser crop etc), this will allow for that to 
    #       happen for a more accurate end result.

    # return contours and debugging image
    return contours, output

    

    

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()
    
    # img_raw = cv2.imread("data/debugging_images/test_normalized.jpg")

    # # apply some processing
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # landmarks = detect_landmarks(img)

    