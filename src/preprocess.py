import sys
import os
import cv2

def preprocess_image(input, size=(256, 256)):
    """
    Receives an image and applies the following:
    - greyscale
    - resize to 256x256
    - apply CLAHE
    - apply Gaussian blur
    
    Parameters:
        input (str)/(array) : Filepath for input (str) / image (array)
        size (tuple:int) : Size to resize image to

    Note: The input can either be a filepath or a numpy 
    array but the output will be returned as an image (array)

    Returns:
        (array) : Image as numpy array
    """

    # load image in greyscale
    if isinstance(input, str):
        # str means filepath
        img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    else:
        # image parsed directly
        img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    # resize
    img = cv2.resize(img, size)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Add Gaussian blur to soften noise
    img = cv2.GaussianBlur(img, (3,3), 0)


    # # save for debugging
    # cv2.imwrite("data/debugging_images/preprocessed.jpg", img)

    # output
    return img


if __name__ == "__main__":
    # test_input = "data/raw/RAF_DB/DATASET/train/1/train_00012_aligned.jpg" 
    # test_output = "data/debugging_images/test_normalized.jpg"

    # result = preprocess_image(test_input, test_output)
    # print("Saved to: ", result)

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    webcam_debug
    sys.exit()