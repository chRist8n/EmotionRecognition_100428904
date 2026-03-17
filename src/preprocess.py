from PIL import Image
import os
import numpy as np

def preprocess_image(input, output_path="", size=(256, 256)):
    """
    Recieves an unprocessed image and applies the following:
    - greyscale
    - resize to 256x256
    - pixel scaling
    
    Parameters:
        input (str)/(array) : Filepath for input (MAYBE CHANGE TO INPUT IMAGE DIRECTLY)
        output_path (str)/(array) : Filepath to save the result to
        size (tuple:int) : Size to resize image to
    ((Note that the input can either be a filepath or a numpy 
    array but the output will be of the same type as the input))

    Returns:
        (str) : Filepath for output
                    or
        (array) : Image as numpy array
    """

    # load image
    if isinstance(input, str):
        # str means filepath
        filetype = "str"
        img = Image.open(input)
    else:
        # image parsed directly
        filetype = "arr"
        img = Image.fromarray(input)

    # convert to greyscale
    img = img.convert("L")

    # resize
    img = img.resize(size)

    # scale down pixel values from 0-255 to 0-1
    arr = np.array(img).astype(np.float32) / 255.0

    # output
    if filetype == "str":
        # save array to output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path.replace(".jpg", ".npy"), arr)
    else:
        #(save as jpg for debugging)
        #{
        os.makedirs(os.path.dirname("data/debugging_images/test_normalized.jpg"), exist_ok=True)
        img.save("data/debugging_images/test_normalized.jpg")
        #}
        # return array
        return arr


if __name__ == "__main__":
    test_input = "data/raw/RAF_DB/DATASET/train/1/train_00012_aligned.jpg" 
    test_output = "data/debugging_images/test_normalized.jpg"

    result = preprocess_image(test_input, test_output)
    print("Saved to: ", result)