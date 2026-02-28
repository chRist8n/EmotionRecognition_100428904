from PIL import Image
import os
import numpy as np

def normalize_image(input_path, output_path, size=(256, 256)):
    """
    Recieves an unprocessed image and applies the following:
    - greyscale
    - resize to 256x256
    - pixel scaling
    
    Parameters:
        input_path (str) : Filepath for input (MAYBE CHANGE TO INPUT IMAGE DIRECTLY)
        output_path (str) : Filepath to save the result to
        size (tuple:int) : Size to resize image to

    Returns:
        (str) : Filepath for output
    """

    # load image
    img = Image.open(input_path)

    # convert to greyscale
    img = img.convert("L")

    # resize
    img = img.resize(size)

    # scale pixels
    arr = np.array(img).astype("float32")
    arr /= 255.0        # scale from 0-255 to 0-1

    # output

        # save as array by default
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path.replace(".jpg", ".npy"), arr)

        #save as jpg for debugging
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

    return output_path

if __name__ == "__main__":
    test_input = "data/raw/RAF_DB/DATASET/train/1/train_00012_aligned.jpg" 
    test_output = "data/debugging_images/test_normalized.jpg"

    result = normalize_image(test_input, test_output)
    print("Saved to: ", result)