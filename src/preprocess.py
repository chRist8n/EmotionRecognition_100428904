from PIL import Image
import os

def normalize_image(input_path, output_path, size=(256, 256)):
    """
    Loads image, converts to grayscale, resizes, saves result.
    """

    # load image
    img = Image.open(input_path)

    # convert to grayscale
    img = img.convert("L")

    # resize
    img = img.resize(size)

    # check output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save
    img.save(output_path)

    return output_path

if __name__ == "__main__":
    test_input = "data/raw/RAF_DB/DATASET/train/1/train_00010_aligned.jpg" 
    test_output = "data/processed/test_normalized.jpg"

    result = normalize_image(test_input, test_output)
    print("Saved to: ", result)