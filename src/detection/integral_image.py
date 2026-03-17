import numpy as np

def compute_integral_image(image):
    """
    Computes the integral image of a given input.

    Parameters:
        image(np.array) : Image to be integrated

    Returns:
        (np.array) : Integral image
    """

    h, w = image.shape
    integral = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range (w):
            top = integral[y-1, x] if y> 0 else 0
            left = integral[y, x-1] if x > 0 else 0
            top_left = integral[y-1, x-1] if (x > 0 and y > 0) else 0

            integral[y, x] = image[y, x] + top + left - top_left
        
    return integral

if __name__ == "__main__":
    img = np.array(
    [
    [1, 2],
    [3, 4]
    ])

    print(compute_integral_image(img))
    #output should be - [1, 3,
    #                   4, 10]
