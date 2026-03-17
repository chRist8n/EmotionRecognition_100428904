def sliding_window(image, window_size, step):
    h, w = image.shape

    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            yield (x, y, window_size, window_size)