def smooth_landmarks(prev, current, alpha=0.7):
    if prev is None:
        return current

    return [
        (
            alpha * px + (1 - alpha) * cx,
            alpha * py + (1 - alpha) * cy
        )
        for (px, py), (cx, cy) in zip(prev, current)
    ]