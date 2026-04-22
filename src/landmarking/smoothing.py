def smooth_landmarks(prev, current, alpha=0.25):
    if prev is None:
        return current

    return [
        (
            alpha * px + (1 - alpha) * cx,
            alpha * py + (1 - alpha) * cy
        )
        for (px, py), (cx, cy) in zip(prev, current)
    ]

def smooth_box(prev, current, pos_alpha=0.3, size_alpha=0.5):
    if prev is None:
        return current
    
    px, py, pw, ph = prev
    cx, cy, cw, ch = current

    return (
        pos_alpha * px + (1 - pos_alpha) * cx,      #x
        pos_alpha * py + (1 - pos_alpha) * cy,      #y
        size_alpha * pw + (1 - size_alpha) * cw,    #width
        size_alpha * ph + (1 - size_alpha) * ch     #height
    )