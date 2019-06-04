import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def find_peak(map, x, y):
    dx = np.array([-1,  0,  1, 0, -1,  1, -1, 0, 1])
    dy = np.array([-1, -1, -1, 0,  0,  0,  1, 1, 1])
    while True:
        nx = np.minimum(np.maximum(x + dx, 0), map.shape[1] - 1)
        ny = np.minimum(np.maximum(y + dy, 0), map.shape[0] - 1)

        max_idx = np.argmax(map[ny, nx])

        if x == nx[max_idx] and y == ny[max_idx]:
            break
        x = nx[max_idx]
        y = ny[max_idx]
    return x, y


if __name__ == '__main__':
    assert find_peak(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 0, 0) == (1, 1)
    assert find_peak(np.array([[0, 2, 0], [0, 1, 0], [0, 0, 0]]), 1, 1) == (1, 0)
