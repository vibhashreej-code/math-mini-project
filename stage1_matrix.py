"""Stage 1: Matrix representation (owner: A, pipeline lead).

Concept : an image is a grid of pixels, each pixel is an RGB vector.
Purpose : flatten the photo into an N x 3 matrix X so linear algebra applies.
Outcome : each row of X is one pixel (R, G, B), values in [0, 1].
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SIZE = (100, 100)          # (width, height) -> N = 10,000 pixels
PHOTO_PATH = "photo.jpg"   # put the team's fixed photo here


def make_demo_photo(size=SIZE, seed=42):
    """Fixed synthetic photo, used if photo.jpg is missing (same every run)."""
    w, h = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    r = x / (w - 1)
    g = y / (h - 1)
    b = 0.6 * (1 - x / (w - 1)) + 0.1 * rng.random((h, w))
    img = np.stack([r, g, b], axis=2)
    # a bright "sun" so the photo has an obvious feature
    sun = (x - 70) ** 2 + (y - 30) ** 2 < 15 ** 2
    img[sun] = [1.0, 0.9, 0.3]
    return np.clip(img, 0, 1)


def load_photo(path=PHOTO_PATH, size=SIZE):
    """Return an (H, W, 3) float array in [0, 1]."""
    if os.path.exists(path):
        img = Image.open(path).convert("RGB").resize(size)
        return np.asarray(img, dtype=float) / 255.0
    print(f"  (no {path} found, using the built-in demo photo)")
    return make_demo_photo(size)


def image_to_matrix(img):
    """(H, W, 3) image -> (N, 3) matrix, N = H * W."""
    return img.reshape(-1, 3)


def matrix_to_image(X, shape):
    """(N, 3) matrix -> (H, W, 3) image, clipped to the valid [0, 1] range."""
    return np.clip(X, 0, 1).reshape(shape)


def run(path=PHOTO_PATH):
    """Stage 1. Returns X (N x 3) and the image shape (H, W, 3)."""
    img = load_photo(path)
    shape = img.shape
    X = image_to_matrix(img)

    print(f"Photo size      : {shape[1]} x {shape[0]} pixels")
    print(f"Matrix X shape  : {X.shape}  (N pixels x 3 colors)")
    print("First 5 rows of X (R, G, B):")
    print(np.round(X[:5], 3))

    ok = np.allclose(matrix_to_image(X, shape), img)
    print(f"SANITY: reshaping X back gives the original photo -> {'OK' if ok else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.set_title("Stage 1: original photo")
    ax.axis("off")
    return X, shape


if __name__ == "__main__":
    run()
    plt.show()
