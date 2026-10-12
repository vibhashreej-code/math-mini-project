import numpy as np
import matplotlib.pyplot as plt
import sys

# --------------------------------------------------
# STAGE 3: RREF, RANK AND INVERSE OF FILTER MATRICES
# --------------------------------------------------

def rref(matrix):
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        best = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))

        if abs(A[best, col]) < 1e-10:
            continue

        A[[pivot_row, best]] = A[[best, pivot_row]]
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]

        for row in range(rows):
            if row != pivot_row:
                A[row] -= A[row, col] * A[pivot_row]

        pivot_row += 1

    A[np.abs(A) < 1e-10] = 0
    return A


def matrix_rank(matrix):
    reduced = rref(matrix)
    return int(np.sum(np.any(np.abs(reduced) > 1e-10, axis=1)))


def analyze_filter(name, matrix):
    print(f"\n--- {name} ---")
    print("Filter matrix:")
    print(np.round(matrix, 3))

    reduced = rref(matrix)
    rank = matrix_rank(matrix)

    print("\nRREF:")
    print(np.round(reduced, 3))
    print("\nRank:", rank)

    if matrix.shape[0] == matrix.shape[1] and rank == matrix.shape[0]:
        inverse = np.linalg.inv(matrix)
        print("Inverse exists.")
        print("Inverse matrix:")
        print(np.round(inverse, 3))

        check = matrix @ inverse
        print("\nCheck: Filter × Inverse:")
        print(np.round(check, 3))
    else:
        print("Inverse does not exist: information has been lost.")

    return rank


def stage3():
    sepia = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    grayscale = np.array([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114]
    ])

    print("=" * 50)
    print("STAGE 3: RREF, RANK AND FILTER INVERSE")
    print("=" * 50)

    sepia_rank = analyze_filter("SEPIA FILTER", sepia)
    gray_rank = analyze_filter("GRAYSCALE FILTER", grayscale)

    print("\n--- FINAL COMPARISON ---")
    print("Sepia rank:", sepia_rank)
    print("Grayscale rank:", gray_rank)

    if sepia_rank == 3:
        print("Sepia preserves all three color dimensions.")
        print("Its transformation is invertible.")
        print("However, clipping pixel values can still lose information.")

    if gray_rank == 1:
        print("Grayscale maps RGB colors onto one brightness direction.")
        print("Different original colors can produce the same gray value.")
        print("Therefore, the original colors cannot generally be recovered.")


def load_or_make_image():
    if len(sys.argv) > 1:
        from PIL import Image
        img = Image.open(sys.argv[1]).convert("RGB").resize((100, 100))
        return np.asarray(img, dtype=float) / 255.0

    rng = np.random.default_rng(0)
    h = w = 100
    yy, xx = np.mgrid[0:h, 0:w] / (h - 1)

    r = 0.2 + 0.7 * xx
    g = 0.1 + 0.5 * xx + 0.3 * yy
    b = 0.8 - 0.5 * yy

    img = np.stack([r, g, b], axis=2)
    img += rng.normal(0, 0.03, (h, w, 3))

    return np.clip(img, 0, 1)


def apply_filter(X, matrix):
    return X @ matrix.T


def show_comparison(img):
    X = img.reshape(-1, 3)

    sepia = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    grayscale = np.array([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114]
    ])

    sepia_img = np.clip(apply_filter(X, sepia), 0, 1).reshape(img.shape)
    gray_img = np.clip(apply_filter(X, grayscale), 0, 1).reshape(img.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original Photo")

    axes[1].imshow(sepia_img)
    axes[1].set_title("Sepia Filter")

    axes[2].imshow(gray_img)
    axes[2].set_title("Grayscale Filter")

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Stage 3: Comparing Photo Filters")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stage3()
    img = load_or_make_image()
    show_comparison(img)