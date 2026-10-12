"""Stage 2: Matrix multiplication (owner: A, pipeline lead).

Concept : a color filter is a 3 x 3 matrix; applying it multiplies every pixel.
Purpose : show that a whole-photo filter is ONE matrix product, and that
          chaining filters is just multiplying the filter matrices.
Outcome : sepia photo from  Y = X @ S.T

Convention: pixels are ROW vectors in X (N x 3), so the filter is applied as
X @ M.T. For a single pixel column vector p this is the usual  M @ p.
"""
import numpy as np
import matplotlib.pyplot as plt

from stage1_matrix import matrix_to_image

# Classic sepia matrix, kept for reference. It is almost singular
# (det ~ 1e-7), so it is NOT used directly: it would make "sepia is
# invertible" (stage 3) fail for numerical reasons.
SEPIA_CLASSIC = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131],
])

# Team sepia matrix: a 60/40 blend of the identity and classic sepia.
# Looks sepia, is clearly invertible (rank 3), and is well conditioned.
SEPIA = 0.4 * np.eye(3) + 0.6 * SEPIA_CLASSIC

# Grayscale: every output channel is the same brightness mix (rank 1).
GRAY = np.tile([0.299, 0.587, 0.114], (3, 1))


def apply_filter(X, M):
    """Apply 3x3 filter M to every pixel of X (N x 3). Not clipped."""
    return X @ M.T


def run(X, shape):
    """Stage 2. Returns the sepia-filtered matrix Y (N x 3, unclipped)."""
    print("Sepia matrix S:")
    print(np.round(SEPIA, 3))

    Y = apply_filter(X, SEPIA)
    print(f"Y = X @ S.T has shape {Y.shape}  (one matrix product for all pixels)")

    # Composition check: sepia then grayscale == the single matrix GRAY @ SEPIA
    two_steps = apply_filter(apply_filter(X, SEPIA), GRAY)
    one_step = apply_filter(X, GRAY @ SEPIA)
    ok = np.allclose(two_steps, one_step)
    print("Chaining filters = multiplying matrices (G @ S):")
    print(np.round(GRAY @ SEPIA, 3))
    print(f"SANITY: sepia-then-gray equals one (G @ S) filter -> {'OK' if ok else 'FAIL'}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(matrix_to_image(X, shape))
    axes[0].set_title("Stage 2: before")
    axes[1].imshow(matrix_to_image(Y, shape))
    axes[1].set_title("Stage 2: after sepia")
    for ax in axes:
        ax.axis("off")
    return Y


if __name__ == "__main__":
    from stage1_matrix import run as stage1
    X, shape = stage1()
    run(X, shape)
    plt.show()
