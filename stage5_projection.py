"""Stage 5: Projection onto the brightness direction (= grayscale).

Concept : Projection of each pixel vector onto a unit vector u: P = u u^T.
Purpose : Grayscale is exactly this projection, so it keeps only brightness.
Outcome : A rank-1 result; the leftover (color) part is orthogonal to u.
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_style import COLORS, apply_style


def stage5_projection(X, S=None, u=None, shape=None, show=True, plot=True):
    """X: N x 3 pixel matrix (values 0..1). u: unit brightness direction
    (pass Q[:, 0] from stage 4, or leave None for [1,1,1]/sqrt(3)).
    shape: (H, W) to reshape for display. Returns (X_proj, P)."""
    X = np.asarray(X, dtype=float)
    if u is None:
        u = np.ones(3)
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)

    P = np.outer(u, u)            # 3x3 projection matrix
    X_proj = X @ P                # P is symmetric, so X P^T = X P
    residual = X - X_proj

    print("=== STAGE 5: Projection (grayscale) ===")
    print("  Brightness direction u:", np.round(u, 4))
    print("  Projection matrix P = u u^T:\n", np.round(P, 4))
    print("  rank(P) =", np.linalg.matrix_rank(P))
    print("  Check P@P == P (idempotent):", np.allclose(P @ P, P))
    print("  Check residual . u == 0   :", np.allclose(residual @ u, 0))

    if plot:
        if shape is None:
            side = int(np.sqrt(len(X)))
            shape = (side, len(X) // side)
        before = np.clip(X, 0, 1).reshape(*shape, 3)
        after = np.clip(X_proj, 0, 1).reshape(*shape, 3)
        apply_style()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(before); axs[0].set_title("Original")
        axs[1].imshow(after); axs[1].set_title("Projected onto brightness")
        for a in axs:
            a.axis("off")
        fig.suptitle("Stage 5: Projection = grayscale")
        plt.tight_layout()
        if show:
            plt.show()
    return X_proj, P


def run(X, S=None, shape=None):
    """Pipeline hook. Uses u = [1,1,1]/sqrt(3), the same brightness axis that
    stage 4 builds first. shape from the pipeline is (H, W, 3); this stage wants (H, W)."""
    stage5_projection(X, S, shape=None if shape is None else shape[:2], show=False)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    stage5_projection(rng.random((100 * 100, 3)), shape=(100, 100))
