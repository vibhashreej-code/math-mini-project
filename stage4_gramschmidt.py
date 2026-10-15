"""Stage 4: Orthogonal basis (Gram-Schmidt) on the RGB color axes.

Concept : Build an orthonormal basis for RGB space from 3 starting vectors.
Purpose : Gives a brightness axis plus two color-difference axes.
Outcome : Q^T Q = I, so every dot product between different axes is ~0.
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_style import COLORS, apply_style


def gram_schmidt(V):
    """Columns of V -> orthonormal columns Q (classical Gram-Schmidt)."""
    V = np.asarray(V, dtype=float)
    Q = np.zeros_like(V)
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for i in range(j):
            v -= (Q[:, i] @ V[:, j]) * Q[:, i]   # subtract component along earlier axes
        Q[:, j] = v / np.linalg.norm(v)
    return Q


def stage4_orthogonal_basis(X=None, S=None, show=True, plot=True):
    """X, S accepted for the shared interface; the basis itself is fixed.
    Returns Q (3x3): column 0 = brightness, columns 1-2 = color differences."""
    # Start vectors: white (brightness), red, green
    V = np.array([[1, 1, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=float).T
    Q = gram_schmidt(V)

    print("=== STAGE 4: Orthogonal basis (Gram-Schmidt) ===")
    names = ["brightness", "color axis 1", "color axis 2"]
    for n, col in zip(names, Q.T):
        print(f"  {n:13s}: {np.round(col, 4)}")
    print("  Dot products (should be ~0):")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        print(f"    q{i+1}.q{j+1} = {Q[:, i] @ Q[:, j]:.2e}")
    ok = np.allclose(Q.T @ Q, np.eye(3))
    print(f"  Sanity check Q^T Q == I: {ok}")

    if plot:
        apply_style()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(COLORS["bg"])
        cols = [COLORS["brightness"], COLORS["color1"], COLORS["color2"]]
        for n, v, c in zip(names, Q.T, cols):
            ax.quiver(0, 0, 0, *v, color=c, linewidth=3, arrow_length_ratio=0.12)
            ax.text(*(v * 1.12), n, color=c)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
        ax.set_title("Stage 4: Orthonormal basis for RGB space")
        plt.tight_layout()
        if show:
            plt.show()
    return Q


def run(X=None, S=None, shape=None):
    """Pipeline hook: builds the figure, pipeline.py shows it."""
    stage4_orthogonal_basis(X, S, show=False)


if __name__ == "__main__":
    stage4_orthogonal_basis()
