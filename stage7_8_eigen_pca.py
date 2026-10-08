"""
Stages 7 and 8: Eigen-analysis of pixel colors (PCA) and the reduced model.

Interface (agreed with the team):
    X         : (N, 3) float array, each row is one pixel [R, G, B] scaled 0..1
    img_shape : (H, W, 3) shape of the original image, needed to rebuild it

Run this file directly to test it on a dummy image, or on a real photo:
    python stage7_8_eigen_pca.py
    python stage7_8_eigen_pca.py my_photo.jpg
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# STAGE 7: Eigenvalues / eigenvectors of the color covariance matrix
# ---------------------------------------------------------------------------
def stage7_eigen(X, show=True):
    """
    Concept : eigenvectors of the 3x3 color covariance matrix.
    Purpose : find the directions in RGB space along which the image's
              colors vary the most (the "dominant color directions").
    Outcome : 3 eigenvalues (how much variation along each direction)
              and 3 eigenvectors (the directions themselves).

    Returns: mean (3,), eigvals (3,) sorted largest first,
             eigvecs (3,3) with eigenvectors as COLUMNS, same order.
    """
    # Step 1: center the data. Subtract the average color from every pixel,
    # so the cloud of pixels is centered at the origin.
    mean = X.mean(axis=0)
    Xc = X - mean

    # Step 2: covariance matrix (3x3). Entry (i, j) says how much
    # color channel i and channel j vary together.
    N = X.shape[0]
    C = (Xc.T @ Xc) / (N - 1)

    # Step 3: eigen-decomposition. C is symmetric, so use eigh, which
    # guarantees real eigenvalues and perpendicular eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(C)

    # eigh returns smallest first, so flip to largest first.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Sign convention: make the largest component of each eigenvector
    # positive, so results are the same every run.
    for i in range(3):
        if eigvecs[np.argmax(np.abs(eigvecs[:, i])), i] < 0:
            eigvecs[:, i] *= -1

    # ---- printed output ----
    print("\n=== STAGE 7: Eigen-analysis of pixel colors ===")
    print("Mean color (R, G, B):", np.round(mean, 3))
    print("\nCovariance matrix C (3x3):\n", np.round(C, 4))
    print("\nEigenvalues (largest first):", np.round(eigvals, 5))
    share = eigvals / eigvals.sum()
    for i in range(3):
        print(f"  Direction {i+1}: {share[i]*100:5.1f}% of color variation, "
              f"vector = {np.round(eigvecs[:, i], 3)}")

    # ---- sanity checks ----
    # 1. Definition of an eigenvector: C v = lambda v
    err = np.max(np.abs(C @ eigvecs - eigvecs * eigvals))
    # 2. Eigenvectors of a symmetric matrix are perpendicular: V^T V = I
    orth = np.max(np.abs(eigvecs.T @ eigvecs - np.eye(3)))
    print(f"\nCheck: max |C v - lambda v| = {err:.2e}  (should be ~0)")
    print(f"Check: max |V^T V - I|      = {orth:.2e}  (should be ~0)")

    if show:
        _plot_eigen(X, mean, eigvals, eigvecs)

    return mean, eigvals, eigvecs


def _plot_eigen(X, mean, eigvals, eigvecs, max_points=1500):
    """Scatter of pixel colors in the R-G plane, with the principal arrows."""
    rng = np.random.default_rng(0)  # fixed seed so the plot is repeatable
    idx = rng.choice(len(X), size=min(max_points, len(X)), replace=False)
    pts = X[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.35, color="tab:blue",
               label="pixels")
    ax.scatter(*mean[:2], color="black", s=60, zorder=5, label="mean color")

    colors = ["tab:orange", "tab:green", "tab:red"]
    for i in range(3):
        # Arrow length = 2 standard deviations along that direction.
        v = eigvecs[:, i] * 2 * np.sqrt(eigvals[i])
        ax.annotate("", xy=mean[:2] + v[:2], xytext=mean[:2],
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=3))
        ax.plot([], [], color=colors[i], lw=3,
                label=f"direction {i+1} ({eigvals[i]/eigvals.sum()*100:.0f}%)")

    ax.set_xlabel("Red", fontsize=14)
    ax.set_ylabel("Green", fontsize=14)
    ax.set_title("Stage 7: Dominant color directions (R-G view of RGB space)",
                 fontsize=15)
    ax.legend(fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# STAGE 8: Final reduced model (compress 3 colors down to k)
# ---------------------------------------------------------------------------
def stage8_reduce(X, img_shape, mean, eigvals, eigvecs, k=2, show=True):
    """
    Concept : projection onto the top-k eigenvectors (PCA compression).
    Purpose : describe each pixel with k numbers instead of 3, keeping
              the directions that carry the most color information.
    Outcome : a rebuilt photo from k components + how much was kept/lost.

    Returns: X_rebuilt (N,3), variance_kept (fraction), rmse (float).
    """
    # Keep only the first k eigenvectors: a 3 x k matrix W.
    W = eigvecs[:, :k]

    # Compress: each pixel (3 numbers) becomes k numbers.
    # Z = (X - mean) W   ->  shape (N, k)
    Z = (X - mean) @ W

    # Rebuild: go back to 3 numbers. X_rebuilt = Z W^T + mean
    X_rebuilt = Z @ W.T + mean
    X_rebuilt = np.clip(X_rebuilt, 0, 1)  # keep valid pixel values

    variance_kept = eigvals[:k].sum() / eigvals.sum()
    rmse = np.sqrt(np.mean((X - X_rebuilt) ** 2))

    # ---- printed output ----
    print(f"\n=== STAGE 8: Reduced model ({k} of 3 components) ===")
    print(f"Each pixel: 3 numbers -> {k} numbers  (matrix shape {X.shape} -> {Z.shape})")
    print(f"Variance kept : {variance_kept*100:.2f}%")
    print(f"Variance lost : {(1-variance_kept)*100:.2f}%")
    print(f"Rebuild error (RMSE, scale 0-1): {rmse:.4f}")

    # ---- sanity check ----
    # With k = 3 nothing is thrown away, so the rebuild must be perfect.
    Z3 = (X - mean) @ eigvecs
    X3 = Z3 @ eigvecs.T + mean
    print(f"Check: rebuild with all 3 components, max error = "
          f"{np.max(np.abs(X - X3)):.2e}  (should be ~0)")

    if show:
        _plot_reduced(X, X_rebuilt, img_shape, k, variance_kept)

    return X_rebuilt, variance_kept, rmse


def _plot_reduced(X, X_rebuilt, img_shape, k, variance_kept):
    """Original vs rebuilt photo, side by side."""
    original = X.reshape(img_shape)
    rebuilt = X_rebuilt.reshape(img_shape)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original (3 numbers per pixel)", fontsize=14)
    axes[1].imshow(rebuilt)
    axes[1].set_title(f"Rebuilt from {k} components\n"
                      f"({variance_kept*100:.1f}% of variation kept)", fontsize=14)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Stage 8: Reduced model", fontsize=16)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Standalone test (not used when A's main script imports these functions)
# ---------------------------------------------------------------------------
def _load_or_make_image():
    """Load a photo if a path is given, otherwise make a dummy colorful one."""
    if len(sys.argv) > 1:
        from PIL import Image
        img = Image.open(sys.argv[1]).convert("RGB").resize((100, 100))
        return np.asarray(img, dtype=float) / 255.0

    rng = np.random.default_rng(0)  # fixed seed
    h = w = 100
    yy, xx = np.mgrid[0:h, 0:w] / (h - 1)
    r = 0.2 + 0.7 * xx
    g = 0.1 + 0.5 * xx + 0.3 * yy
    b = 0.8 - 0.5 * yy
    img = np.stack([r, g, b], axis=2) + rng.normal(0, 0.03, (h, w, 3))
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    img = _load_or_make_image()
    X = img.reshape(-1, 3)  # image -> N x 3 matrix (this is Stage 1's job)

    mean, eigvals, eigvecs = stage7_eigen(X)
    stage8_reduce(X, img.shape, mean, eigvals, eigvecs, k=2)
