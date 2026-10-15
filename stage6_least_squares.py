import numpy as np
import matplotlib.pyplot as plt
import sys

# --------------------------------------------------
# STAGE 6: LEAST SQUARES
# Recover a filter matrix from before-and-after pixels
# --------------------------------------------------

def stage6(img=None, true_matrix=None, show=True):
    # The actual Sepia filter matrix (unless the pipeline passes the team matrix)
    if true_matrix is None:
        true_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])

    # Use the image from the pipeline if given, else load one if a path is provided
    if img is not None:
        pass
    elif len(sys.argv) > 1:
        from PIL import Image

        img = Image.open(sys.argv[1]).convert("RGB")
        img = img.resize((100, 100))

        img = np.asarray(img, dtype=float) / 255.0
    else:
        # Create a reproducible sample image
        rng = np.random.default_rng(0)
        img = rng.uniform(0.1, 0.7, (100, 100, 3))

    # Convert the image into a matrix of RGB pixels
    X = img.reshape(-1, 3)

    # Apply the true filter without clipping
    # so the original matrix can be recovered accurately
    Y = X @ true_matrix.T

    # Estimate the filter matrix using least squares
    estimated_transpose, residuals, rank, singular_values = (
        np.linalg.lstsq(X, Y, rcond=None)
    )

    estimated_matrix = estimated_transpose.T

    # Calculate the output predicted by the estimated matrix
    Y_estimated = X @ estimated_matrix.T

    # Calculate the total squared error
    error = np.sum((Y - Y_estimated) ** 2)

    # Display the matrices and results
    print("=" * 50)
    print("STAGE 6: LEAST SQUARES")
    print("=" * 50)

    print("\nTrue Sepia Matrix:")
    print(true_matrix)

    print("\nEstimated Filter Matrix:")
    print(np.round(estimated_matrix, 6))

    print("\nTotal Squared Error:")
    print(error)

    print("\nRank of Original Pixel Matrix:")
    print(rank)

    print("\nSanity Check:")
    print("Estimated matrix is close to the true matrix:",
          np.allclose(true_matrix, estimated_matrix, atol=1e-8))

    # Convert filtered pixels back into images
    filtered_img = np.clip(Y, 0, 1).reshape(img.shape)
    estimated_img = np.clip(Y_estimated, 0, 1).reshape(img.shape)

    # Display the original and filtered images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Stage 6: Original Photo")

    axes[1].imshow(filtered_img)
    axes[1].set_title("Stage 6: Sepia Photo")

    axes[2].imshow(estimated_img)
    axes[2].set_title("Stage 6: Estimated Filter")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if show:
        plt.show()


def run(X, S, shape):
    """Pipeline hook: recover the team sepia matrix S from the before/after pixels."""
    stage6(img=X.reshape(shape), true_matrix=S, show=False)


if __name__ == "__main__":
    stage6()
