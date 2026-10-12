# Linear Algebra of a Photo Filter

A math mini-project that shows how a photo filter works using linear algebra. A photo is turned into a matrix, and each stage of the program applies one linear algebra idea to it: matrix multiplication, rank and inverse, orthogonal bases, projection, least squares, eigenvectors and PCA.

The whole project runs as one Python script on a small photo (about 100 x 100 pixels). It pauses between stages so each result can be explained during the demo.

## The idea

Every pixel is an RGB vector. Stack all the pixels of a photo and you get an N x 3 matrix `X`. A color filter (sepia, grayscale, tint) is a 3 x 3 matrix, and applying it to the whole photo is one matrix product. Everything else in the project builds on that.

## Stages

| Stage | Concept | What the screen shows | Outcome |
|-------|---------|-----------------------|---------|
| 1 | Matrix representation | Photo, N x 3 shape, first rows | Each pixel is an RGB vector |
| 2 | Matrix multiplication | Sepia 3 x 3 matrix, before/after photo | A filter is one matrix times every pixel |
| 3 | RREF / rank / inverse | Rank of sepia vs grayscale, inverse | Sepia has rank 3 (can be undone); grayscale has rank 1 (information is lost) |
| 4 | Orthogonal basis (Gram-Schmidt) | Three vectors, dot products near 0 | A brightness axis plus two color axes |
| 5 | Projection | Photo projected onto the brightness axis | Grayscale result |
| 6 | Least squares | True vs estimated matrix, error | Filter recovered from a before/after pair |
| 7 | Eigenvalues / eigenvectors | Eigen-decomposition of the color covariance | The dominant color directions |
| 8 | Reduced model (PCA) | Photo rebuilt from 2 components, variance kept | 3 colors reduced to 2 with small error |

## How to run

1. Install the requirements:

   ```
   pip install numpy pillow matplotlib
   ```

2. (Optional) Put the photo you want to use in the project folder as `photo.jpg`. If there is no `photo.jpg`, the program uses a built-in demo photo.

3. Run the pipeline:

   ```
   python pipeline.py
   ```

   Press Enter to move to the next stage. To run all stages without pausing:

   ```
   python pipeline.py --no-pause
   ```

Stages whose files are not in the folder yet are skipped, so the program always runs.

## Files

| File | What it does |
|------|--------------|
| `pipeline.py` | Main script: runs the stages in order, pausing between them |
| `stage1_matrix.py` | Stage 1: loads the photo and turns it into the N x 3 matrix |
| `stage2_filter.py` | Stage 2: sepia and grayscale matrices, applying and chaining filters |
| `stage3_rref_rank_inverse.py` | Stage 3: RREF, rank and inverse |
| `stage7_8_eigen_pca.py` | Stages 7 and 8: eigen-analysis and the PCA reduced model |

Stages 4, 5 and 6 will be added as their own files.

## Adding a stage

Each stage file has a function `run(X, S, shape)`:

- `X`: the N x 3 pixel matrix from stage 1
- `S`: the 3 x 3 sepia matrix from stage 2
- `shape`: the photo shape `(height, width, 3)`, for turning `X` back into an image

`run` should print its numbers, finish with a one-line sanity check, and create matplotlib figures. Do not call `plt.show()` inside `run`, because `pipeline.py` shows the figures. Put any test code under `if __name__ == "__main__":` so that importing the file does not run it.

## Notes

- Pixels are stored as rows of `X`, so a filter `M` is applied as `X @ M.T`.
- The sepia matrix is a blend of the identity and the classic sepia matrix. The classic one is almost singular, which would make the stage 3 inverse unreliable.

## Team

Fill in your names and the stages each person owns.

- Pipeline lead (stages 1-2 and `pipeline.py`): 
- Stage 3: 
- Stages 4-6: 
- Stages 7-8:
