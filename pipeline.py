"""Main staged script (owner: A, pipeline lead).

Runs the 8 stages in order and waits for a keypress between them.
Stages 1-2 are A's. Stages 3-8 live in teammates' files and are imported
if they exist; a missing file is skipped so the pipeline always runs.

Teammate interface (agreed with the team):
    each stage file has   run(X, S, shape)
        X     : N x 3 pixel matrix (from stage 1)
        S     : 3 x 3 sepia matrix (from stage 2)
        shape : (H, W, 3), for reshaping X back into a photo
    run() prints its numbers + a SANITY line and creates matplotlib figures
    (do NOT call plt.show() inside run; this script does that).

Usage:
    python run_pipeline.py              # pauses between stages
    python run_pipeline.py --no-pause   # runs straight through
"""
import importlib
import sys

import matplotlib.pyplot as plt

import stage1_matrix
import stage2_filter

plt.style.use("dark_background")

# (module name, title). Change a module name here if a teammate names theirs differently.
TEAM_STAGES = [
    (3, "stage3_rank",        "RREF / rank / inverse"),
    (4, "stage4_gramschmidt", "Orthogonal basis (Gram-Schmidt)"),
    (5, "stage5_projection",  "Projection"),
    (6, "stage6_lstsq",       "Least squares"),
    (7, "stage7_eigen",       "Eigenvalues / eigenvectors"),
    (8, "stage8_reduced",     "Reduced model (PCA)"),
]

PAUSE = "--no-pause" not in sys.argv


def banner(n, title):
    print("\n" + "=" * 60)
    print(f" STAGE {n}: {title}")
    print("=" * 60)


def pause():
    plt.show(block=False)
    plt.pause(0.1)
    if PAUSE:
        input("\n[press Enter for the next stage] ")
    plt.close("all")


def main():
    # Stage 1 (A)
    banner(1, "Matrix representation")
    X, shape = stage1_matrix.run()
    pause()

    # Stage 2 (A)
    banner(2, "Matrix multiplication")
    stage2_filter.run(X, shape)
    S = stage2_filter.SEPIA
    pause()

    # Stages 3-8 (teammates)
    for n, modname, title in TEAM_STAGES:
        banner(n, title)
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name != modname:      # a real import error inside their file
                raise
            print(f"  ({modname}.py not found yet, skipping)")
            continue
        mod.run(X, S, shape)
        pause()

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
