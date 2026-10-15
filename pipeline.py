"""Main staged script (owner: A, pipeline lead).

Runs the demo stages in order and waits for Enter between them.
Stages 1-2 are A's. The other stages live in teammates' files and are
imported if they exist; a missing file is skipped so the pipeline always runs.

Each teammate file should have a function   run(X, S, shape)
    X     : N x 3 pixel matrix (from stage 1)
    S     : 3 x 3 sepia matrix (from stage 2)
    shape : (H, W, 3), for turning X back into a photo
A run() that takes fewer arguments (e.g. run(X, S) or run(X)) also works.
Do NOT call plt.show() inside run(); this script shows the figures.
Put any test code under  if __name__ == "__main__":  so importing is safe.

Usage:
    python pipeline.py              # pauses between stages
    python pipeline.py --no-pause   # runs straight through
"""
import importlib
import inspect
import sys
import traceback

import matplotlib.pyplot as plt

import stage1_matrix
import stage2_filter

plt.style.use("dark_background")

# (stage label, module name, title). Edit a module name here if a file is renamed.
TEAM_STAGES = [
    ("3",   "stage3_rref_rank_inverse", "RREF / rank / inverse"),
    ("4",   "stage4_gramschmidt",       "Orthogonal basis (Gram-Schmidt)"),
    ("5",   "stage5_projection",        "Projection"),
    ("6",   "stage6_least_squares",             "Least squares"),
    ("7-8", "stage7_8_eigen_pca",       "Eigenvalues / eigenvectors and reduced model (PCA)"),
]

PAUSE = "--no-pause" not in sys.argv


def banner(label, title):
    print("\n" + "=" * 60)
    print(f" STAGE {label}: {title}")
    print("=" * 60)


def pause():
    plt.show(block=False)
    plt.pause(0.1)
    if PAUSE:
        input("\n[press Enter for the next stage] ")
    plt.close("all")


def call_run(mod, X, S, shape):
    """Call mod.run with as many of (X, S, shape) as it accepts."""
    fn = getattr(mod, "run", None)
    if fn is None:
        funcs = [n for n, f in vars(mod).items()
                 if inspect.isfunction(f) and f.__module__ == mod.__name__
                 and not n.startswith("_")]
        print(f"  ({mod.__name__}.py has no run() function. Functions found: {funcs})")
        return False
    params = inspect.signature(fn).parameters.values()
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        n = 3
    else:
        n = len([p for p in params
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    fn(*(X, S, shape)[:n])
    return True


def main():
    # Stage 1 (A)
    banner("1", "Matrix representation")
    X, shape = stage1_matrix.run()
    pause()

    # Stage 2 (A)
    banner("2", "Matrix multiplication")
    stage2_filter.run(X, shape)
    S = stage2_filter.SEPIA
    pause()

    # Teammates' stages
    for label, modname, title in TEAM_STAGES:
        banner(label, title)
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name != modname:          # a real import error inside their file
                print(f"  (error while importing {modname}.py: {e})")
            else:
                print(f"  ({modname}.py not found yet, skipping)")
            continue
        except Exception:
            print(f"  (error while importing {modname}.py:)")
            traceback.print_exc()
            continue
        try:
            if call_run(mod, X, S, shape):
                pause()
        except Exception:
            print(f"  (stage {label} crashed, moving on:)")
            traceback.print_exc()
            plt.close("all")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
