"""Shared plot style for the whole project (dark background, one palette).

Stages 4 and 5 import  COLORS  and  apply_style  from here.
"""
import matplotlib.pyplot as plt

COLORS = {
    "bg":         "#111111",   # plot background
    "text":       "#eeeeee",
    "brightness": "#ffd166",   # brightness axis / grayscale
    "color1":     "#ef476f",   # first color-difference axis
    "color2":     "#06d6a0",   # second color-difference axis
    "accent":     "#4cc9f0",
}


def apply_style():
    """Call before making a figure. Matches the dark theme used by pipeline.py."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
    })

