"""
CL Spectroscopy Pipeline
Processes .sur files from a directory tree, separating spectra, CL images,
and SEM images, then applies standard preprocessing and analysis.
"""

from pathlib import Path
import numpy as np
import hyperspy.api as hs
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────
home_path = Path.home()
data_root = home_path / 'Library/CloudStorage/OneDrive-UniversityofCambridge/Desktop/OneDrive/project_data/raw/2026/04_02_2026/ZEKI-INP-SPAD'    # <-- change to your data root
plots_root = Path("../plots")
WAVELENGTH_UNIT = "nm"
REBIN_FACTOR = 2                  # spectral binning (set to 1 to disable)
# ─────────────────────────────────────────────────────────────────────────────


def get_save_dir(path: Path) -> Path:
    """Mirror folder structure under plots_root."""
    folder_name = path.parent.name
    save_dir = plots_root / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def classify_signal(s) -> str:
    """Return 'hyperspectral', 'spectrum', 'cl_image', or 'sem_image'."""
    ndim = s.data.ndim
    axes = str(s.axes_manager).lower()
    if ndim == 3:
        return "hyperspectral"
    elif ndim == 1 or "wavelength" in axes or "energy" in axes:
        return "spectrum"
    else:
        # Heuristic: CL images are typically brighter / different metadata
        # Adjust this logic based on your file naming conventions if needed
        name = s.metadata.get_item("General.original_filename", "").lower()
        if "cl" in name:
            return "cl_image"
        return "sem_image"


def process_hyperspectral(s, path: Path):
    """Full preprocessing + analysis pipeline for a hyperspectral CL cube."""
    save_dir = get_save_dir(path)
    stem = path.stem
    print(f"  [hyperspectral] {path.name}")

    # 1. Axis calibration (skip if already set correctly)
    wl_axis = s.axes_manager[-1]
    if wl_axis.units in ("", "undefined", None):
        wl_axis.name = "Wavelength"
        wl_axis.units = WAVELENGTH_UNIT
        print("    ⚠ Wavelength axis units not set — please verify calibration.")

    # 2. Spike removal
    try:
        s.spikes_removal_tool(interactive=False)
    except Exception:
        print("    ⚠ Automatic spike removal failed — skipping.")

    # 3. Background subtraction
    try:
        s = s.remove_background(interactive=False)
    except Exception:
        print("    ⚠ Background subtraction failed — skipping.")

    # 4. Spectral rebinning
    if REBIN_FACTOR > 1:
        s = s.rebin(scale=[1, 1, REBIN_FACTOR])

    # 5. Panchromatic CL image
    pan_cl = s.sum(axis=-1)
    pan_cl.plot()
    plt.title(f"Panchromatic CL — {stem}")
    plt.savefig(save_dir / f"{stem}_pan_cl.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # 6. Mean spectrum
    mean_spec = s.mean()
    mean_spec.plot()
    plt.title(f"Mean Spectrum — {stem}")
    plt.savefig(save_dir / f"{stem}_mean_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # 7. Peak wavelength map
    wl_values = s.axes_manager[-1].axis
    peak_map = wl_values[s.data.argmax(axis=-1)]
    fig, ax = plt.subplots()
    im = ax.imshow(peak_map, cmap="plasma")
    plt.colorbar(im, ax=ax, label=f"Peak wavelength ({WAVELENGTH_UNIT})")
    ax.set_title(f"Peak Wavelength Map — {stem}")
    fig.savefig(save_dir / f"{stem}_peak_map.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # 8. PCA decomposition
    try:
        s.decomposition(algorithm="SVD")
        s.plot_explained_variance_ratio()
        plt.savefig(save_dir / f"{stem}_pca_variance.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        s.plot_decomposition_results()
        plt.savefig(save_dir / f"{stem}_pca_results.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"    ⚠ PCA failed: {e}")

    return s


def process_image(s, path: Path, kind: str):
    """Save a CL or SEM image."""
    save_dir = get_save_dir(path)
    stem = path.stem
    print(f"  [{kind}] {path.name}")

    s.plot()
    plt.title(f"{kind.upper()} — {stem}")
    plt.savefig(save_dir / f"{stem}_{kind}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def process_spectrum(s, path: Path):
    """Save a single 1D spectrum."""
    save_dir = get_save_dir(path)
    stem = path.stem
    print(f"  [spectrum] {path.name}")

    s.plot()
    plt.title(f"Spectrum — {stem}")
    plt.savefig(save_dir / f"{stem}_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    sur_files = sorted(data_root.rglob("*.sur"))
    print(f"Found {len(sur_files)} .sur files under {data_root}\n")

    for path in sur_files:
        print(f"Loading: {path.relative_to(data_root)}")
        try:
            s = hs.load(path)
            kind = classify_signal(s)

            if kind == "hyperspectral":
                process_hyperspectral(s, path)
            elif kind == "spectrum":
                process_spectrum(s, path)
            elif kind == "cl_image":
                process_image(s, path, "cl_image")
            else:
                process_image(s, path, "sem_image")

        except Exception as e:
            print(f"  ✗ Failed to process {path.name}: {e}")

    print("\nDone. Figures saved to:", plots_root.resolve())


if __name__ == "__main__":
    run()