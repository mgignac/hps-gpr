from __future__ import annotations

from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import yaml


NOTE_DIR = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_pdf(
    pdf_path: Path,
    page_index: int,
    crop_fracs: tuple[float, float, float, float],
    out_path: Path,
    *,
    scale: float = 4.0,
) -> None:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    page_rect = page.rect
    x0, y0, x1, y1 = crop_fracs
    clip = fitz.Rect(
        page_rect.x0 + x0 * page_rect.width,
        page_rect.y0 + y0 * page_rect.height,
        page_rect.x0 + x1 * page_rect.width,
        page_rect.y0 + y1 * page_rect.height,
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    ensure_dir(out_path.parent)
    pix.save(out_path)


def stack_images(paths: list[Path], out_path: Path, *, bg: str = "white", pad: int = 20) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    width = max(img.width for img in imgs)
    normalized = []
    for img in imgs:
        if img.width != width:
            new_height = int(round(img.height * width / img.width))
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
        normalized.append(ImageOps.expand(img, border=2, fill="white"))
    total_height = sum(img.height for img in normalized) + pad * (len(normalized) - 1)
    canvas = Image.new("RGB", (width, total_height), color=bg)
    y = 0
    for img in normalized:
        canvas.paste(img, (0, y))
        y += img.height + pad
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def tile_images_horizontal(paths: list[Path], out_path: Path, *, bg: str = "white", pad: int = 24) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    height = max(img.height for img in imgs)
    normalized = []
    for img in imgs:
        if img.height != height:
            new_width = int(round(img.width * height / img.height))
            img = img.resize((new_width, height), Image.Resampling.LANCZOS)
        normalized.append(ImageOps.expand(img, border=2, fill="white"))
    total_width = sum(img.width for img in normalized) + pad * (len(normalized) - 1)
    canvas = Image.new("RGB", (total_width, height), color=bg)
    x = 0
    for img in normalized:
        canvas.paste(img, (x, 0))
        x += img.width + pad
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def make_pvalue_schematic(out_path: Path) -> None:
    x = np.linspace(-4.0, 4.0, 1000)
    y = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    obs = 1.15

    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    panels = [
        (r"$p_{\rm strong} = P(x \leq x_{\rm obs})$", x <= obs, "#4C72B0"),
        (r"$p_{\rm weak} = P(x \geq x_{\rm obs})$", x >= obs, "#DD8452"),
        (r"$p_{\rm two} = 2 \min(p_{\rm strong}, p_{\rm weak})$", x >= obs, "#55A868"),
    ]
    for ax, (title, mask, color) in zip(axs, panels):
        ax.plot(x, y, color="0.2", lw=2.0)
        ax.fill_between(x, 0.0, y, where=mask, color=color, alpha=0.85)
        if title.startswith(r"$p_{\rm two}"):
            ax.fill_between(x, 0.0, y, where=x <= -obs, color=color, alpha=0.35)
        ax.axvline(obs, color="black", lw=1.4, ls="--")
        ax.text(obs + 0.08, 0.34, r"$x_{\rm obs}$", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Test statistic / toy-limit summary variable")
        ax.set_ylabel("Density")
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(0.0, 0.43)
        ax.grid(alpha=0.25)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_constraints_panel(out_path: Path) -> None:
    entries = [
        ("BaBar prompt visible", 20.0, 10000.0, "#8172B3"),
        ("NA48/2 $\\pi^0 \\to \\gamma A'$", 10.0, 125.0, "#CCB974"),
        ("NA64 visible / X17-inspired", 1.0, 17.0, "#64B5CD"),
        ("HPS 2015 prompt", 19.0, 81.0, "#4C72B0"),
        ("HPS 2016 prompt", 39.0, 179.0, "#DD8452"),
        ("Current HPS GPR scope", 20.0, 250.0, "#55A868"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ypos = np.arange(len(entries))[::-1]
    for y, (label, xmin, xmax, color) in zip(ypos, entries):
        ax.hlines(y, xmin, xmax, color=color, lw=9.0, alpha=0.9)
        ax.scatter([xmin, xmax], [y, y], color=color, s=36, zorder=3)
        ax.text(xmax * 1.06, y, label, va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 2.0e4)
    ax.set_ylim(-0.8, len(entries) - 0.2)
    ax.set_xlabel(r"Approximate prompt-visible mass reach $m_{A'}$ [MeV]")
    ax.set_yticks([])
    ax.grid(alpha=0.25, axis="x", which="both")
    ax.set_title("Representative prompt-visible dark-photon constraints", fontsize=11.5)
    fig.text(
        0.035,
        0.03,
        "Comparison panel shows approximate mass reach, not exclusion depth. "
        "NA64 coverage is model-dependent in visible/X17-motivated interpretations.",
        fontsize=8.6,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.18)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_2021_parameterization_fig(out_path: Path) -> None:
    cfg_path = NOTE_DIR.parent / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    m = np.linspace(0.03, 0.25, 300)
    sigma_coeffs = cfg["sigma_coeffs_2021"]
    frad = float(cfg["frad_coeffs_2021"][0])
    penalty = float(cfg["radiative_penalty_frac_2021"])
    sigma = sum(float(c) * m**i for i, c in enumerate(sigma_coeffs))

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

    axs[0].plot(m * 1e3, sigma * 1e3, color="#009E73", lw=2.2)
    axs[0].set_xlabel(r"$m_{A'}$ [MeV]")
    axs[0].set_ylabel(r"$\sigma_m$ [MeV]")
    axs[0].set_title("2021 mass-resolution parameterization", fontsize=11)
    axs[0].grid(alpha=0.25)

    axs[1].plot(m * 1e3, np.full_like(m, frad), color="#0072B2", lw=2.2, label=r"Baseline $f_{\rm rad}$")
    axs[1].plot(m * 1e3, np.full_like(m, frad * (1.0 - penalty)), color="#D55E00", lw=2.0, ls="--",
                label=rf"Penalty-adjusted $(1-\delta_f)f_{{\rm rad}}$, $\delta_f={penalty:.2f}$")
    axs[1].set_xlabel(r"$m_{A'}$ [MeV]")
    axs[1].set_ylabel(r"$f_{\rm rad}$")
    axs[1].set_title("2021 radiative-fraction inputs", fontsize=11)
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="upper right", fontsize=8.5)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_2021_resolution_only_fig(out_path: Path) -> None:
    cfg_path = NOTE_DIR.parent / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    m = np.linspace(0.03, 0.25, 300)
    sigma_coeffs = cfg["sigma_coeffs_2021"]
    sigma = sum(float(c) * m**i for i, c in enumerate(sigma_coeffs))

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(m * 1e3, sigma * 1e3, color="#009E73", lw=2.2)
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"$\sigma_m$ [MeV]")
    ax.set_title("2021 mass-resolution parameterization", fontsize=11)
    ax.grid(alpha=0.25)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_2021_radiative_inputs_fig(out_path: Path) -> None:
    cfg_path = NOTE_DIR.parent / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    m = np.linspace(0.03, 0.25, 300)
    frad = float(cfg["frad_coeffs_2021"][0])
    penalty = float(cfg["radiative_penalty_frac_2021"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(m * 1e3, np.full_like(m, frad), color="#0072B2", lw=2.2, label=r"Baseline $f_{\rm rad}$")
    ax.plot(
        m * 1e3,
        np.full_like(m, frad * (1.0 - penalty)),
        color="#D55E00",
        lw=2.0,
        ls="--",
        label=rf"Penalty-adjusted $(1-\delta_f)f_{{\rm rad}}$, $\delta_f={penalty:.2f}$",
    )
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"$f_{\rm rad}$")
    ax.set_title("2021 radiative-fraction inputs", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_projection_placeholder(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    masses = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240], float)
    baseline = 1.8e-9 * np.exp(-0.007 * (masses - 20))
    proj = baseline / np.sqrt(np.where(masses < 130, 10.0, 100.0))
    ax.plot(masses, baseline, color="#4C72B0", lw=2.0, label="Current staged baseline")
    ax.plot(masses, proj, color="#D55E00", lw=2.2, ls="--", label="Projected full-luminosity reach")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"Projected 95\% CL upper limit on $\epsilon^2$")
    ax.set_title("Projected Unblinded Reach in $\epsilon^2$", fontsize=11.5)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.text(
        0.02,
        0.03,
        "Placeholder figure built from illustrative sqrt(L) rescaling.\n"
        "Final panel should be regenerated from combined UL-band CSV inputs.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92),
    )
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    crop_pdf(
        NOTE_DIR / "HPS_2016_Bump_Hunt_Internal_Note_note.pdf",
        4,
        (0.03, 0.02, 0.97, 0.35),
        NOTE_DIR / "apparatus_figs" / "hps_2015_2016_svt_baseline.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_Experiment_2022.pdf",
        7,
        (0.50, 0.62, 0.96, 0.74),
        NOTE_DIR / "apparatus_figs" / "hps_2019_2021_svt_upgrade.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_2015_PRL_2018.pdf",
        4,
        (0.04, 0.05, 0.49, 0.27),
        NOTE_DIR / "published_reference_figs" / "hps2015_published_limit.png",
    )
    crop_pdf(
        NOTE_DIR / "2016_HPS_Paper.pdf",
        16,
        (0.05, 0.40, 0.45, 0.58),
        NOTE_DIR / "published_reference_figs" / "hps2016_published_prompt_pvalue.png",
    )
    crop_pdf(
        NOTE_DIR / "2016_HPS_Paper.pdf",
        16,
        (0.54, 0.05, 0.97, 0.16),
        NOTE_DIR / "published_reference_figs" / "hps2016_published_prompt_limit.png",
    )
    crop_pdf(
        NOTE_DIR / "hps_2015_resonance_search_internal_note.pdf",
        29,
        (0.06, 0.05, 0.94, 0.58),
        NOTE_DIR / "resolution_figs" / "hps2015_mass_resolution_internal_fig24.png",
    )
    crop_pdf(
        NOTE_DIR / "hps_2015_resonance_search_internal_note.pdf",
        39,
        (0.09, 0.05, 0.90, 0.46),
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_internal_fig31.png",
    )
    crop_pdf(
        NOTE_DIR / "2015_radiative_radiative_fraction.pdf",
        0,
        (0.04, 0.07, 0.97, 0.92),
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_cross_section_review.png",
    )
    crop_pdf(
        NOTE_DIR / "2015_radiative_radiative_fraction.pdf",
        1,
        (0.08, 0.07, 0.96, 0.92),
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_review_right.png",
    )
    tile_images_horizontal(
        [
            NOTE_DIR / "normalization_figs" / "hps2015_radiative_cross_section_review.png",
            NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_review_right.png",
        ],
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_review.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_2016_Bump_Hunt_Internal_Note_note.pdf",
        37,
        (0.07, 0.04, 0.94, 0.31),
        NOTE_DIR / "resolution_figs" / "hps2016_mass_resolution_internal_fig29.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_2016_Bump_Hunt_Internal_Note_note.pdf",
        40,
        (0.07, 0.04, 0.93, 0.50),
        NOTE_DIR / "normalization_figs" / "hps2016_radiative_fraction_internal_fig30.png",
    )
    make_pvalue_schematic(NOTE_DIR / "methodology_figs" / "pvalue_tail_schematic.png")
    make_constraints_panel(NOTE_DIR / "context_figs" / "prompt_visible_constraints_panel.png")
    make_2021_parameterization_fig(NOTE_DIR / "resolution_figs" / "hps2021_resolution_and_frad.png")
    make_2021_resolution_only_fig(NOTE_DIR / "resolution_figs" / "hps2021_mass_resolution_parameterization.png")
    make_2021_radiative_inputs_fig(NOTE_DIR / "normalization_figs" / "hps2021_radiative_fraction_inputs.png")
    stack_images(
        [
            NOTE_DIR / "summary_combined_all_rad_penalty" / "2015_p0_local_global.png",
            NOTE_DIR / "summary_combined_all_rad_penalty" / "2016_p0_local_global.png",
            NOTE_DIR / "summary_combined_all_rad_penalty" / "2021_p0_local_global.png",
        ],
        NOTE_DIR / "methodology_figs" / "local_significances_across_datasets.png",
    )
    make_projection_placeholder(
        NOTE_DIR / "combined_search_figs" / "projected_unblinded_reach_eps2_placeholder.png"
    )


if __name__ == "__main__":
    main()
