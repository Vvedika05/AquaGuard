"""
AquaGuard - Exploratory Data Analysis
-------------------------------------
Generates presentation-ready, colorful EDA outputs for the
Water Potability Classification project.

Run from the project root:
    python src/eda.py

Input:
    data/water_potability.csv

Output:
    outputs/eda/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "water_potability.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "eda"


# ============================================================
# AQUAGUARD COLOR PALETTE
# ============================================================

# Different features intentionally receive different colors.
# Potability classes use fixed semantic colors throughout:
#   Non-Potable -> red
#   Potable     -> teal/green
FEATURE_COLORS = [
    "#2563EB",  # blue
    "#06B6D4",  # cyan
    "#8B5CF6",  # purple
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#F97316",  # orange
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#6366F1",  # indigo
]

CLASS_COLORS = {
    0: "#EF4444",  # Non-Potable
    1: "#10B981",  # Potable
}

TEXT_COLOR = "#172033"
GRID_COLOR = "#CBD5E1"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def style_axes(ax, grid_axis="y"):
    """Apply a clean presentation style to a Matplotlib axis."""
    ax.set_axisbelow(True)

    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color=GRID_COLOR,
            alpha=0.38,
            linewidth=0.8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")

    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def style_title(ax, title, subtitle=None):
    """Apply consistent AquaGuard title styling."""
    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=14,
    )

    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#64748B",
        )


def class_name(value):
    """Convert binary target values into readable class names."""
    return "Non-Potable" if int(value) == 0 else "Potable"


def load_dataset(path=DEFAULT_DATA_PATH):
    """Load and validate the water potability dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Make sure water_potability.csv is inside the data folder."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if "Potability" not in df.columns:
        raise ValueError(
            "Expected target column 'Potability' was not found."
        )

    return df


def save_figure(fig, path, dpi=220):
    """Save a figure consistently and close it."""
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# MAIN EDA FUNCTION
# ============================================================

def run_eda(df, target="Potability", out_dir=DEFAULT_OUTPUT_DIR, train_df=None):
    """
    Run the complete EDA workflow.

    The analytical workflow is intentionally kept equivalent to the
    original project:
      1. Data quality
      2. Class distribution
      3. Missing values
      4. Feature distributions
      5. Boxplots
      6. Correlation heatmap
      7. Features by class
      8. Key feature relationships
      9. Pairwise relationships
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    numeric = df.select_dtypes(include="number").columns.tolist()
    features = [column for column in numeric if column != target]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    print("=" * 65)
    print("AQUAGUARD - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    print(f"Dataset shape : {df.shape}")
    print(f"Target        : {target}")
    print(f"Output folder : {out}")
    print()

    # ========================================================
    # 1. DATA QUALITY
    # ========================================================

    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [
            int(df[c].isna().sum()) for c in df.columns
        ],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 4)
            for c in df.columns
        ],
        "unique_values": [
            int(df[c].nunique(dropna=True)) for c in df.columns
        ],
        "constant_column": [
            bool(df[c].nunique(dropna=False) <= 1)
            for c in df.columns
        ],
    })

    quality.to_csv(
        out / "data_quality_summary.csv",
        index=False,
    )

    df.describe(include="all").transpose().to_csv(
        out / "descriptive_statistics.csv"
    )

    # Duplicate information
    duplicate_summary = pd.DataFrame({
        "metric": [
            "Total rows",
            "Total columns",
            "Duplicate rows",
            "Rows after duplicate removal",
        ],
        "value": [
            len(df),
            df.shape[1],
            int(df.duplicated().sum()),
            int(len(df) - df.duplicated().sum()),
        ],
    })

    duplicate_summary.to_csv(
        out / "duplicate_summary.csv",
        index=False,
    )

    # ========================================================
    # 2. CLASS DISTRIBUTION
    # ========================================================

    counts = df[target].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    labels = [class_name(value) for value in counts.index]
    colors = [
        CLASS_COLORS.get(int(value), FEATURE_COLORS[i])
        for i, value in enumerate(counts.index)
    ]

    bars = ax.bar(
        labels,
        counts.values,
        color=colors,
        width=0.58,
        edgecolor="white",
        linewidth=2,
    )

    style_title(
        ax,
        "Water Potability Class Distribution",
        "Target balance in the available water-quality samples",
    )

    ax.set_xlabel("Potability Class", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    style_axes(ax)

    total = counts.sum()

    for bar, value in zip(bars, counts.values):
        percentage = value / total * 100

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.012,
            f"{int(value):,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ax.set_ylim(0, counts.max() * 1.16)

    save_figure(
        fig,
        out / "01_class_distribution.png",
    )

    # ========================================================
    # 3. MISSING VALUES
    # ========================================================

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if len(missing) > 0:
        colors = [
            FEATURE_COLORS[i % len(FEATURE_COLORS)]
            for i in range(len(missing))
        ]

        bars = ax.barh(
            missing.index[::-1],
            missing.values[::-1],
            color=colors[::-1],
            edgecolor="white",
            linewidth=1.5,
        )

        style_title(
            ax,
            "Missing Values by Feature",
            "Fields requiring preprocessing before model training",
        )

        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Feature")
        style_axes(ax)

        max_missing = missing.max()

        for bar, value in zip(
            bars,
            missing.values[::-1],
        ):
            ax.text(
                bar.get_width() + max_missing * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=TEXT_COLOR,
            )

        ax.set_xlim(0, max_missing * 1.17)

    else:
        ax.text(
            0.5,
            0.5,
            "✓ No missing values detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#10B981",
        )
        ax.axis("off")

    save_figure(
        fig,
        out / "02_missing_values.png",
    )

    # ========================================================
    # 4. FEATURE DISTRIBUTIONS
    # ========================================================

    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 4.0 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for i, (ax, feature) in enumerate(
        zip(axes, features)
    ):
        values = df[feature].dropna()

        ax.hist(
            values,
            bins=30,
            color=FEATURE_COLORS[i % len(FEATURE_COLORS)],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.7,
        )

        median = values.median()

        ax.axvline(
            median,
            color="#172033",
            linestyle="--",
            linewidth=1.5,
        )

        ax.text(
            0.97,
            0.91,
            f"Median\n{median:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#172033",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#CBD5E1",
                alpha=0.9,
            ),
        )

        ax.set_title(
            f"Distribution of {feature}",
            fontsize=11.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        style_axes(ax)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        "Water Quality Feature Distributions",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.005,
    )

    save_figure(
        fig,
        out / "03_feature_distributions.png",
    )

    # ========================================================
    # 5. FEATURE BOXPLOTS / OUTLIERS
    # ========================================================

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 4.0 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for i, (ax, feature) in enumerate(
        zip(axes, features)
    ):
        values = df[feature].dropna()

        box = ax.boxplot(
            values,
            vert=False,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(
                facecolor=FEATURE_COLORS[i % len(FEATURE_COLORS)],
                alpha=0.62,
                edgecolor="#334155",
                linewidth=1.1,
            ),
            medianprops=dict(
                color="#172033",
                linewidth=2,
            ),
            whiskerprops=dict(
                color="#475569",
                linewidth=1.1,
            ),
            capprops=dict(
                color="#475569",
                linewidth=1.1,
            ),
            flierprops=dict(
                marker="o",
                markersize=3.2,
                markerfacecolor="#F43F5E",
                markeredgecolor="white",
                alpha=0.55,
            ),
        )

        ax.set_title(
            f"Boxplot of {feature}",
            fontsize=11.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        ax.set_xlabel(feature)
        style_axes(ax)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        "Outlier & Spread Analysis",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.005,
    )

    save_figure(
        fig,
        out / "04_feature_boxplots.png",
    )

    # ========================================================
    # 6. CORRELATION HEATMAP
    # ========================================================

    corr = df[numeric].corr()

    fig, ax = plt.subplots(
        figsize=(12, 9),
    )

    correlation_cmap = LinearSegmentedColormap.from_list(
        "aquaguard_correlation",
        [
            "#2563EB",
            "#93C5FD",
            "#F8FAFC",
            "#FDA4AF",
            "#DC2626",
        ],
        N=256,
    )

    image = ax.imshow(
        corr,
        cmap=correlation_cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Pearson Correlation",
        fontsize=10,
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(
        corr.columns,
        fontsize=9,
    )

    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]

            text_color = (
                "white"
                if abs(value) >= 0.55
                else TEXT_COLOR
            )

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=text_color,
            )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=16,
    )

    save_figure(
        fig,
        out / "05_correlation_heatmap.png",
    )

    # ========================================================
    # 7. FEATURES BY POTABILITY CLASS
    # ========================================================

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 4.0 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for ax, feature in zip(axes, features):

        non_potable = df.loc[
            df[target] == 0,
            feature,
        ].dropna()

        potable = df.loc[
            df[target] == 1,
            feature,
        ].dropna()

        box = ax.boxplot(
            [non_potable, potable],
            labels=["Non-Potable", "Potable"],
            patch_artist=True,
            widths=0.54,
            boxprops=dict(
                facecolor="white",
                edgecolor="#475569",
                linewidth=1.1,
            ),
            medianprops=dict(
                color="#172033",
                linewidth=2,
            ),
            whiskerprops=dict(
                color="#475569",
            ),
            capprops=dict(
                color="#475569",
            ),
            flierprops=dict(
                marker="o",
                markersize=3,
                markeredgecolor="white",
                alpha=0.5,
            ),
        )

        box["boxes"][0].set_facecolor(
            CLASS_COLORS[0]
        )
        box["boxes"][0].set_alpha(0.55)

        box["boxes"][1].set_facecolor(
            CLASS_COLORS[1]
        )
        box["boxes"][1].set_alpha(0.55)

        ax.set_title(
            f"{feature} by Potability Class",
            fontsize=11.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        ax.set_xlabel("Class")
        ax.set_ylabel(feature)

        style_axes(ax)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        "Feature Distributions Across Potability Classes",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.005,
    )

    save_figure(
        fig,
        out / "06_features_by_class.png",
    )

    # ========================================================
    # 8. KEY FEATURE RELATIONSHIPS
    # ========================================================

    target_corr = (
        corr[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )

    top = target_corr.head(
        min(4, len(target_corr))
    ).index.tolist()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    for ax, feature in zip(axes, top):

        for label, label_name in [
            (0, "Non-Potable"),
            (1, "Potable"),
        ]:
            subset = df.loc[
                df[target] == label,
                [feature, target],
            ].dropna()

            ax.scatter(
                subset[feature],
                subset[target],
                s=24,
                alpha=0.38,
                color=CLASS_COLORS[label],
                label=label_name,
                edgecolors="white",
                linewidths=0.25,
            )

        ax.set_title(
            f"{feature} vs {target}",
            fontsize=12.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Potability")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(
            ["Non-Potable", "Potable"]
        )

        style_axes(ax)

    for ax in axes[len(top):]:
        ax.axis("off")

    if top:
        axes[0].legend(
            frameon=False,
            fontsize=9,
        )

    fig.suptitle(
        "Key Feature Relationships with Potability",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.995,
    )

    save_figure(
        fig,
        out / "07_key_feature_relationships.png",
    )

    # ========================================================
    # 9. PAIRWISE RELATIONSHIPS
    # ========================================================

    pair = top[:4]

    if len(pair) >= 2:

        sample = df[
            pair + [target]
        ].dropna()

        # Keep the visualization readable.
        if len(sample) > 700:
            sample = sample.sample(
                700,
                random_state=42,
            )

        k = len(pair)

        fig, axes = plt.subplots(
            k,
            k,
            figsize=(13, 13),
        )

        axes = np.asarray(axes)

        for i, feature_y in enumerate(pair):

            for j, feature_x in enumerate(pair):

                ax = axes[i, j]

                if i == j:

                    ax.hist(
                        sample[feature_y],
                        bins=25,
                        color=FEATURE_COLORS[
                            i % len(FEATURE_COLORS)
                        ],
                        alpha=0.82,
                        edgecolor="white",
                    )

                else:

                    for label in [0, 1]:

                        subset = sample[
                            sample[target] == label
                        ]

                        ax.scatter(
                            subset[feature_x],
                            subset[feature_y],
                            s=13,
                            alpha=0.34,
                            color=CLASS_COLORS[label],
                            edgecolors="none",
                        )

                if i == k - 1:
                    ax.set_xlabel(
                        feature_x,
                        fontsize=8.5,
                    )
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(
                        feature_y,
                        fontsize=8.5,
                    )
                else:
                    ax.set_yticklabels([])

                style_axes(ax)

        fig.suptitle(
            "Pairwise Relationships of Selected Features",
            fontsize=18,
            fontweight="bold",
            color=TEXT_COLOR,
            y=0.995,
        )

        save_figure(
            fig,
            out / "08_pairwise_relationships.png",
            dpi=190,
        )

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "features_analyzed": features,
        "target": target,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
            if value > 0
        },
        "top_features_by_absolute_target_correlation": [
            {
                "feature": feature,
                "absolute_correlation": round(
                    float(target_corr[feature]),
                    6,
                ),
                "signed_correlation": round(
                    float(corr.loc[feature, target]),
                    6,
                ),
            }
            for feature in top
        ],
    }

    import json

    with open(
        out / "eda_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("EDA completed successfully.")
    print()
    print("Generated:")
    for file in sorted(out.iterdir()):
        print(f"  ✓ {file.name}")

    print()
    print("Top features by absolute Pearson correlation with target:")

    for feature in top:
        print(
            f"  {feature}: "
            f"{corr.loc[feature, target]:.4f}"
        )

    return {
        "numeric_features": features,
        "top_correlated_features": top,
    }


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

def main():
    df = load_dataset()
    run_eda(
        df=df,
        target="Potability",
        out_dir=DEFAULT_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
