from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "/home/suvihaara/Documents/PhD/CODE/audio_data_benchmarking_mml_lab/experiments/vocalset_non_overlapping/metrics/vtc_vocalset_non_overlapping_silencetrimmed/vtc_vocalset_non_overlapping_full_metrics_outline.csv"

TRAD_ML_LATENCY_CSV = Path(CSV_PATH).parent / "trad_ml_latency.csv"

COST_COLUMN = "End To End Latency (ms/window) (bs=1)"
 # Reported numbers are on the evaluation (held-out) split. No hyperparameters,
# learning rates, or epochs were selected using this split.
PERFORMANCE_COLUMN = "Val Balanced Accuracy"  # corresponds to evaluation split in this benchmark

OUTPUT_BACC = "supplementary_heatmaps_bacc.pdf"
OUTPUT_F1 = "supplementary_heatmaps_f1.pdf"
OUTPUT_COST = "supplementary_cost_vs_performance.pdf"
OUTPUT_COMPACT_BACC = "supplementary_heatmap_compact_bacc.pdf"
OUTPUT_COMPACT_F1 = "supplementary_heatmap_compact_f1.pdf"

OUTPUT_BACC_PNG = "supplementary_heatmaps_bacc.png"
OUTPUT_F1_PNG = "supplementary_heatmaps_f1.png"
OUTPUT_COST_PNG = "supplementary_cost_vs_performance.png"
OUTPUT_COMPACT_BACC_PNG = "supplementary_heatmap_compact_bacc.png"
OUTPUT_COMPACT_F1_PNG = "supplementary_heatmap_compact_f1.png"


# Ordering
CLASSIFIER_ORDER = ["knn", "rf", "mlp", "svm"]
REPRESENTATION_ORDER = ["mfcc", "opensmile", "vggish", "whisper", "ms-clap", "laion-clap"]

CLASSIFIER_LABELS = {
    "knn": "KNN",
    "rf": "RF",
    "mlp": "MLP",
    "svm": "SVM",
}

REPRESENTATION_LABELS = {
    "mfcc": "MFCC",
    "opensmile": "OpenSMILE",
    "vggish": "VGGish",
    "whisper": "Whisper",
    "ms-clap": "MS-CLAP",
    "laion-clap": "LAION_CLAP"
}

CLASSIFIER_MARKERS = {
    "knn": "o",
    "rf": "s",
    "mlp": "^",
    "svm": "D",
}

FINETUNING_MODELS = [
    "clap_head_only",
    "clap_full",
    "whisper_head_only",
    "whisper_full",
]

FINETUNING_LABELS = {
    "clap_head_only": "CLAP (frozen)",
    "clap_full": "CLAP (fine-tuned)",
    "whisper_head_only": "Whisper (frozen)",
    "whisper_full": "Whisper (fine-tuned)",
}

FINETUNING_COLORS = {
    "clap_head_only": "crimson",
    "clap_full": "darkorange",
    "whisper_head_only": "royalblue",
    "whisper_full": "black",
}

# The representation can appear under different names depending on whether the
# metrics come from a run (dataset name) or from a hand-assembled outline.
REPRESENTATION_ALIASES = {
    "mfcc": "mfcc",
    "mfcc-13d": "mfcc",
    "opensmile": "opensmile",
    "opensmile-compare-2016": "opensmile",
    "vggish": "vggish",
    "whisper": "whisper",
    "whisper-base": "whisper",
    "ms-clap": "ms-clap",
    "ms_clap": "ms-clap",
    "ms-clap-2023": "ms-clap",
    "laion-clap": "laion-clap",
    "laion_clap": "laion-clap",
    "laion-clap-htsat-fused": "laion-clap",
    "clap": "laion-clap",
}


def normalise_representation(value):
    key = str(value).strip().lower()
    return REPRESENTATION_ALIASES.get(key, key)


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

df["_representation"] = df["Input Data"].map(normalise_representation)


# ============================================================
# Traditional ML latency (end-to-end cost)
# ============================================================


def merge_traditional_latencies(frame):
    """Fill traditional-ML latency columns from trad_ml_latency.csv.

    The metrics outline only carries end-to-end costs for the fine-tuned
    models; the traditional-ML costs live in the dedicated latency CSV.
    Existing values are kept, only missing cells are filled.
    """

    if not TRAD_ML_LATENCY_CSV.exists():
        print(f"Traditional-ML latency CSV not found: {TRAD_ML_LATENCY_CSV}")
        return frame

    latency = pd.read_csv(TRAD_ML_LATENCY_CSV)
    latency.columns = latency.columns.str.strip()
    latency["_representation"] = latency["Input Data"].map(
        normalise_representation
    )

    identity = {"Model", "Input Data", "Block", "_representation"}
    shared_columns = [
        column
        for column in latency.columns
        if column in frame.columns and column not in identity
    ]

    lookup = latency.drop_duplicates(["Model", "_representation"]).set_index(
        ["Model", "_representation"]
    )

    frame = frame.copy()
    traditional = frame["Model"].isin(CLASSIFIER_ORDER)
    keys = list(zip(frame["Model"], frame["_representation"]))

    for column in shared_columns:
        values = lookup[column].to_dict()
        filled = pd.Series(
            [values.get(key) for key in keys],
            index=frame.index,
            dtype="object",
        ).infer_objects()
        mask = traditional & frame[column].isna() & filled.notna()
        frame.loc[mask, column] = filled[mask]

    print(
        f"Traditional-ML latency CSV: {TRAD_ML_LATENCY_CSV} "
        f"({len(latency)} rows, {len(shared_columns)} columns merged)"
    )
    return frame


df = merge_traditional_latencies(df)


# ============================================================
# Traditional ML heatmap
# ============================================================


def make_traditional_matrix(metric):

    traditional = df[
        df["Model"].isin(CLASSIFIER_ORDER)
        & df["_representation"].isin(REPRESENTATION_ORDER)
    ].copy()

    matrix = traditional.pivot(
        index="Model",
        columns="_representation",
        values=metric,
    )

    matrix = matrix.reindex(
        index=CLASSIFIER_ORDER,
        columns=REPRESENTATION_ORDER,
    )

    matrix.index = [CLASSIFIER_LABELS[x] for x in matrix.index]

    matrix.columns = [REPRESENTATION_LABELS[x] for x in matrix.columns]

    return matrix


# ============================================================
# CLAP / Whisper fine-tuning heatmap
# ============================================================


def make_finetuning_matrix(metric):

    # Map CSV model names to readable backbone/training labels
    pretrained = df[
        df["Model"].isin(
            [
                "clap_head_only",
                "whisper_head_only",
                "clap_full",
                "whisper_full",
            ]
        )
    ].copy()

    # Determine backbone
    pretrained["Backbone"] = pretrained["Model"].map(
        {
            "clap_head_only": "LAION-CLAP",
            "clap_full": "LAION-CLAP",
            "whisper_head_only": "Whisper",
            "whisper_full": "Whisper",
        }
    )

    # Determine training strategy
    pretrained["Training"] = pretrained["Model"].map(
        {
            "clap_head_only": "Frozen encoder",
            "clap_full": "Fine-tuned encoder",
            "whisper_head_only": "Frozen encoder",
            "whisper_full": "Fine-tuned encoder",
        }
    )

    matrix = pretrained.pivot(
        index="Backbone",
        columns="Training",
        values=metric,
    )

    matrix = matrix.reindex(
        index=["LAION-CLAP", "Whisper"],
        columns=["Frozen encoder", "Fine-tuned encoder"],
    )

    return matrix


# ============================================================
# Plot function
# ============================================================


def make_figure(metric, output_pdf, output_png):

    traditional_matrix = make_traditional_matrix(metric)
    # Add means the same way as the compact version
    traditional_matrix = traditional_matrix.transpose()
    traditional_matrix.loc["Mean"] = traditional_matrix.mean(axis=0)
    traditional_matrix["Mean"] = traditional_matrix.mean(axis=1)
    traditional_matrix = traditional_matrix.transpose()

    finetuning_matrix = make_finetuning_matrix(metric)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.0),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    # --------------------------------------------------------
    # Panel (a): Classifier × representation
    # --------------------------------------------------------

    n_rows_a, n_cols_a = traditional_matrix.shape
    sns.heatmap(
        traditional_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={
            "fontsize": 7,
        },
        ax=axes[0],
    )

    # Separate the marginal means from the per-configuration cells
    axes[0].axhline(n_rows_a - 1, color="black", linewidth=1.0)
    axes[0].axvline(n_cols_a - 1, color="black", linewidth=1.0)

    axes[0].set_xlabel("Representation", fontsize=9)
    axes[0].set_ylabel("Classifier", fontsize=9)

    axes[0].tick_params(
        axis="x",
        labelrotation=0,
        labelsize=8,
    )

    axes[0].tick_params(
        axis="y",
        labelrotation=0,
        labelsize=8,
    )

    axes[0].get_xticklabels()[-1].set_fontweight("bold")
    axes[0].get_yticklabels()[-1].set_fontweight("bold")

    axes[0].set_title(
        "(a) Classifier × Representation",
        fontsize=9,
        pad=8,
    )

    # --------------------------------------------------------
    # Panel (b): Backbone × training strategy
    # --------------------------------------------------------

    sns.heatmap(
        finetuning_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={
            "fontsize": 7,
        },
        ax=axes[1],
    )

    axes[1].set_xlabel("Training strategy", fontsize=9)
    axes[1].set_ylabel("Backbone", fontsize=9)

    axes[1].tick_params(
        axis="x",
        labelrotation=25,
        labelsize=7,
    )

    axes[1].tick_params(
        axis="y",
        labelrotation=0,
        labelsize=8,
    )

    axes[1].set_title(
        "(b) Backbone × Training Strategy",
        fontsize=9,
        pad=8,
    )

    # --------------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------------

    # Create a ScalarMappable purely for the shared colorbar
    import matplotlib as mpl

    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=1,
    )

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap="coolwarm",
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        fraction=0.025,
        pad=0.03,
    )

    cbar_label = metric
    if metric.startswith("Val "):
        cbar_label = metric.replace("Val ", "Eval ")
    else:
        cbar_label = metric.replace("Validation", "Evaluation")
    cbar.set_label(
        cbar_label,
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    # --------------------------------------------------------
    # Overall layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.08,
        right=0.80,
        bottom=0.20,
        top=0.84,
        wspace=0.4,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)


# ============================================================
# Compact representation x classifier heatmap (with row/column means)
# ============================================================


def make_compact_matrix(metric):
    """Representation x classifier matrix with marginal means.

    Rows are representations (plus a mean-of-representations row), columns
    are classifiers (plus a mean-of-classifiers column).
    """

    matrix = make_traditional_matrix(metric).transpose()
    matrix.loc["Mean"] = matrix.mean(axis=0)
    matrix["Mean"] = matrix.mean(axis=1)
    return matrix


def make_compact_figure(metric, output_pdf, output_png):
    matrix = make_compact_matrix(metric)

    n_rows, n_cols = matrix.shape

    fig, ax = plt.subplots(figsize=(5.4, 4.1))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 7},
        cbar_kws={
            "label": metric,
            "fraction": 0.046,
            "pad": 0.03,
        },
        ax=ax,
    )

    # Separate the marginal means from the per-configuration cells
    ax.axhline(n_rows - 1, color="black", linewidth=1.0)
    ax.axvline(n_cols - 1, color="black", linewidth=1.0)

    ax.set_xlabel("Classifier", fontsize=9)
    ax.set_ylabel("Representation", fontsize=9)

    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)

    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_fontweight("bold")

    ax.set_title(metric.replace("Val ", "Eval ") + " (held-out)", fontsize=9, pad=8)

    fig.tight_layout()

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")

    plt.show()

    plt.close(fig)


# ============================================================
# Cost vs performance (all models on one measured inference latency axis)
# ============================================================


def make_cost_performance_figure(output_pdf, output_png):
    """Single scatter: measured inference latency vs balanced accuracy, all models."""

    scatter = df[
        df[COST_COLUMN].notna() & df[PERFORMANCE_COLUMN].notna()
    ].copy()
    scatter["_cost"] = scatter[COST_COLUMN].astype(float)
    scatter["_performance"] = scatter[PERFORMANCE_COLUMN].astype(float)
    scatter = scatter[(scatter["_cost"] > 0) & (scatter["_performance"] > 0)]

    traditional = scatter[scatter["Model"].isin(CLASSIFIER_ORDER)]
    finetuning = scatter[scatter["Model"].isin(FINETUNING_MODELS)]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    palette = dict(
        zip(
            REPRESENTATION_ORDER,
            sns.color_palette("magma", n_colors=len(REPRESENTATION_ORDER)),
        )
    )

    # Traditional ML: colour = representation, marker = classifier
    for representation in REPRESENTATION_ORDER:
        subset = traditional[traditional["_representation"] == representation]
        if subset.empty:
            continue
        for classifier in CLASSIFIER_ORDER:
            points = subset[subset["Model"] == classifier]
            if points.empty:
                continue
            ax.scatter(
                points["_cost"],
                points["_performance"],
                s=95,
                marker=CLASSIFIER_MARKERS[classifier],
                color=palette[representation],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    # Frozen / fine-tuned encoders (one colour per model, no labels)
    for model in FINETUNING_MODELS:
        points = finetuning[finetuning["Model"] == model]
        if points.empty:
            continue
        ax.scatter(
            points["_cost"],
            points["_performance"],
            s=330,
            marker="*",
            color=FINETUNING_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    # Flag the single-query KNN outlier so the long tail is not misread.
    outlier = traditional[
        (traditional["Model"] == "knn")
        & (traditional["_representation"] == "opensmile")
    ]
    if not outlier.empty:
        row = outlier.iloc[0]
        ax.annotate(
            f"KNN + OpenSMILE ({row['_cost']:.0f} ms)",
            (row["_cost"], row["_performance"]),
            textcoords="offset points",
            xytext=(-8, 8),
            ha="right",
            fontsize=6.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Measured Inference Latency (ms/window, batch size 1, log scale)",
        fontsize=9,
    )
    ax.set_ylabel("Evaluation balanced accuracy (held-out)", fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.3)
    ax.set_title("Measured Inference Latency vs Evaluation Performance (held-out split)", fontsize=10, pad=8)

    # Legend (two rows below the axes)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[representation],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=10,
            label=REPRESENTATION_LABELS[representation],
        )
        for representation in REPRESENTATION_ORDER
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=CLASSIFIER_MARKERS[classifier],
            linestyle="none",
            markerfacecolor="0.75",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=10,
            label=CLASSIFIER_LABELS[classifier],
        )
        for classifier in CLASSIFIER_ORDER
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markerfacecolor=FINETUNING_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=15,
            label=FINETUNING_LABELS[model],
        )
        for model in FINETUNING_MODELS
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        fontsize=9,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.6,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.29,
        top=0.92,
    )

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")

    plt.show()

    plt.close(fig)


# ============================================================
# Generate figures
# ============================================================

make_figure(
    metric="Val Balanced Accuracy",
    output_pdf=OUTPUT_BACC,
    output_png=OUTPUT_BACC_PNG,
)

make_figure(
    metric="Val F1",
    output_pdf=OUTPUT_F1,
    output_png=OUTPUT_F1_PNG,
)

make_cost_performance_figure(
    output_pdf=OUTPUT_COST,
    output_png=OUTPUT_COST_PNG,
)

# make_compact_figure(
#     metric="Val Balanced Accuracy",
#     output_pdf=OUTPUT_COMPACT_BACC,
#     output_png=OUTPUT_COMPACT_BACC_PNG,
# )

# make_compact_figure(
#     metric="Val F1",
#     output_pdf=OUTPUT_COMPACT_F1,
#     output_png=OUTPUT_COMPACT_F1_PNG,
# )
