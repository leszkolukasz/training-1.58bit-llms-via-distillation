import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from mlflow.entities import Run
from src.mlflow import client, get_run_by_name

matplotlib.use("qtagg")  # uv has some problems with Tkinter
sns.set_theme(style="white", context="paper")

SAVE_DIR = "./data/plots"


def get_metric_df(run: Run, metric_name: str, rolling_mean: int = 1):
    metrics = client.get_metric_history(run.info.run_id, metric_name)
    df = pl.DataFrame(
        {
            "step": [m.step for m in metrics],
            metric_name: [m.value for m in metrics],
        }
    ).filter(pl.col("step") != 0)

    if rolling_mean > 1:
        df = df.with_columns(
            pl.col(metric_name).rolling_mean(rolling_mean).alias(metric_name)
        )

    df = df.with_columns(pl.lit(run.info.run_name).alias("run_name"))

    return df


def plot_metric(
    fig: Figure,
    runs: list[Run],
    metric_name: str,
    pretty_run_names: list[str],
    rolling_mean: int = 1,
    x_lims: tuple = (-500, 17999),
):
    dfs = [get_metric_df(run, metric_name, rolling_mean=rolling_mean) for run in runs]
    combined_df = pl.concat(dfs).to_pandas()
    
    alpha = 1
    ax = fig.gca()
    #palette = sns.color_palette("colorblind", len(runs))
    for idx, (run, pretty_name) in enumerate(zip(runs, pretty_run_names)):
        run_df = combined_df[combined_df["run_name"] == run.info.run_name]
        
        sns.lineplot(
            data=run_df,
            x="step",
            y=metric_name,
            label=pretty_name,
            ax=ax,
            linewidth=2,
            #color=palette[idx % len(palette)],
            alpha=alpha,
            errorbar=None,
        )
        # alpha -= 0.5

    # Replace legend labels with pretty names
    handles, labels = ax.get_legend_handles_labels()
    run_names = [run.info.run_name for run in runs]
    label_map = dict(zip(run_names, pretty_run_names))
    new_labels = [label_map.get(label, label) for label in labels]

    ax.legend(handles=handles, labels=new_labels, title="", fontsize=16)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(x_lims[0], x_lims[1])

    fig.tight_layout()


def get_plot(
    run_names: list[str],
    metric_name: str,
    pretty_run_names: list[str] = None,
    rolling_mean: int = 1,
) -> Figure:
    if pretty_run_names is None:
        pretty_run_names = run_names

    runs = [get_run_by_name(run_name) for run_name in run_names]

    fig = plt.figure(figsize=(10, 6))
    plot_metric(fig, runs, metric_name, pretty_run_names, rolling_mean=rolling_mean)

    return fig

def render_loss_variants_grid():
    run_names = {
        "CrossEntropy": "quant_1_58b_impl_OneBit_loss_CrossEntropy",
        "KL": "quant_1_58b_impl_OneBit_loss_KL",
        "CAKL": "quant_1_58b_impl_OneBit_loss_CAKL",
        "Wasserstein": "quant_1_58b_impl_OneBit_loss_Wasserstein",
    }

    palette = {
        "CrossEntropy": "#1f77b4",   
        "KL": "#b96113",             
        "CAKL": "#248024",           
        "Wasserstein": "#941e1e",   
    }

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for idx, (loss_name, run_name) in enumerate(run_names.items()):
        run = get_run_by_name(run_name)
        df = get_metric_df(run, "train_loss_step", rolling_mean=5).to_pandas()

        ax = axs[idx]
        sns.lineplot(
            data=df,
            x="step",
            y="train_loss_step",
            ax=ax,
            label=loss_name,
            linewidth=2,
            errorbar=None,
            color=palette[loss_name],
        )

        ax.set_title(loss_name, fontsize=14)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Train Loss", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(left=0)  # Optional
        ax.set_autoscaley_on(True)
        ax.set_xlim(-500, None)

    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/loss_comparison_grid.png", dpi=300)
    return fig


def render_ZD_comparision():
    run_names = [
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_0_least",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_25_least",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_50_least",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_75_least",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_100_least",
    ]

    pretty_names = ["ZD 0% (baseline)", "ZD 25%", "ZD 50%", "ZD 75%", "ZD 100%"]

    fig = get_plot(
        run_names, "train_loss_step", pretty_run_names=pretty_names, rolling_mean=5
    )
    fig.gca().set_ylim(0.0, 15.0)
    fig.savefig(f"{SAVE_DIR}/ZD_least_important.png", dpi=300)

    return fig


def render_LIM_most_comparision():
    run_names = [
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_0_least",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_LIM_25_most",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_LIM_50_most",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_LIM_75_most",
        "quant_1_58b_impl_OneBit_loss_CrossEntropy_ZD_100_least",
    ]

    pretty_names = ["LIM 0% (baseline)", "LIM 25%", "LIM 50%", "LIM 75%", "LIM 100%"]

    fig = get_plot(
        run_names, "train_loss_step", pretty_run_names=pretty_names, rolling_mean=5
    )
    fig.gca().set_ylim(0.0, 15.0)
    fig.savefig(f"{SAVE_DIR}/LIM_most_important.png", dpi=300)

    return fig


def render_lr(layer: str):
    run_names = [
        f"quant_1_58b_impl_{layer}_loss_CrossEntropy_lr_0.01",
        f"quant_1_58b_impl_{layer}_loss_CrossEntropy_lr_0.001",
        f"quant_1_58b_impl_{layer}_loss_CrossEntropy_lr_0.0001",
        f"quant_1_58b_impl_{layer}_loss_CrossEntropy_lr_1e-05",
        f"quant_1_58b_impl_{layer}_loss_CrossEntropy_lr_1e-06",
    ]

    pretty_names = ["1e-02", "1e-03", "1e-04", "1e-05", "1e-06"]

    fig = get_plot(
        run_names, "train_loss_step", pretty_run_names=pretty_names, rolling_mean=5
    )
    fig.gca().set_ylim(0.0, 15.0)
    fig.savefig(f"{SAVE_DIR}/{layer}_lr.png", dpi=300)

    return fig

def render_explicit_quant(metric: str = "train_loss_step",):
    run_names = [
        "quant_1b_impl_BitNet_loss_CrossEntropy",
        "quant_1_58b_impl_BitNet_loss_CrossEntropy",
    ]

    pretty_names = ["1-bit", "1.58-bit"]

    fig = get_plot(
        run_names, metric, pretty_run_names=pretty_names, rolling_mean=5
    )
    
    if metric == "training_loss_step":
        fig.savefig(f"{SAVE_DIR}/quant_precision.png", dpi=300)
    elif metric == "flip_flop_step":
        fig.savefig(f"{SAVE_DIR}/quant_precision_ff.png", dpi=300)

    return fig
    
def render_explicit_layer(metric: str = "train_loss_step",):
    run_names = [
        "quant_1_58b_impl_OneBit_loss_CrossEntropy",
        "quant_1_58b_impl_BitNet_loss_CrossEntropy",
        "quant_1_58b_impl_FBI_loss_CrossEntropy",
    ]

    pretty_names = ["OneBit", "BitNet", "FBI",]

    fig = get_plot(
        run_names, metric, pretty_run_names=pretty_names, rolling_mean=5
    )
    fig.gca().set_ylim(None, 0.0125)
    if metric == "training_loss_step":
        fig.savefig(f"{SAVE_DIR}/quant_layer.png", dpi=300)
    elif metric == "flip_flop_step":
        fig.savefig(f"{SAVE_DIR}/quant_layer_ff.png", dpi=300)

    return fig

def render_explicit_loss(metric: str = "train_loss_step",):
    run_names = [
        "quant_1_58b_impl_OneBit_loss_CrossEntropy",
        "quant_1_58b_impl_OneBit_loss_KL",
        "quant_1_58b_impl_OneBit_loss_CAKL",
        "quant_1_58b_impl_OneBit_loss_Wasserstein",
    ]

    pretty_names = [
        "CE", "KL", "CAKL", "Wasserstein",
        ]

    fig = get_plot(
        run_names, metric, pretty_run_names=pretty_names, rolling_mean=5
    )
    fig.gca().set_ylim(-0.0001, 0.003)
    if metric == "training_loss_step":
        fig.savefig(f"{SAVE_DIR}/quant_loss.png", dpi=300)
    elif metric == "flip_flop_step":
        fig.savefig(f"{SAVE_DIR}/quant_loss_ff.png", dpi=300)

    return fig
    
    
if __name__ == "__main__":
    # fig = render_ZD_comparision()
    fig = render_explicit_layer(metric="flip_flop_step",)
    # fig = render_explicit_quant()
    # fig = render_explicit_loss(metric="flip_flop_step",)
    # fig = render_loss_variants_grid()
    fig.show()
