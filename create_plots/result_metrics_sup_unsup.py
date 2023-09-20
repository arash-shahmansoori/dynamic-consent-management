import torch
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib import style

from .load_metrics_supervised import load_metrics_sup
from .load_metrics_supervised_scratch import load_metrics_sup_scratch
from .load_metrics_unsupervised import load_metrics_unsup


def create_time_axis(x, p):
    time_axis = []
    ts = 0
    for i in range(p):
        ts += x[i]
        time_axis.append(ts)

    return time_axis


train_dvec_modes = {
    "train_dvec_literature": "train_dvec",
    "train_dvec_adapted": "train_dvec_adapted",
    "train_dvec_proposed": "train_dvec_proposed",
}


def plot_metrics_sup_unsup(
    spk_per_bkt_collection,
    agent,
    args,
    hparams,
):
    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    out_unsup_literature = load_metrics_unsup(
        args,
        hparams,
        spk_per_bkt_collection[0],
        train_dvec_modes["train_dvec_literature"],
        agent,
    )

    out_unsup = load_metrics_unsup(
        args,
        hparams,
        spk_per_bkt_collection[1],
        train_dvec_modes["train_dvec_proposed"],
        agent,
    )

    out_sup_literature = load_metrics_sup_scratch(args, agent)

    out_sup = load_metrics_sup(
        args,
        hparams,
        spk_per_bkt_collection[1],
        train_dvec_modes["train_dvec_proposed"],
        agent,
    )

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Plot/save the results
    style.use("ggplot")

    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Create time axis for the training plots
    time_axis_unsup_literature = create_time_axis(
        out_unsup_literature["elapsed_time"], len(out_unsup_literature["elapsed_time"])
    )
    time_axis_sup_literature = create_time_axis(
        out_sup_literature["elapsed_time"], len(out_sup_literature["elapsed_time"])
    )
    time_axis_sup = create_time_axis(
        out_sup["elapsed_time"], len(out_sup["elapsed_time"])
    )
    time_axis_unsup = create_time_axis(
        out_unsup["elapsed_time"], len(out_unsup["elapsed_time"])
    )

    elapsed_time_proposed_sup = torch.tensor(time_axis_sup).view(-1).max().item() / 60
    elapsed_time_proposed_unsup = (
        torch.tensor(time_axis_unsup).view(-1).max().item() / 60
    )
    elapsed_time_unsup_literature = (
        torch.tensor(time_axis_unsup_literature).view(-1).max().item() / 60
    )
    elapsed_time_sup_literature = (
        torch.tensor(time_axis_sup_literature).view(-1).max().item() / 60
    )

    elapsed_proposed_sup = int(round(elapsed_time_proposed_sup, ndigits=0))
    elapsed_proposed_unsup = int(round(elapsed_time_proposed_unsup, ndigits=0))

    elapsed_unsup_literature = int(round(elapsed_time_unsup_literature, ndigits=0))
    elapsed_sup_literature = int(round(elapsed_time_sup_literature, ndigits=0))

    # Accuracies
    ax1.plot(
        torch.tensor(time_axis_unsup_literature) / 60,
        out_unsup_literature["val_acc"],
        "--",
        color="black",
        label=f"Literature",
    )
    ax1.plot(
        torch.tensor(time_axis_sup_literature) / 60,
        out_sup_literature["val_acc"],
        "-",
        color="brown",
        label=f"Literature sup",
    )
    ax1.plot(
        torch.tensor(time_axis_unsup) / 60,
        out_unsup["val_acc"]["7"],
        "-",
        marker="o",
        color="blue",
        label=f"Proposed unsup",
        ms=4,
    )
    ax1.plot(
        torch.tensor(time_axis_sup) / 60,
        out_sup["val_acc"]["7"],
        "-",
        marker="x",
        color="green",
        label=f"Proposed sup",
        ms=4,
    )

    # Losses
    ax2.plot(
        torch.tensor(time_axis_unsup_literature) / 60,
        out_unsup_literature["val_loss"],
        "--",
        color="black",
        label=f"Literature",
    )
    ax2.plot(
        torch.tensor(time_axis_sup_literature) / 60,
        out_sup_literature["val_loss"],
        "-",
        color="brown",
        label=f"Literature sup",
    )
    ax2.plot(
        torch.tensor(time_axis_unsup) / 60,
        out_unsup["val_loss"]["7"],
        "-",
        marker="o",
        color="blue",
        label=r"Proposed unsup",
        ms=4,
    )
    ax2.plot(
        torch.tensor(time_axis_sup) / 60,
        out_sup["val_loss"]["7"],
        "-",
        marker="x",
        color="green",
        label=r"Proposed sup",
        ms=4,
    )

    ax1.set_ylabel(r"Accuracy")
    ax2.set_ylabel(r"Loss")

    ax1.set_xlabel(r"Total elapsed time (min)")
    ax2.set_xlabel(r"Total elapsed time (min)")

    ax2.set_ylim(0, 5)

    ax1.legend(prop={"size": 11})
    ax2.legend(prop={"size": 11})

    plt.tight_layout()

    # fig.savefig(
    #     f"{result_dir_plot_path}/acc_loss_eval_sup_unsup.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     dpi=1200,
    # )

    plt.show()
    plt.close()
