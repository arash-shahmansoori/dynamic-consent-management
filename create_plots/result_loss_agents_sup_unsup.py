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


def plot_loss_agents_sup_unsup(
    spk_per_bkt_collection,
    agents,
    args,
    hparams,
):
    train_dvec_modes = {
        "train_dvec_literature": "train_dvec",
        "train_dvec_adapted": "train_dvec_adapted",
        "train_dvec_proposed": "train_dvec_proposed",
    }

    elapsed_time_sup_agnts = {}
    elapsed_time_unsup_agnts = {}
    elapsed_time_unsup_literature_agnts = {}
    elapsed_time_sup_literature_agnts = {}

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Plot/save the results
    style.use("ggplot")

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    new_axs = [nu for nv in axs for nu in nv]

    for indx in range(len(agents)):
        out_unsup_literature = load_metrics_unsup(
            args,
            hparams,
            spk_per_bkt_collection[0],
            train_dvec_modes["train_dvec_literature"],
            agents[indx],
        )

        out_unsup = load_metrics_unsup(
            args,
            hparams,
            spk_per_bkt_collection[1],
            train_dvec_modes["train_dvec_proposed"],
            agents[indx],
        )

        out_sup_literature = load_metrics_sup_scratch(args, agents[indx])

        out_sup = load_metrics_sup(
            args,
            hparams,
            spk_per_bkt_collection[1],
            train_dvec_modes["train_dvec_proposed"],
            agents[indx],
        )

        # Create time axis for the training plots
        axis_unsup_literature = create_time_axis(
            out_unsup_literature["elapsed_time"],
            len(out_unsup_literature["elapsed_time"]),
        )
        axis_sup_literature = create_time_axis(
            out_sup_literature["elapsed_time"],
            len(out_sup_literature["elapsed_time"]),
        )
        axis_sup = create_time_axis(
            out_sup["elapsed_time"],
            len(out_sup["elapsed_time"]),
        )
        axis_unsup = create_time_axis(
            out_unsup["elapsed_time"],
            len(out_unsup["elapsed_time"]),
        )

        elapsed_time_proposed_sup = torch.tensor(axis_sup).view(-1) / 60
        elapsed_time_proposed_unsup = torch.tensor(axis_unsup).view(-1) / 60
        elapsed_time_unsup_literature = (
            torch.tensor(axis_unsup_literature).view(-1) / 60
        )
        elapsed_time_sup_literature = torch.tensor(axis_sup_literature).view(-1) / 60

        elapsed_time_sup_agnts[indx] = elapsed_time_proposed_sup
        elapsed_time_unsup_agnts[indx] = elapsed_time_proposed_unsup
        elapsed_time_unsup_literature_agnts[indx] = elapsed_time_unsup_literature
        elapsed_time_sup_literature_agnts[indx] = elapsed_time_sup_literature

        # Accuracies
        new_axs[indx].plot(
            elapsed_time_unsup_literature_agnts[indx],
            out_unsup_literature["val_loss"],
            "-",
            "--",
            color="black",
            label=f"Literature",
        )
        new_axs[indx].plot(
            elapsed_time_sup_literature_agnts[indx],
            out_sup_literature["val_loss"],
            "-",
            color="brown",
            label=f"Literature sup",
        )
        new_axs[indx].plot(
            elapsed_time_unsup_agnts[indx],
            out_unsup["val_loss"]["7"],
            "-",
            "-",
            marker="o",
            color="blue",
            label=f"Proposed unsup",
            ms=4,
        )
        new_axs[indx].plot(
            elapsed_time_sup_agnts[indx],
            out_sup["val_loss"]["7"],
            "-",
            marker="x",
            color="green",
            label=f"Proposed sup",
            ms=4,
        )

        new_axs[indx].set_ylabel(r"Loss")
        new_axs[indx].set_xlabel(r"Total elapsed time (min)")

        new_axs[indx].legend(prop={"size": 11})

    fig.tight_layout()

    # fig.savefig(
    #     f"{result_dir_plot_path}/acc_eval_agents_sup_unsup_updated.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     dpi=1200,
    # )

    plt.show()
    plt.close()
