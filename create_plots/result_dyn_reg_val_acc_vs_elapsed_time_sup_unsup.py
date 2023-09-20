import torch
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from matplotlib import style
from matplotlib.legend import Legend

from .load_dynamic_reg_supervised import load_dyn_reg_sup
from .load_dynamic_reg_supervised_scratch import load_dyn_reg_sup_scratch

from .load_dynamic_reg_unsupervised import load_dyn_reg_unsup
from .load_dynamic_reg_unsupervised_scratch import load_dyn_reg_unsup_scratch


def plot_result_dyn_reg_val_acc_vs_elapsed_time_sup_unsup(rounds_indx, args, hparams):

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    markers = ["2", "o", "s", "^"]
    markers_round_scratch = ["1", "*", "D", "v"]

    colors = ["red", "blue", "green", "black"]
    colors_scratch = ["purple", "cyan", "olive", "gray"]

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    style.use("ggplot")
    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    lines1, lines2 = [], []
    lines3, lines4 = [], []

    for _, i in enumerate(rounds_indx):
        val_acc_round, elapsed_time_round = load_dyn_reg_sup(
            args,
            hparams,
            i,
            "fifty",
            args.agnt_num,
        )
        val_acc_round_scratch, elapsed_time_round_scratch = load_dyn_reg_sup_scratch(
            args,
            hparams,
            i,
            args.agnt_num,
        )

        val_acc_round_unsup, elapsed_time_round_unsup = load_dyn_reg_unsup(
            args,
            hparams,
            i,
            "fifty",
            args.agnt_num,
        )
        (
            val_acc_round_scratch_unsup,
            elapsed_time_round_scratch_unsup,
        ) = load_dyn_reg_unsup_scratch(
            args,
            hparams,
            i,
            args.agnt_num,
        )

        last_key = str(hparams.num_of_buckets - 1)

        lines1 += ax1.plot(
            torch.tensor(elapsed_time_round).view(-1).sum() / 60,
            torch.tensor(val_acc_round[last_key][-1]).reshape(1),
            markers[i],
            color=colors[i],
        )
        lines2 += ax1.plot(
            torch.tensor(elapsed_time_round_scratch).view(-1).sum() / 60,
            torch.tensor(val_acc_round_scratch[last_key][-1]).reshape(1),
            markers_round_scratch[i],
            color=colors_scratch[i],
        )

        lines3 += ax2.plot(
            torch.tensor(elapsed_time_round_unsup).view(-1).sum() / 60,
            torch.tensor(val_acc_round_unsup[last_key][-1]),
            markers[i],
            color=colors[i],
        )
        lines4 += ax2.plot(
            torch.tensor(elapsed_time_round_scratch_unsup).view(-1).sum() / 60,
            torch.tensor(val_acc_round_scratch_unsup[last_key][-1]),
            markers_round_scratch[i],
            color=colors_scratch[i],
        )

    ax1.legend(
        lines1[:4],
        [
            "Dynamic, round 0",
            "Dynamic, round 1",
            "Dynamic, round 2",
            "Dynamic, round 3",
        ],
        loc="lower left",
        fontsize=11,
    )

    leg1 = Legend(
        ax1,
        lines2[:4],
        [
            "Re-train, round 0",
            "Re-train, round 1",
            "Re-train, round 2",
            "Re-train, round 3",
        ],
        loc="lower right",
        fontsize=11,
    )
    ax1.add_artist(leg1)

    ax2.legend(
        lines3[:4],
        [
            "Dynamic, round 0",
            "Dynamic, round 1",
            "Dynamic, round 2",
            "Dynamic, round 3",
        ],
        loc="lower left",
        fontsize=11,
    )

    leg2 = Legend(
        ax2,
        lines4[:4],
        [
            "Re-train, round 0",
            "Re-train, round 1",
            "Re-train, round 2",
            "Re-train, round 3",
        ],
        loc="lower right",
        fontsize=11,
    )
    ax2.add_artist(leg2)

    ax1.set_ylim([90, 100])
    ax2.set_ylim([90, 100])

    ax1.set_ylabel(r"Accuracy")
    ax1.set_xlabel(r"Total elapsed time per round (min)")

    ax2.set_ylabel(r"Accuracy")
    ax2.set_xlabel(r"Total elapsed time per round (min)")

    plt.tight_layout()

    fig.savefig(
        f"{result_dir_plot_path}/acc_eval_vs_elapsed_time_dyn_reg_sup_unsup.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1200,
    )

    plt.show()
    plt.close()
