import torch
import matplotlib.pyplot as plt


from pathlib import Path
from matplotlib import style
from matplotlib.legend import Legend

from .load_unreg_supervised import load_unreg_sup
from .load_re_reg_supervised import load_re_reg_sup

from .load_unreg_unsupervised import load_unreg_unsup
from .load_re_reg_unsupervised import load_re_reg_unsup


def plot_result_unreg_re_reg_sup_unsup(unreg_spks_groups, args, hparams):

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    val_accs, elapsed_times = {}, {}
    val_accs_re_reg, elapsed_times_re_reg = {}, {}

    val_accs_unsup, elapsed_times_unsup = {}, {}
    val_accs_re_reg_unsup, elapsed_times_re_reg_unsup = {}, {}

    markers_storage = {
        "4_1": ["x", "x", "x", "x", "x", "x", "x", "x"],
        "4_2": ["*", "*", "*", "*", "*", "*", "*", "*"],
        "4_3": ["v", "v", "v", "v", "v", "v", "v", "v"],
        "4_5": ["d", "d", "d", "d", "d", "d", "d", "d"],
    }

    markers_others_storage = {
        "4_1": ["+", "+", "+", "+", "+", "+", "+", "+"],
        "4_2": ["s", "s", "s", "s", "s", "s", "s", "s"],
        "4_3": ["o", "o", "o", "o", "o", "o", "o", "o"],
        "4_5": ["^", "^", "^", "^", "^", "^", "^", "^"],
    }

    colors = ["red", "blue"]

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    style.use("ggplot")
    fig = plt.figure(figsize=(6, 7))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    lines1, lines2 = [], []
    lines3, lines4 = [], []
    lines5, lines6 = [], []
    lines7, lines8 = [], []

    for unreg_spk_indx, unreg_spk in unreg_spks_groups.items():
        val_acc, elapsed_time = load_unreg_sup(args, hparams, unreg_spk)
        val_acc_re_reg, elapsed_time_re_reg = load_re_reg_sup(args, hparams, unreg_spk)

        val_acc_unsup, elapsed_time_unsup = load_unreg_unsup(args, hparams, unreg_spk)
        val_acc_re_reg_unsup, elapsed_time_re_reg_unsup = load_re_reg_unsup(
            args,
            hparams,
            unreg_spk,
        )

        val_accs[unreg_spk_indx] = val_acc
        elapsed_times[unreg_spk_indx] = elapsed_time

        val_accs_re_reg[unreg_spk_indx] = val_acc_re_reg
        elapsed_times_re_reg[unreg_spk_indx] = elapsed_time_re_reg

        val_accs_unsup[unreg_spk_indx] = val_acc_unsup
        elapsed_times_unsup[unreg_spk_indx] = elapsed_time_unsup

        val_accs_re_reg_unsup[unreg_spk_indx] = val_acc_re_reg_unsup
        elapsed_times_re_reg_unsup[unreg_spk_indx] = elapsed_time_re_reg_unsup

        for i in range(hparams.num_of_buckets):
            bkt = str(i)
            unreg_spk_bkt_indx, unreg_spk_bkt_cnt = unreg_spk_indx.split("_")

            if str(i) == unreg_spk_bkt_indx:
                lines1 += ax1.plot(
                    torch.tensor(elapsed_times[unreg_spk_indx]).view(-1).sum() / 60,
                    torch.tensor(val_accs[unreg_spk_indx][str(i)])[-1],
                    markers_storage[unreg_spk_indx][int(unreg_spk_bkt_cnt)],
                    color=colors[0],
                )
                lines2 += ax2.plot(
                    torch.tensor(elapsed_times_re_reg[unreg_spk_indx]).view(-1).sum()
                    / 60,
                    torch.tensor(val_accs_re_reg[unreg_spk_indx][str(i)])[-1],
                    markers_storage[unreg_spk_indx][int(unreg_spk_bkt_cnt)],
                    color=colors[0],
                )

                lines3 += ax3.plot(
                    torch.tensor(elapsed_times_unsup[unreg_spk_indx]).view(-1).sum()
                    / 60,
                    torch.tensor(val_accs_unsup[unreg_spk_indx][str(i)])[-1],
                    markers_storage[unreg_spk_indx][int(unreg_spk_bkt_cnt)],
                    color=colors[0],
                )
                lines4 += ax4.plot(
                    torch.tensor(elapsed_times_re_reg_unsup[unreg_spk_indx])
                    .view(-1)
                    .sum()
                    / 60,
                    torch.tensor(val_accs_re_reg_unsup[unreg_spk_indx][str(i)])[-1],
                    markers_storage[unreg_spk_indx][int(unreg_spk_bkt_cnt)],
                    color=colors[0],
                )

            else:
                if i == hparams.num_of_buckets - 1:
                    lines5 += ax1.plot(
                        [
                            (
                                torch.tensor(elapsed_times[unreg_spk_indx])
                                .view(-1)
                                .sum()
                                / 60
                            ).item()
                        ],
                        torch.tensor(val_accs[unreg_spk_indx][str(i)])[-1],
                        markers_others_storage[unreg_spk_indx][i],
                        color=colors[1],
                    )
                    lines6 += ax2.plot(
                        torch.tensor(elapsed_times_re_reg[unreg_spk_indx])
                        .view(-1)
                        .sum()
                        / 60,
                        torch.tensor(val_accs_re_reg[unreg_spk_indx][str(i)])[-1],
                        markers_others_storage[unreg_spk_indx][i],
                        color=colors[1],
                    )

                    lines7 += ax3.plot(
                        [
                            (
                                torch.tensor(elapsed_times_unsup[unreg_spk_indx])
                                .view(-1)
                                .sum()
                                / 60
                            ).item()
                        ],
                        torch.tensor(val_accs_unsup[unreg_spk_indx][str(i)])[-1],
                        markers_others_storage[unreg_spk_indx][i],
                        color=colors[1],
                    )
                    lines8 += ax4.plot(
                        torch.tensor(elapsed_times_re_reg_unsup[unreg_spk_indx])
                        .view(-1)
                        .sum()
                        / 60,
                        torch.tensor(val_accs_re_reg_unsup[unreg_spk_indx][str(i)])[-1],
                        markers_others_storage[unreg_spk_indx][i],
                        color=colors[1],
                    )

    ax1.legend(
        lines1,
        [
            r"[20] $\leftarrow$ 4",
            r"[20, 21] $\leftarrow$ 4",
            r"[20, 21, 22] $\leftarrow$ 4",
            # r"all $\leftarrow$ 4",
        ],
        loc="lower left",
        fontsize=11,
    )

    # leg1 = Legend(
    #     ax1,
    #     lines5,
    #     [
    #         "Other buckets",
    #         "Other buckets",
    #         "Other buckets",
    #     ],
    #     loc="center",
    #     fontsize=11,
    # )
    # ax1.add_artist(leg1)

    ax2.legend(
        lines2,
        [
            r"[20] $\rightarrow$ 4",
            r"[20, 21] $\rightarrow$ 4",
            r"[20, 21, 22] $\rightarrow$ 4",
            # r"all $\rightarrow$ 4",
        ],
        loc="lower left",
        fontsize=11,
    )

    # leg2 = Legend(
    #     ax2,
    #     lines6,
    #     [
    #         "Other buckets",
    #         "Other buckets",
    #         "Other buckets",
    #     ],
    #     loc="center",
    #     fontsize=11,
    # )
    # ax2.add_artist(leg2)

    ax3.legend(
        lines3,
        [
            r"[20] $\leftarrow$ 4",
            r"[20, 21] $\leftarrow$ 4",
            r"[20, 21, 22] $\leftarrow$ 4",
            # r"all $\leftarrow$ 4",
        ],
        loc="lower left",
        fontsize=11,
    )

    # leg3 = Legend(
    #     ax3,
    #     lines7,
    #     [
    #         "Other buckets",
    #         "Other buckets",
    #         "Other buckets",
    #     ],
    #     loc="center",
    #     fontsize=11,
    # )
    # ax3.add_artist(leg3)

    ax4.legend(
        lines4,
        [
            r"[20] $\rightarrow$ 4",
            r"[20, 21] $\rightarrow$ 4",
            r"[20, 21, 22] $\rightarrow$ 4",
            # r"all $\rightarrow$ 4",
        ],
        loc="lower left",
        fontsize=11,
    )

    # leg4 = Legend(
    #     ax4,
    #     lines8,
    #     [
    #         "Other buckets",
    #         "Other buckets",
    #         "Other buckets",
    #     ],
    #     loc="center",
    #     fontsize=11,
    # )
    # ax4.add_artist(leg4)

    ax1.set_ylim(30, 103)
    ax2.set_ylim(60, 103)
    ax3.set_ylim(30, 103)
    ax4.set_ylim(60, 103)

    ax1.set_ylabel(r"Accuracy")
    ax2.set_ylabel(r"Accuracy")
    ax3.set_ylabel(r"Accuracy")
    ax4.set_ylabel(r"Accuracy")

    ax1.set_title(r"Removal")
    ax2.set_title(r"Re-registration")
    ax3.set_title(r"Removal")
    ax4.set_title(r"Re-registration")

    ax1.set_xlabel(r"Total elapsed time (min)")
    ax2.set_xlabel(r"Total elapsed time (min)")
    ax3.set_xlabel(r"Total elapsed time (min)")
    ax4.set_xlabel(r"Total elapsed time (min)")

    plt.tight_layout()

    fig.savefig(
        f"{result_dir_plot_path}/acc_eval_unreg_rereg_sup.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1200,
    )

    plt.show()
    plt.close()
