import torch
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from matplotlib import style

from .load_dynamic_reg_supervised import load_dyn_reg_sup
from .load_dynamic_reg_unsupervised import load_dyn_reg_unsup


def calibrate_elapsed_time_reduced(elapsed_time_round):

    elapsed_times = []

    for indx in range(1, len(elapsed_time_round)):
        elapsed_times.append(
            torch.tensor(elapsed_time_round[:indx]).view(-1).sum() / 60
        )

    return elapsed_times


def compute_indx(elapsed_time, per_round_total_elapsed_time):
    indx_clollection = []
    for indx in range(1, len(torch.tensor(elapsed_time).view(-1))):
        if (torch.tensor(elapsed_time).view(-1)[:indx].sum() / 60) <= torch.stack(
            per_round_total_elapsed_time
        ).min():
            indx_clollection.append(indx)

    return indx_clollection[-1]


def plot_result_dyn_reg_val_acc_vs_old_utts_pcnt_sup_unsup(
    rounds_indx,
    pcnts_old,
    args,
    hparams,
):

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    # rgba_colors_a = np.zeros((1, 3))
    # rgba_colors_a[:, 0] = 0.7

    # rgba_colors_b = np.zeros((1, 3))
    # rgba_colors_b[:, 2] = 0.7

    # rgba_colors_c = np.zeros((1, 3))
    # rgba_colors_c[:, 1] = 0.7

    # rgba_colors_d = np.zeros((1, 3))
    # rgba_colors_d[:, 0] = 0
    # rgba_colors_d[:, 1] = 0
    # rgba_colors_d[:, 2] = 0

    markers = ["2", "o", "s", "^"]

    # colors = [rgba_colors_a, rgba_colors_b, rgba_colors_c, rgba_colors_d]
    colors = ["red", "blue", "green", "black"]

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    style.use("ggplot")
    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    keys_ten = [("ten", i) for i in range(4)]
    keys_thirty = [("thirty", i) for i in range(4)]
    keys_fifty = [("fifty", i) for i in range(4)]
    keys_seventy = [("seventy", i) for i in range(4)]
    keys_ninty = [("ninty", i) for i in range(4)]

    keys = keys_ten + keys_thirty + keys_fifty + keys_seventy + keys_ninty

    elapsed_times_storage_sup = {key: [] for key in keys}
    elapsed_times_storage_unsup = {key: [] for key in keys}

    total_elapsed_time_sup = {key: [] for key in keys}
    total_elapsed_time_unsup = {key: [] for key in keys}

    per_round_total_elapsed_time_sup = {r: [] for r in range(4)}
    per_round_total_elapsed_time_unsup = {r: [] for r in range(4)}

    per_round_indx_sup = {r: [] for r in range(4)}
    per_round_indx_unsup = {r: [] for r in range(4)}

    for pcnt, pcnt_value in pcnts_old.items():
        for _, i in enumerate(rounds_indx):
            _, elapsed_time_round_sup = load_dyn_reg_sup(
                args,
                hparams,
                i,
                pcnt,
                args.agnt_num,
            )
            _, elapsed_time_round_unsup = load_dyn_reg_unsup(
                args,
                hparams,
                i,
                pcnt,
                args.agnt_num,
            )

            elapsed_times_sup = calibrate_elapsed_time_reduced(elapsed_time_round_sup)
            elapsed_times_unsup = calibrate_elapsed_time_reduced(
                elapsed_time_round_unsup
            )

            elapsed_times_storage_sup[(pcnt, i)].append(elapsed_times_sup)
            elapsed_times_storage_unsup[(pcnt, i)].append(elapsed_times_unsup)
            if pcnt != "ten":
                total_elapsed_time_sup[(pcnt, i)] = (
                    torch.tensor(elapsed_times_storage_sup[(pcnt, i)]).view(-1).sum()
                    / 60
                )
                total_elapsed_time_unsup[(pcnt, i)] = (
                    torch.tensor(elapsed_times_storage_unsup[(pcnt, i)]).view(-1).sum()
                    / 60
                )

                per_round_total_elapsed_time_sup[i].append(
                    total_elapsed_time_sup[(pcnt, i)]
                )
                per_round_total_elapsed_time_unsup[i].append(
                    total_elapsed_time_unsup[(pcnt, i)]
                )

                indx_round_sup = compute_indx(
                    elapsed_times_storage_sup[("ten", i)],
                    per_round_total_elapsed_time_sup[i],
                )
                indx_round_unsup = compute_indx(
                    elapsed_times_storage_unsup[("ten", i)],
                    per_round_total_elapsed_time_unsup[i],
                )

                per_round_indx_sup[i].append(indx_round_sup)
                per_round_indx_unsup[i].append(indx_round_unsup)

    for pcnt, pcnt_value in pcnts_old.items():
        for _, i in enumerate(rounds_indx):
            val_acc_round, _ = load_dyn_reg_sup(
                args,
                hparams,
                i,
                pcnt,
                args.agnt_num,
            )
            val_acc_round_unsup, _ = load_dyn_reg_unsup(
                args,
                hparams,
                i,
                pcnt,
                args.agnt_num,
            )

            last_key = str(hparams.num_of_buckets - 1)

            if pcnt == "ten":

                ax1.scatter(
                    pcnt_value * torch.ones(1, 1),
                    torch.tensor(
                        val_acc_round[last_key][per_round_indx_sup[i][-1]]
                    ),  # To limit the elapsed time
                    marker=markers[i],
                    color=colors[i],
                    label=f"Round {i}",
                )
                ax2.scatter(
                    pcnt_value * torch.ones(1, 1),
                    torch.tensor(
                        val_acc_round_unsup[last_key][per_round_indx_unsup[0][-1]]
                    ),  # To limit the elapsed time
                    marker=markers[i],
                    color=colors[i],
                    label=f"Round {i}",
                )
            else:
                ax1.scatter(
                    pcnt_value * torch.ones(1, 1),
                    torch.tensor(val_acc_round[last_key][-1]),
                    marker=markers[i],
                    color=colors[i],
                )
                ax2.scatter(
                    pcnt_value * torch.ones(1, 1),
                    torch.tensor(val_acc_round_unsup[last_key][-1]),
                    marker=markers[i],
                    color=colors[i],
                )

    ax1.legend(prop={"size": 11})
    ax2.legend(prop={"size": 11})

    ax1.set_xlim([10 - 5, 100])
    ax2.set_xlim([10 - 5, 100])

    ax1.set_ylim([85, 100])
    ax2.set_ylim([85, 100])

    ax1.set_xticks([i * 10 for i in range(1, 11, 2)])
    ax2.set_xticks([i * 10 for i in range(1, 11, 2)])

    ax1.set_ylabel(r"Accuracy")
    ax1.set_xlabel(r"Percentage of old utterances (\%)")

    ax2.set_ylabel(r"Accuracy")
    ax2.set_xlabel(r"Percentage of old utterances (\%)")

    plt.tight_layout()

    fig.savefig(
        f"{result_dir_plot_path}/acc_eval_vs_utts_pcnt_dyn_reg_sup_unsup.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1200,
    )

    plt.show()
    plt.close()
