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


def plot_result_dyn_reg_val_acc_old_utts_pcnt_vs_agnt_sup_unsup(
    agnt_round_indx,
    pcnts_old,
    args,
    hparams,
):
    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    markers = ["2", "o", "s", "^", "x"]
    colors = {
        "ten": "red",
        "thirty": "blue",
        "fifty": "green",
        "seventy": "brown",
        "ninty": "purple",
    }

    # Settings
    x = 4  # Want figures to be A4
    plt.rc("figure", figsize=[46.82 * 0.5 ** (0.5 * x), 33.11 * 0.5 ** (0.5 * x)])
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    style.use("ggplot")
    fig = plt.figure(figsize=(6, 7))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    keys_calibrated = []
    keys_ten, keys_thirty, keys_fifty, keys_seventy, keys_ninty = [], [], [], [], []
    for agnt_indx, round_nums in agnt_round_indx.items():
        for i in round_nums:
            keys_ten.append(("ten", agnt_indx, i))
            keys_thirty.append(("thirty", agnt_indx, i))
            keys_fifty.append(("fifty", agnt_indx, i))
            keys_seventy.append(("seventy", agnt_indx, i))
            keys_ninty.append(("ninty", agnt_indx, i))

            keys_calibrated.append((agnt_indx, i))

    keys = keys_ten + keys_thirty + keys_fifty + keys_seventy + keys_ninty

    elapsed_times_storage_sup = {key: [] for key in keys}
    elapsed_times_storage_unsup = {key: [] for key in keys}

    total_elapsed_time_sup = {key: [] for key in keys}
    total_elapsed_time_unsup = {key: [] for key in keys}

    per_round_total_elapsed_time_sup_agnt = {r: [] for r in keys_calibrated}
    per_round_total_elapsed_time_unsup_agnt = {r: [] for r in keys_calibrated}

    per_round_indx_sup_agnt = {r: [] for r in keys_calibrated}
    per_round_indx_unsup_agnt = {r: [] for r in keys_calibrated}

    for pcnt, pcnt_value in pcnts_old.items():
        for agnt_indx, round_nums in agnt_round_indx.items():
            for _, i in enumerate(round_nums):
                _, elapsed_time_round_sup = load_dyn_reg_sup(
                    args,
                    hparams,
                    i,
                    pcnt,
                    agnt_indx,
                )
                _, elapsed_time_round_unsup = load_dyn_reg_unsup(
                    args,
                    hparams,
                    i,
                    pcnt,
                    agnt_indx,
                )

                elapsed_times_sup = calibrate_elapsed_time_reduced(
                    elapsed_time_round_sup
                )
                elapsed_times_unsup = calibrate_elapsed_time_reduced(
                    elapsed_time_round_unsup
                )

                elapsed_times_storage_sup[(pcnt, agnt_indx, i)].append(
                    elapsed_times_sup
                )
                elapsed_times_storage_unsup[(pcnt, agnt_indx, i)].append(
                    elapsed_times_unsup
                )
                if pcnt != "ten":
                    total_elapsed_time_sup[(pcnt, agnt_indx, i)] = (
                        torch.tensor(elapsed_times_storage_sup[(pcnt, agnt_indx, i)])
                        .view(-1)
                        .sum()
                        / 60
                    )
                    total_elapsed_time_unsup[(pcnt, agnt_indx, i)] = (
                        torch.tensor(elapsed_times_storage_unsup[(pcnt, agnt_indx, i)])
                        .view(-1)
                        .sum()
                        / 60
                    )

                    per_round_total_elapsed_time_sup_agnt[(agnt_indx, i)].append(
                        total_elapsed_time_sup[(pcnt, agnt_indx, i)]
                    )
                    per_round_total_elapsed_time_unsup_agnt[(agnt_indx, i)].append(
                        total_elapsed_time_unsup[(pcnt, agnt_indx, i)]
                    )

                    indx_round_sup = compute_indx(
                        elapsed_times_storage_sup[("ten", agnt_indx, i)],
                        per_round_total_elapsed_time_sup_agnt[(agnt_indx, i)],
                    )
                    indx_round_unsup = compute_indx(
                        elapsed_times_storage_unsup[("ten", agnt_indx, i)],
                        per_round_total_elapsed_time_unsup_agnt[(agnt_indx, i)],
                    )

                    per_round_indx_sup_agnt[(agnt_indx, i)].append(indx_round_sup)
                    per_round_indx_unsup_agnt[(agnt_indx, i)].append(indx_round_unsup)

    for pcnt, pcnt_value in pcnts_old.items():
        for agnt_indx, round_nums in agnt_round_indx.items():
            for _, i in enumerate(round_nums):
                val_acc_round, _ = load_dyn_reg_sup(
                    args,
                    hparams,
                    i,
                    pcnt,
                    agnt_indx,
                )
                val_acc_round_unsup, _ = load_dyn_reg_unsup(
                    args,
                    hparams,
                    i,
                    pcnt,
                    agnt_indx,
                )

                last_key = str(hparams.num_of_buckets - 1)

                if pcnt == "ten":
                    ax1.scatter(
                        (0 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(
                            val_acc_round[last_key][
                                per_round_indx_sup_agnt[(agnt_indx, i)][-1]
                            ]
                        ),  # To limit the elapsed time
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                        label=f"Round {i}",
                    )
                    ax2.scatter(
                        (0 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(
                            val_acc_round_unsup[last_key][
                                per_round_indx_unsup_agnt[(agnt_indx, i)][-1]
                            ]
                        ),  # To limit the elapsed time
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                        label=f"Round {i}",
                    )
                elif pcnt == "thirty":
                    chosen_indx = min(
                        per_round_indx_sup_agnt[(agnt_indx, i)][-1],
                        len(val_acc_round[last_key]) - 1,
                    )
                    chosen_indx_unsup = min(
                        per_round_indx_unsup_agnt[(agnt_indx, i)][-1],
                        len(val_acc_round_unsup[last_key]) - 1,
                    )

                    ax1.scatter(
                        (0.1 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round[last_key][chosen_indx]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                    ax2.scatter(
                        (0.1 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round_unsup[last_key][chosen_indx_unsup]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                elif pcnt == "fifty":
                    ax1.scatter(
                        (0.2 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round[last_key][-1]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                    ax2.scatter(
                        (0.2 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round_unsup[last_key][-1]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                elif pcnt == "seventy":
                    ax1.scatter(
                        (0.3 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round[last_key][-1]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                    ax2.scatter(
                        (0.3 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round_unsup[last_key][-1]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                elif pcnt == "ninty":
                    ax1.scatter(
                        (0.4 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round[last_key][-1]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )
                    ax2.scatter(
                        (0.4 + int(agnt_indx)) * torch.ones(1, 1),
                        torch.tensor(val_acc_round_unsup[last_key][-1]),
                        marker=markers[i],
                        color=colors[pcnt],
                        alpha=0.5,
                    )

    # ax1.legend(prop={"size": 11})
    # ax2.legend(prop={"size": 11})

    ax1.set_xlim([-0.5, 5])
    ax2.set_xlim([-0.5, 5])

    y_axis_values_1 = [i for i in range(70, 100, 5)] + [97, 100]
    y_axis_values_2 = [i for i in range(80, 100, 5)] + [97, 100]

    ax1.set_yticks(y_axis_values_1)
    ax2.set_yticks(y_axis_values_2)

    ax1.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticks([0, 1, 2, 3, 4])

    ax1.set_ylabel(r"Accuracy")
    ax1.set_xlabel(r"Agents")

    ax2.set_ylabel(r"Accuracy")
    ax2.set_xlabel(r"Agents")

    plt.tight_layout()

    fig.savefig(
        f"{result_dir_plot_path}/acc_eval_old_utts_pcnt_dyn_reg_sup_unsup_vs_agnt.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1200,
    )

    plt.show()
    plt.close()
