import torch

from .load_metrics_supervised_vox import load_metrics_sup_vox
from .load_metrics_unsupervised_vox import load_metrics_unsup_vox

from .load_metrics_supervised_scratch_vox import load_metrics_sup_scratch_vox

from .load_metrics_unsupervised_scratch_vox import load_metrics_unsup_scratch_vox


def create_time_axis(x, p):
    time_axis = []
    ts = 0
    for i in range(p):
        ts += x[i]
        time_axis.append(ts)

    return time_axis


def compute_elapsed_time(spk_per_bkt_collection, agents, args, hparams):
    train_dvec_modes = {
        "train_dvec_literature": "train_dvec",
        "train_dvec_adapted": "train_dvec_adapted",
        "train_dvec_proposed": "train_dvec_proposed",
    }

    # elapsed_time_sup_agnts = {}
    # elapsed_time_unsup_agnts = {}
    # elapsed_time_unsup_literature_agnts = {}
    # elapsed_time_sup_literature_agnts = {}

    for _, agnt in enumerate(agents):
        out_unsup_literature = load_metrics_unsup_scratch_vox(
            args,
            hparams,
            spk_per_bkt_collection[0],
            train_dvec_modes["train_dvec_literature"],
            agnt,
        )
        out_unsup = load_metrics_unsup_vox(
            args,
            hparams,
            spk_per_bkt_collection[1],
            train_dvec_modes["train_dvec_proposed"],
            agnt,
        )

        out_sup_literature = load_metrics_sup_scratch_vox(
            args,
            hparams,
            spk_per_bkt_collection[0],
            train_dvec_modes["train_dvec_literature"],
            agnt,
        )
        out_sup = load_metrics_sup_vox(
            args,
            hparams,
            spk_per_bkt_collection[1],
            train_dvec_modes["train_dvec_proposed"],
            agnt,
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

        elapsed_time_sup = elapsed_time_proposed_sup[-1]
        elapsed_time_sup_literature = elapsed_time_sup_literature[-1]

        elapsed_time_unsup = elapsed_time_proposed_unsup[-1]
        elapsed_time_unsup_literature = elapsed_time_unsup_literature[-1]

        reduction_pcnt_sup = 100 * (
            (elapsed_time_sup - elapsed_time_sup_literature)
            / elapsed_time_sup_literature
        )
        reduction_pcnt_unsup = 100 * (
            (elapsed_time_unsup - elapsed_time_unsup_literature)
            / elapsed_time_unsup_literature
        )

        print(
            f"Agent:{agnt} | reduction_pcnt_sup: {reduction_pcnt_sup:.2f} | reduction_pcnt_unsup: {reduction_pcnt_unsup:.2f} |"
        )

        # print(
        #     f"Agent:{agnt} | et-sup: {elapsed_time_sup:.2f} | et-sup-lit: {elapsed_time_sup_literature:.2f} |"
        #     f"et-unsup: {elapsed_time_unsup:.2f} | et-unsup-lit: {elapsed_time_unsup_literature:.2f} |"
        # )

        # print(
        #     f"Agent:{agnt} | et-sup: {elapsed_time_sup:.2f} | et-sup-lit: {elapsed_time_sup_literature:.2f} |"
        #     f"et-unsup: {elapsed_time_unsup:.2f} |"
        # )

        # print(
        #     f"Agent:{agnt} | et-sup: {elapsed_time_sup:.2f}|"
        #     f"et-unsup: {elapsed_time_unsup:.2f} |"
        # )
