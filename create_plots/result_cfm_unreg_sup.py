import torch
import seaborn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from .cfm_preparation_per_bkt import prepare_cfm_per_bkt
from utils import (
    custom_confusion_matrix,
    normalize_custom_confusion_matrix,
    create_filenames_cls,
    create_cls_checkpoint_dir_unreg,
)


def plot_cfm_unreg_sup(unreg_spks_groups, args, hparams):

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    fig = plt.figure(figsize=(7, 5))

    for unreg_spk_indx, unreg_spk in unreg_spks_groups.items():
        # Filenames for the checkpoints of classifier
        filenames_and_dirs = create_filenames_cls(args, hparams, unreg_spk)

        ckpt_cls, _ = create_cls_checkpoint_dir_unreg(
            args,
            filenames_and_dirs["filename"],
            filenames_and_dirs["filename_unreg"],
            filenames_and_dirs["filename_dir"],
            filenames_and_dirs["filename_dir_unreg"],
        )

        val_out_per_bucket = prepare_cfm_per_bkt(
            args,
            hparams,
            buckets,
            device,
            unreg_spk,
            ckpt_cls=ckpt_cls,
        )

        bucket_indx, sub_indx = unreg_spk_indx.split("_")

        fig.add_subplot(2, 2, int(sub_indx) + 1)

        actual = np.array(
            torch.tensor(val_out_per_bucket["gtruth_indx"][int(bucket_indx)])
            .view(-1)
            .tolist()
        )
        predicted = np.array(
            torch.tensor(val_out_per_bucket["pred_indx"][int(bucket_indx)])
            .view(-1)
            .tolist()
        )

        confusion_mtx, norm_sum = custom_confusion_matrix(actual, predicted)
        confusion_mtx_normalized = normalize_custom_confusion_matrix(
            confusion_mtx,
            norm_sum,
        )

        seaborn.heatmap(
            confusion_mtx_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=np.unique(predicted),
            yticklabels=np.unique(actual),
        )

    plt.tight_layout()

    fig.savefig(
        f"{result_dir_plot_path}/cfm_eval_unreg_sup.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1200,
    )

    plt.show()
    plt.close()
