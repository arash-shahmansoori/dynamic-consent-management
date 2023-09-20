import json

from pathlib import Path
from utils import create_filenames_re_reg_results


def load_re_reg_sup(args, hparams, unreg_spks):

    args.max_mem = 120

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_re_reg_results(
        args,
        hparams,
        args.spk_per_bucket,
        unreg_spks,
    )

    # Loading JSON files
    with open(
        Path(
            paths_filenames["dir_td"],
            paths_filenames["filename_time_delay"],
        ),
        "r",
    ) as elapsed_time:
        elapsed_time = json.load(elapsed_time)

    with open(
        Path(
            paths_filenames["dir_acc_val"],
            paths_filenames["filename_acc_val"],
        ),
        "r",
    ) as val_acc:
        val_acc = json.load(val_acc)

    return val_acc, elapsed_time
