import json

from pathlib import Path
from utils import create_filenames_unsupervised_results_vox, GE2ELoss


def load_metrics_unsup_vox(
    args,
    hparams,
    spk_per_bucket,
    train_dvec_mode,
    agnt_num,
):
    """To load JSON files from the results directory."""

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unsupervised_results_vox(
        args,
        hparams.ma_mode,
        args.max_mem_unsup,
        spk_per_bucket,
        train_dvec_mode,
        agnt_num,
        GE2ELoss,
    )

    # Loading the JSON files
    with open(
        Path(
            paths_filenames["dir_td"],
            paths_filenames["filename_time_delay"],
        ),
        "r",
    ) as elapsed_time:
        elapsed_time = json.load(elapsed_time)

    out = {"elapsed_time": elapsed_time}

    return out
