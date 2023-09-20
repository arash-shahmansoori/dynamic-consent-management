import json

from pathlib import Path
from utils import create_filenames_results_vox


def load_metrics_sup_vox(args, hparams, spk_per_bkt, train_dvec_mode, agnt_num):
    """To load the metrics from the JSON files in the results directory (supervised training)."""

    # Paths and filenames for the training/validation metrics
    paths_filenames = create_filenames_results_vox(
        args,
        hparams.ma_mode,
        args.max_mem,
        spk_per_bkt,
        train_dvec_mode,
        agnt_num,
    )

    # Loading the JSON files
    with open(
        Path(paths_filenames["dir_td"], paths_filenames["filename_time_delay"]),
        "r",
    ) as elapsed_time:
        elapsed_time = json.load(elapsed_time)

    out = {"elapsed_time": elapsed_time}

    return out
