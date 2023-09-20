import json

from pathlib import Path
from utils import create_filenames_reg_supervised_scratch_results


def load_dyn_reg_sup_scratch(args, hparams, round_num, agnt_num):
    """To load the metrics from the JSON files in the results directory (dynamic registrations)."""

    # Create path file names for saving the metrics per round
    paths_filenames = create_filenames_reg_supervised_scratch_results(
        args,
        hparams.ma_mode,
        args.max_mem,
        args.epochs_per_dvector,
        args.epochs_per_cls,
        round_num,
        agnt_num,
    )

    with open(
        Path(
            paths_filenames["dir_td"],
            paths_filenames["filename_time_delay"],
        ),
        "r",
    ) as elapsed_time:
        elapsed_time_round = json.load(elapsed_time)

    with open(
        Path(
            paths_filenames["dir_acc_val"],
            paths_filenames["filename_acc_val"],
        ),
        "r",
    ) as val_acc:
        val_acc_round = json.load(val_acc)

    return val_acc_round, elapsed_time_round
