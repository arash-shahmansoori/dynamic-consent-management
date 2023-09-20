import json

from pathlib import Path
from utils import create_filenames_results


def load_metrics_sup(args, hparams, spk_per_bkt, train_dvec_mode, agnt_num):
    """To load the metrics from the JSON files in the results directory (supervised training)."""

    # Create path for saving the plots
    output_dir_results = args.output_dir_results
    output_dir_results_path = Path(output_dir_results)
    output_dir_results_path.mkdir(parents=True, exist_ok=True)

    # Paths and filenames for the training/validation metrics
    paths_filenames = create_filenames_results(
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

    with open(
        Path(paths_filenames["dir_acc_train"], paths_filenames["filename_acc_train"]),
        "r",
    ) as train_acc:
        train_acc = json.load(train_acc)

    with open(
        Path(paths_filenames["dir_acc_val"], paths_filenames["filename_acc_val"]),
        "r",
    ) as val_acc:
        val_acc = json.load(val_acc)

    with open(
        Path(paths_filenames["dir_loss_train"], paths_filenames["filename_loss_train"]),
        "r",
    ) as train_loss:
        train_loss = json.load(train_loss)

    with open(
        Path(paths_filenames["dir_loss_val"], paths_filenames["filename_loss_val"]),
        "r",
    ) as val_loss:
        val_loss = json.load(val_loss)

    out = {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "elapsed_time": elapsed_time,
        "output_dir_results_path": output_dir_results_path,
    }

    return out
