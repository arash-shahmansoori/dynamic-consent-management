import json

from pathlib import Path
from utils import create_filenames_unsupervised_results


def load_metrics_unsup(args, hparams, spk_per_bucket, train_dvec_mode, agnt_num):
    """To load JSON files from the results directory."""

    # Create path for saving the plots
    output_dir_results = args.output_dir_results
    output_dir_results_path = Path(output_dir_results)
    output_dir_results_path.mkdir(parents=True, exist_ok=True)

    # Create paths and filenames for saving the validation metrics
    paths_filenames = create_filenames_unsupervised_results(
        args,
        hparams.ma_mode,
        args.max_mem_unsup,
        spk_per_bucket,
        train_dvec_mode,
        agnt_num,
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

    with open(
        Path(
            paths_filenames["dir_acc_cont_val"],
            paths_filenames["filename_acc_cont_val"],
        ),
        "r",
    ) as val_acc:
        val_acc = json.load(val_acc)

    with open(
        Path(
            paths_filenames["dir_loss_cont_val"],
            paths_filenames["filename_loss_cont_val"],
        ),
        "r",
    ) as val_loss:
        val_loss = json.load(val_loss)

    out = {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "elapsed_time": elapsed_time,
        "output_dir_results_path": output_dir_results_path,
    }

    return out
