import json

from pathlib import Path
from utils import create_filenames_unsupervised_results


def load_elapsed_time_unsup(args, hparams):
    """To load JSON files from the results directory."""

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unsupervised_results(args, hparams)

    # Loading the JSON files
    with open(
        Path(paths_filenames["dir_td"], paths_filenames["filename_time_delay"]),
        "r",
    ) as train_elapsed_time:
        train_elapsed_time = json.load(train_elapsed_time)

    return train_elapsed_time
