import json

from pathlib import Path


def create_dataset_arguments(args, dir_name):
    """Create dataset arguments.

    Args:
        - args: necessary arguments to create the dataset.
        - dir_name: base directory name.

    Returns:
        - data_dir (Union[str, Path]): data directory.
        - speaker_infos (dict): speaker information.

    """

    data_dir = f"{dir_name}_{args.agnt_num}/"

    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    speaker_infos = metadata["speaker_gender"]

    return data_dir, speaker_infos
