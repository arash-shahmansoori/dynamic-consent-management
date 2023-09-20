import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_scratch_vox,
    create_cls_scratch_checkpoint_dir,
)
from training import train_from_scratch_vox


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Dictionaries of filenames for the checkpoints of classifier
    filename, filename_dir = create_filenames_scratch_vox(args)

    # Set ``ckpt_cls'' to the available checkpoint
    ckpt_cls, _ = create_cls_scratch_checkpoint_dir(args, filename, filename_dir)

    train_from_scratch_vox(args, hparams, buckets, device, ckpt_cls)
