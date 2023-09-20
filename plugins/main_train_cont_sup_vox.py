import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_cls_vox,
    create_cls_checkpoint_dir,
)
from training import train_contrastive_supervised_vox


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]
    # buckets = [7]  # 0,1,2,3,4,5,6, => agnt 1

    filenames_and_dirs = create_filenames_cls_vox(args, hparams)

    # Set ``ckpt_cls'' to the available checkpoint
    ckpt_cls, _ = create_cls_checkpoint_dir(
        args,
        filenames_and_dirs["filename"],
        filenames_and_dirs["filename_dir"],
    )

    train_contrastive_supervised_vox(
        args,
        hparams,
        buckets,
        device,
        ckpt_cls=ckpt_cls,
    )
