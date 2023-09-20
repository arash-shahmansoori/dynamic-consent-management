import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_cls,
    create_cls_checkpoint_dir_unreg,
)
from unreg_re_reg_spks import unreg_sup


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # List of speaker(s) to be unregistered from the bucket(s)
    unreg_spks = [20, 21, 22]

    # Filenames for the checkpoints of classifier
    filenames_and_dirs = create_filenames_cls(args, hparams, unreg_spks)

    ckpt_cls, status_cls = create_cls_checkpoint_dir_unreg(
        args,
        filenames_and_dirs["filename"],
        filenames_and_dirs["filename_unreg"],
        filenames_and_dirs["filename_dir"],
        filenames_and_dirs["filename_dir_unreg"],
    )

    unreg_sup(
        args,
        hparams,
        buckets,
        device,
        unreg_spks,
        status_cls,
        ckpt_cls=ckpt_cls,
    )
