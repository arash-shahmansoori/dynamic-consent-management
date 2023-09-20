import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_cls,
    create_cls_checkpoint_dir_reg,
)
from reg_new_spks import dyn_reg_sup


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Choose the the data directory according to percentage of old utterances
    data_dir_old = args.data_dir_fifty_pcnt

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Filenames for the checkpoints of classifier
    filenames_and_dirs = create_filenames_cls(args, hparams)

    ckpt_cls, status_cls = create_cls_checkpoint_dir_reg(
        args,
        filenames_and_dirs["filename"],
        filenames_and_dirs["filename_reg"],
        filenames_and_dirs["filename_dir"],
        filenames_and_dirs["filename_dir_reg"],
    )

    dyn_reg_sup(
        args,
        hparams,
        data_dir_old,
        buckets,
        device,
        status_cls=status_cls,
        ckpt_cls=ckpt_cls,
    )
