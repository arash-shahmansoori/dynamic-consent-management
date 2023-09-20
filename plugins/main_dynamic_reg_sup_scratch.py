import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_cls_dynamic_scratch,
    create_cls_checkpoint_dir_dynamic_reg_scratch,
)
from reg_new_spks import dyn_reg_sup_scratch


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Filenames for the checkpoints of classifier
    filenames_and_dirs = create_filenames_cls_dynamic_scratch(
        args,
        hparams,
        hparams.round_num,
    )

    ckpt_cls, _ = create_cls_checkpoint_dir_dynamic_reg_scratch(
        args,
        filenames_and_dirs["filename_reg"],
        filenames_and_dirs["filename_dir_reg"],
    )

    dyn_reg_sup_scratch(
        args,
        hparams,
        buckets,
        device,
        ckpt_cls=ckpt_cls,
    )
