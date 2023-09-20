import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_cls,
    create_cls_checkpoint_dir_unreg,
    create_filenames_dvec_unsupervised_latent,
    create_dvec_latent_checkpoint_dir_unsup_unreg,
)
from unreg_re_reg_spks import unreg_unsup_v2


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # List of speaker(s) to be unregistered from the bucket(s)
    unreg_spks = [20, 21, 22, 23, 24]

    # Filenames for the checkpoints of classifier
    filenames_and_dirs_unsup = create_filenames_dvec_unsupervised_latent(
        args,
        hparams,
        unreg_spks,
    )

    ckpt_cls_unsup, status_cls_unsup = create_dvec_latent_checkpoint_dir_unsup_unreg(
        args,
        filenames_and_dirs_unsup["filename"],
        filenames_and_dirs_unsup["filename_unreg"],
        filenames_and_dirs_unsup["filename_dir"],
        filenames_and_dirs_unsup["filename_dir_unreg"],
    )

    status_cls = status_cls_unsup
    ckpt_cls = ckpt_cls_unsup

    unreg_unsup_v2(
        args,
        hparams,
        buckets,
        device,
        unreg_spks,
        status_cls,
        ckpt_dvec_latent=ckpt_cls,
    )
