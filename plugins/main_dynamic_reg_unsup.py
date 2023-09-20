import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_dvec_unsupervised,
    create_filenames_dvec_unsupervised_latent,
    create_cls_checkpoint_dir_reg_unsup,
)
from reg_new_spks import dyn_reg_unsup


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Choose the the data directory according to percentage of old utterances
    data_dir_old = args.data_dir_ten_pcnt

    # Filenames for the checkpoints of latent d-vec
    filenames_and_dirs = create_filenames_dvec_unsupervised_latent(
        args,
        hparams,
    )

    ckpt_cls, status_cls = create_cls_checkpoint_dir_reg_unsup(
        args,
        filenames_and_dirs["filename"],
        filenames_and_dirs["filename_reg"],
        filenames_and_dirs["filename_dir"],
        filenames_and_dirs["filename_dir_reg"],
    )

    dyn_reg_unsup(
        args,
        hparams,
        data_dir_old,
        buckets,
        device,
        status_cls,
        ckpt_cls=ckpt_cls,
    )
