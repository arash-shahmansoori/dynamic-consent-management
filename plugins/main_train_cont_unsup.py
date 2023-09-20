import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_dvec_unsupervised_latent,
    create_dvec_latent_checkpoint_dir,
)
from training import train_contrastive_unsupervised


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Filenames for the checkpoints of latent d-vec
    filenames_and_dirs = create_filenames_dvec_unsupervised_latent(
        args,
        hparams,
    )

    # Set ``ckpt_dvec_latent'' to the available checkpoint
    ckpt_dvec_latent, _ = create_dvec_latent_checkpoint_dir(
        args,
        filenames_and_dirs["filename"],
        filenames_and_dirs["filename_dir"],
    )

    train_contrastive_unsupervised(
        args,
        hparams,
        buckets,
        device,
        ckpt_dvec_latent=ckpt_dvec_latent,
    )
