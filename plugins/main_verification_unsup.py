import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_dvec_unsupervised_latent,
)
from verification import verification_performance_unsup


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    filenames_and_dirs = create_filenames_dvec_unsupervised_latent(
        args,
        hparams,
    )

    ckpt_dvec_latent = filenames_and_dirs["filename_dir"]

    verification_performance_unsup(
        args,
        hparams,
        buckets,
        device,
        ckpt_dvec_latent=ckpt_dvec_latent,
    )
