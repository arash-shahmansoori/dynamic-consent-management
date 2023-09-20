import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_scratch_unsupervised,
)
from verification import verification_performance_literature_unsup


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device to run the simulations on
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    _, filename_dir = create_filenames_scratch_unsupervised(args)

    verification_performance_literature_unsup(
        args,
        hparams,
        device,
        ckpt_dvec_latent=filename_dir,
    )