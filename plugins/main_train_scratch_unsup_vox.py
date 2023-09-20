import torch

from utils import (
    parse_args,
    HyperParams,
    create_filenames_scratch_unsupervised_proto_vox,
    create_dvec_latent_scratch_checkpoint_dir,
    AngleProtoLoss,
)
from training import train_from_scratch_unsup_vox


def main_execute():
    args = parse_args()
    hparams = HyperParams()

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    # Dictionaries of filenames for the checkpoints of classifier
    filename, filename_dir = create_filenames_scratch_unsupervised_proto_vox(args)

    # Set ``ckpt_cls'' to the available checkpoint
    ckpt_cls, _ = create_dvec_latent_scratch_checkpoint_dir(
        args,
        filename,
        filename_dir,
    )

    train_from_scratch_unsup_vox(
        args,
        hparams,
        buckets,
        AngleProtoLoss,
        device,
        ckpt_cls,
    )
