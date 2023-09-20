import torch
import numpy as np
import pandas as pd

from pathlib import Path

from utils import (
    tsne,
    create_filenames_tsne_unsup_results,
    save_as_json,
    create_filenames_dvec_unsupervised_latent,
    create_cls_checkpoint_dir_reg_unsup,
)

from .tsne_preparation_old_spks_unsup import prepare_data_tsne_old_spks_unsup
from .tsne_preparation_new_spks_unsup import prepare_data_tsne_new_spks_unsup


def create_tsne_dyn_reg_unsup(
    rounds_indx,
    args,
    hparams,
):
    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    filenames_and_dirs = create_filenames_dvec_unsupervised_latent(
        args,
        hparams,
    )

    ckpt_cls, _ = create_cls_checkpoint_dir_reg_unsup(
        args,
        filenames_and_dirs["filename"],
        filenames_and_dirs["filename_reg"],
        filenames_and_dirs["filename_dir"],
        filenames_and_dirs["filename_dir_reg"],
    )

    # Create paths for saving the plots
    result_dir_plot = args.output_dir_results
    result_dir_plot_path = Path(result_dir_plot)
    result_dir_plot_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for old and new speaker registrations
    embs_old = prepare_data_tsne_old_spks_unsup(
        args,
        hparams,
        buckets,
        device,
        ckpt_cls,
    )

    embs_new_storage = []
    for _, i in enumerate(rounds_indx):
        embs_new = prepare_data_tsne_new_spks_unsup(
            args,
            hparams,
            buckets,
            i,
            device,
            ckpt_cls,
        )
        embs_new_storage.append(torch.tensor(embs_new))

    old_activation = torch.tensor(embs_old)
    new_activation = torch.cat(embs_new_storage, dim=0)

    combined_activation = np.array(torch.cat([old_activation, new_activation], dim=0))

    create_binary_labels = []
    for i in range(old_activation.shape[0] + new_activation.shape[0]):
        if i < old_activation.shape[0]:
            create_binary_labels.append(r"old-speakers")
        else:
            create_binary_labels.append(r"new-speakers")

    Y = tsne(combined_activation, 2, args.latent_dim, 20)

    dfY = pd.DataFrame(Y)
    dfY[r"dynamic registrations"] = np.array(create_binary_labels)

    dfY.columns = ["x1", "x2", "dynamic registrations"]

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_tsne_unsup_results(args, hparams)

    save_as_json(
        paths_filenames["dir_tsne"],
        paths_filenames["tsne_pred_feats"],
        torch.tensor(Y).tolist(),
    )

    save_as_json(
        paths_filenames["dir_tsne"],
        paths_filenames["tsne_pred_labels"],
        create_binary_labels,
    )
