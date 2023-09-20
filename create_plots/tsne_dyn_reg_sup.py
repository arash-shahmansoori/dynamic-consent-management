import torch
import pandas as pd
import numpy as np

from pathlib import Path

from utils import (
    tsne,
    create_filenames_cls,
    create_cls_checkpoint_dir_reg,
    create_filenames_tsne_results,
    save_as_json,
)

from .tsne_preparation_old_spks import prepare_data_tsne_old_spks
from .tsne_preparation_new_spks import prepare_data_tsne_new_spks


def create_tsne_dyn_reg_sup(
    rounds_indx,
    args,
    hparams,
):

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of buckets
    buckets = [bucket_id for bucket_id in range(hparams.num_of_buckets)]

    filenames_and_dirs = create_filenames_cls(args, hparams)

    ckpt_cls, _ = create_cls_checkpoint_dir_reg(
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
    embs_old, pred_spk_old, _ = prepare_data_tsne_old_spks(
        args,
        hparams,
        buckets,
        device,
        ckpt_cls,
    )

    embs_new_storage, pred_spk_new_storage = [], []
    for _, i in enumerate(rounds_indx):
        embs_new, pred_spk_new, _ = prepare_data_tsne_new_spks(
            args,
            hparams,
            buckets,
            i,
            device,
            ckpt_cls,
        )
        embs_new_storage.append(torch.tensor(embs_new))
        pred_spk_new_storage.append(torch.tensor(pred_spk_new))

    old_activation = torch.tensor(embs_old)
    new_activation = torch.cat(embs_new_storage, dim=0)

    pred_old_labels = torch.tensor(pred_spk_old).view(-1, 1)
    pred_new_labels = torch.cat(pred_spk_new_storage, dim=0).view(-1, 1)

    combined_activation = np.array(torch.cat([old_activation, new_activation], dim=0))
    pred_combined_labels = np.array(
        torch.cat([pred_old_labels, pred_new_labels], dim=0).view(-1)
    )

    create_binary_labels = []
    for i in pred_combined_labels:
        if i < 40:
            create_binary_labels.append(r"Old speakers")
        else:
            create_binary_labels.append(r"New speakers")

    Y = tsne(combined_activation, 2, args.latent_dim, 20)

    dfY = pd.DataFrame(Y)
    dfY[r"dynamic registrations"] = np.array(create_binary_labels)

    dfY.columns = ["x1", "x2", "dynamic registrations"]

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_tsne_results(args, hparams)

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
