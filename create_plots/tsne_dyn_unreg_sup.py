import torch
import pandas as pd
import numpy as np

from pathlib import Path

from utils import (
    tsne,
    create_filenames_cls,
    create_cls_checkpoint_dir_unreg,
    create_filenames_unreg_tsne_results,
    save_as_json,
)

from .tsne_preparation_unreg_spks import prepare_data_tsne_unreg_spks


def create_tsne_dyn_unreg_sup(unreg_spks, bucket, args, hparams):

    # Specify the device to run the simulations on
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Filenames for the checkpoints of classifier
    filenames_and_dirs = create_filenames_cls(args, hparams, unreg_spks)

    ckpt_cls, _ = create_cls_checkpoint_dir_unreg(
        args,
        filenames_and_dirs["filename"],
        # filenames_and_dirs["filename_unreg"],
        None,
        filenames_and_dirs["filename_dir"],
        # filenames_and_dirs["filename_dir_unreg"],
        None,
    )

    # Prepare data for old and new speaker registrations
    embs, pred_spk, spk = prepare_data_tsne_unreg_spks(
        args,
        hparams,
        bucket,
        unreg_spks,
        device,
        ckpt_cls,
    )

    activation = torch.tensor(embs)
    # pred_labels = torch.tensor(pred_spk).view(-1, 1)
    true_labels = torch.tensor(spk).view(-1, 1)
    # print(pred_labels)

    create_binary_labels = []
    for i in true_labels:
        if i in unreg_spks:
            create_binary_labels.append(r"Unregistered speakers")
        else:
            create_binary_labels.append(r"Residual speakers")

    Y = tsne(activation, 2, args.dim_emb, 20)

    dfY = pd.DataFrame(Y)
    dfY[r"dynamic removal"] = np.array(create_binary_labels)

    dfY.columns = ["x1", "x2", "dynamic removal"]

    # Create paths and filenames for saving the training/validation metrics
    paths_filenames = create_filenames_unreg_tsne_results(args, hparams, bucket)

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
