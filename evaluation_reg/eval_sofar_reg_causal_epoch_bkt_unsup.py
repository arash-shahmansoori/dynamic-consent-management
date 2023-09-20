import torch
import numpy as np

from torch.utils.data import DataLoader


def eval_sofar_reg_causal_per_round_per_epoch_per_bkt_unsup(
    indx_opt_unique_bkt_sofar,
    indx_selected_id_sofar,
    args,
    outputs,
    indx_selected,
    device,
    **kwargs_validation,
):

    sub_lbs_old = outputs[indx_selected]
    sub_lbs_other_sofar = np.array(indx_opt_unique_bkt_sofar)[
        indx_selected_id_sofar
    ].tolist()

    sub_dataset_validation = kwargs_validation["SubDatasetGdrSpk"](
        kwargs_validation["dataset_val"],
        sub_lbs_old,
    )
    validation_sub_loader = DataLoader(
        sub_dataset_validation,
        batch_size=len(sub_lbs_old),
        collate_fn=kwargs_validation["collateGdrSpkr"],
        drop_last=True,
    )

    sub_dataset_other_validation = kwargs_validation["SubDatasetGdrSpk"](
        kwargs_validation["dataset_other_val"],
        sub_lbs_other_sofar,
    )
    validation_sub_loader_other = DataLoader(
        sub_dataset_other_validation,
        batch_size=len(sub_lbs_other_sofar),
        collate_fn=kwargs_validation["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch_validation = next(iter(validation_sub_loader))
    mel_db_batch_other_validation = next(iter(validation_sub_loader_other))

    x_val, _, spk_val = mel_db_batch_validation
    x_val = x_val.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk_val = spk_val.to(device)

    x_other_val, _, spk_other_val = mel_db_batch_other_validation
    x_other_val = x_other_val.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk_other_val = spk_other_val.to(device)

    x_val_old = x_val
    x_val_new = x_other_val
    x_cat_val = torch.cat((x_val, x_other_val), dim=0)

    spk_val_old = spk_val
    spk_val_new = spk_other_val + args.n_speakers
    spk_val_cat = torch.cat((spk_val, spk_val_new), dim=0)

    xe_val_old = kwargs_validation["dvectors"][indx_selected](x_val_old).detach()
    xe_val_new = kwargs_validation["dvectors"][indx_selected](x_val_new).detach()
    xe_val_cat = kwargs_validation["dvectors"][indx_selected](x_cat_val).detach()

    return {
        "spk_val_cat": spk_val_cat,
        "xe_val_cat": xe_val_cat,
        "indx_selected_new_spks_overall": sub_lbs_other_sofar,
        "indx_selected": indx_selected,
    }
