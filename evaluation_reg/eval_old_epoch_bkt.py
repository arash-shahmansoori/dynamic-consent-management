from torch.utils.data import DataLoader


def eval_old_per_round_per_epoch_per_bkt(
    args,
    outputs,
    indx_selected,
    device,
    **kwargs_validation,
):

    sub_lbs_old = outputs[indx_selected]

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

    mel_db_batch_validation = next(iter(validation_sub_loader))

    x_val, _, spk_val = mel_db_batch_validation
    x_val = x_val.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk_val = spk_val.to(device)

    x_val_old = x_val

    spk_val_old = spk_val

    xe_val_old = kwargs_validation["dvectors"][indx_selected](x_val_old).detach()

    return {
        "spk_val_old": spk_val_old,
        "spk_val_new": [],
        "spk_val_cat": spk_val_old,
        "xe_val_old": xe_val_old,
        "xe_val_new": [],
        "xe_val_cat": xe_val_old,
        "indx_selected_spk": -1,
        "indx_selected_new_spks_overall": -1,
        "indx_selected": indx_selected,
    }
