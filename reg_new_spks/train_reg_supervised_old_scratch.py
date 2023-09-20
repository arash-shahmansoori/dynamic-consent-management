import torch

from torch.utils.data import DataLoader


def train_selective_reg_inductive_bias_old_sup_scratch(
    epoch,
    early_stopping,
    args,
    outputs,
    indx_selected,
    device,
    kwargs_training_val,
):

    sub_lbs_old1 = outputs[indx_selected]

    sub_dataset1 = kwargs_training_val["SubDatasetGdrSpk"](
        kwargs_training_val["dataset"], sub_lbs_old1
    )
    train_sub_loader1 = DataLoader(
        sub_dataset1,
        batch_size=len(sub_lbs_old1),
        shuffle=False,
        collate_fn=kwargs_training_val["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch1 = next(iter(train_sub_loader1))

    x1, _, spk1 = mel_db_batch1
    x1 = x1.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk1 = spk1.to(device)

    # Create input data
    input_data = {"x": x1, "y": spk1}

    if early_stopping.early_stop:
        return input_data

    # Re-train the d-vector with the new registered bucket
    kwargs_training_val["agent"].train_dvec(
        kwargs_training_val["dvectors"][indx_selected],
        kwargs_training_val["opt_dvec_type"],
        kwargs_training_val["contrastive_loss"],
        indx_selected,
        input_data,
        epoch,
        kwargs_training_val["filename_dvec_reg"][indx_selected],
        kwargs_training_val["filename_dvec_dir_reg"][indx_selected],
    )

    return input_data
