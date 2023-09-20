def dataset_kwargs(
    SubDatasetGdrSpk,
    collateGdrSpkr,
    dataset,
    dataset_val,
    dataset_other=None,
    dataset_other_val=None,
):
    """Create the dictionary of dataset(s) as input arguments for the training/validation function."""
    dataset_kwargs_dict = {
        "SubDatasetGdrSpk": SubDatasetGdrSpk,
        "collateGdrSpkr": collateGdrSpkr,
        "dataset": dataset,
        "dataset_val": dataset_val,
        "dataset_other": dataset_other,
        "dataset_other_val": dataset_other_val,
    }

    return dataset_kwargs_dict


def dataset_spk_kwargs(
    SubDatasetSpk,
    collateSpkr,
    dataset,
    dataset_val,
):
    """Create the dictionary of dataset(s) as input arguments for the training/validation function."""
    dataset_kwargs_dict = {
        "SubDatasetSpk": SubDatasetSpk,
        "collateSpkr": collateSpkr,
        "dataset": dataset,
        "dataset_val": dataset_val,
    }

    return dataset_kwargs_dict


def filename_kwargs_dvec(
    filename_dvec,
    filename_dvec_reg,
    filename_dvec_unreg,
    filename_dvec_re_reg,
    filename_dvec_dir,
    filename_dvec_dir_reg,
    filename_dvec_dir_unreg,
):
    """Create the dictionary of d-vector filenames as input arguments for the training function."""

    filename_kwargs_dict = {
        "filename_dvec": filename_dvec,
        "filename_dvec_reg": filename_dvec_reg,
        "filename_dvec_unreg": filename_dvec_unreg,
        "filename_dvec_re_reg": filename_dvec_re_reg,
        "filename_dvec_dir": filename_dvec_dir,
        "filename_dvec_dir_reg": filename_dvec_dir_reg,
        "filename_dvec_dir_unreg": filename_dvec_dir_unreg,
    }

    return filename_kwargs_dict


def filename_kwargs_cls(
    filename,
    filename_reg,
    filename_unreg,
    filename_dir,
    filename_dir_reg,
    filename_dir_unreg,
):
    """Create the dictionary of classifier filenames as input arguments for the training function."""

    filename_kwargs_dict = {
        "filename": filename,
        "filename_reg": filename_reg,
        "filename_unreg": filename_unreg,
        "filename_dir": filename_dir,
        "filename_dir_reg": filename_dir_reg,
        "filename_dir_unreg": filename_dir_unreg,
    }

    return filename_kwargs_dict


def filename_kwargs_scratch(filename, filename_dir):
    """Create the dictionary of classifier filenames as input arguments for the training function."""

    filename_kwargs_dict = {"filename": filename, "filename_dir": filename_dir}

    return filename_kwargs_dict


def model_kwargs(agent, dvectors, classifier):
    """Create the dictionary of models as input arguments for the training function."""
    model_kwargs_dict = {"agent": agent, "dvectors": dvectors, "classifier": classifier}

    return model_kwargs_dict


def model_kwargs_unsupervised(agent, dvectors, dvec_latent):
    """Create the dictionary of models as input arguments for the training function."""
    model_kwargs_dict = {
        "agent": agent,
        "dvectors": dvectors,
        "dvec_latent": dvec_latent,
    }

    return model_kwargs_dict


def model_kwargs_unsupervised_unreg(agent, dvectors, classifier):
    """Create the dictionary of models as input arguments for the training function."""
    model_kwargs_dict = {
        "agent": agent,
        "dvectors": dvectors,
        "classifier": classifier,
    }

    return model_kwargs_dict


def model_unsupervised_kwargs(agent, dvectors):
    """Create the dictionary of models as input arguments for the training function."""
    model_kwargs_dict = {"agent": agent, "dvectors": dvectors}

    return model_kwargs_dict


def opt_kwargs(
    opt_dvec_type,
    opt_dvecs,
    opt_cls_type,
    optimizer,
    early_stop,
):
    """Create the dictionary of optimizers as input arguments for the training function."""
    opt_kwargs_dict = {
        "opt_dvec_type": opt_dvec_type,
        "opt_dvecs": opt_dvecs,
        "opt_cls_type": opt_cls_type,
        "optimizer": optimizer,
        "early_stop": early_stop,
    }

    return opt_kwargs_dict


def loss_kwargs(contrastive_loss, ce_loss):
    """Create the dictionary of losses as input arguments for the training function."""
    loss_kwargs_dict = {"contrastive_loss": contrastive_loss, "ce_loss": ce_loss}

    return loss_kwargs_dict


def loss_kwargs_unsupervised(contrastive_loss, contrastive_loss_latent):
    """Create the dictionary of losses as input arguments for the training function."""
    loss_kwargs_dict = {
        "contrastive_loss": contrastive_loss,
        "contrastive_loss_latent": contrastive_loss_latent,
    }

    return loss_kwargs_dict