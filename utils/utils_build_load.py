import os
import torch


from abc import ABC, abstractmethod
from pathlib import Path


class BuildLoadModel(ABC):
    """Build and load model according to the available checkpoints."""

    @abstractmethod
    def build_model(self, *args):
        """Build the model."""

    @abstractmethod
    def load_model(self, *args):
        """Load the model from the corresponding checkpoints."""


class BuildOptimizer(ABC):
    """Build optimizer according to the available checkpoints."""

    @abstractmethod
    def build_optimizer(self, *args):
        """Build the optimizer."""


class BuildLoadModelOptimizer:
    """Build and load model & optimizer according to the available checkpoints."""

    def __init__(self, model_obj: BuildLoadModel, opt_obj: BuildOptimizer):
        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, *args):
        """Build the model & the optimizer."""
        raise NotImplementedError

    def load_model_opt(self, *args):
        """Load the model & the optimizer from the corresponding checkpoints."""
        raise NotImplementedError


def cont_loss_loader_dvec(args, hparams, buckets, cont_loss_dict, filename_dict):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_dict = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}
    for _, _, files in os.walk(checkpoint_dir_path):
        for _, bkt_id in enumerate(buckets):
            start_epoch_dict[bkt_id] = 0

            for file in files:
                if file == filename_dict[bkt_id]:
                    checkpoint = torch.load(checkpoint_dir_path / filename_dict[bkt_id])

                    start_epoch_dict[bkt_id] = checkpoint["epoch"]
                    cont_loss_dict[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    return cont_loss_dict, start_epoch_dict


def model_loader_dvec(args, hparams, buckets, model_dict, filename_dict):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_dict = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}
    for _, _, files in os.walk(checkpoint_dir_path):
        for _, bkt_id in enumerate(buckets):
            start_epoch_dict[bkt_id] = 0

            for file in files:
                if file == filename_dict[bkt_id]:
                    checkpoint = torch.load(checkpoint_dir_path / filename_dict[bkt_id])

                    start_epoch_dict[bkt_id] = checkpoint["epoch"]
                    model_dict[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    return model_dict, start_epoch_dict


def cont_loss_loader(
    args,
    hparams,
    buckets,
    cont_loss_storage,
    filename_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename:
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    return cont_loss_storage, start_epoch_storage


def dvec_model_loader(
    args,
    hparams,
    buckets,
    model_storage,
    filename_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename:
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    return model_storage, start_epoch_storage


def cont_loss_loader_dvec_latent(args, hparams, cont_loss, filename):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_path):
        start_epoch = 0

        for file in files:
            if file == filename:
                checkpoint = torch.load(checkpoint_dir_path / filename)

                start_epoch = checkpoint["epoch"]
                cont_loss.load_state_dict(checkpoint[hparams.contloss_str])

    return cont_loss, start_epoch


def model_loader_dvec_latent(args, hparams, model, filename):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_path):
        start_epoch = 0

        for file in files:
            if file == filename:
                checkpoint = torch.load(checkpoint_dir_path / filename)

                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint[hparams.model_str])

    return model, start_epoch


def model_loader_dvec_dynamic_reg(
    args,
    hparams,
    buckets,
    model_dict,
    filename_dict,
    filename_dict_reg,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_dict = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}
    for _, _, files in os.walk(checkpoint_dir_path):
        for _, bkt_id in enumerate(buckets):
            start_epoch_dict[bkt_id] = 0

            for file in files:
                if file == filename_dict_reg[bkt_id]:
                    checkpoint = torch.load(
                        checkpoint_dir_path / filename_dict_reg[bkt_id]
                    )

                    start_epoch_dict[bkt_id] = checkpoint["epoch"]
                    model_dict[bkt_id].load_state_dict(checkpoint[hparams.model_str])

                elif (
                    file == filename_dict[bkt_id] and file != filename_dict_reg[bkt_id]
                ):
                    checkpoint = torch.load(checkpoint_dir_path / filename_dict[bkt_id])

                    start_epoch_dict[bkt_id] = checkpoint["epoch"]
                    model_dict[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    return model_dict, start_epoch_dict


def cont_loss_loader_dvec_dynamic_reg(
    args,
    hparams,
    buckets,
    cont_loss_dict,
    filename_dict,
    filename_dict_reg,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_dict = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}
    for _, _, files in os.walk(checkpoint_dir_path):
        for _, bkt_id in enumerate(buckets):
            start_epoch_dict[bkt_id] = 0

            for file in files:
                if file == filename_dict_reg[bkt_id]:
                    checkpoint = torch.load(
                        checkpoint_dir_path / filename_dict_reg[bkt_id]
                    )

                    start_epoch_dict[bkt_id] = checkpoint["epoch"]
                    cont_loss_dict[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

                elif (
                    file == filename_dict[bkt_id] and file != filename_dict_reg[bkt_id]
                ):
                    checkpoint = torch.load(checkpoint_dir_path / filename_dict[bkt_id])

                    start_epoch_dict[bkt_id] = checkpoint["epoch"]
                    cont_loss_dict[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    return cont_loss_dict, start_epoch_dict


def dvec_model_loader_dynamic_reg(
    args,
    hparams,
    buckets,
    model_storage,
    filename_storage,
    filename_reg_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    bkt_id_reg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_reg in filename_reg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_reg:
                    bkt_id_reg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_reg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename and bkt_id not in bkt_id_reg:
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    return model_storage, start_epoch_storage


def dvec_model_loader_dynamic_unreg(
    args,
    hparams,
    buckets,
    model_storage,
    unreg_bkts_storage,
    filename_storage,
    filename_unreg_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    bkt_id_unreg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_unreg in filename_unreg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_unreg:
                    bkt_id_unreg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_unreg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if (
                    file == filename
                    and (bkt_id not in bkt_id_unreg)
                    and (bkt_id not in unreg_bkts_storage)
                ):
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    return model_storage, start_epoch_storage


def dvec_model_loader_dynamic_re_reg(
    args,
    hparams,
    buckets,
    model_storage,
    spk_per_bkt_storage_old,
    filename_storage,
    filename_unreg_storage,
    filename_re_reg_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    bkt_id_re_reg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_re_reg in filename_re_reg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_re_reg:
                    bkt_id_re_reg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_re_reg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    bkt_id_unreg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_unreg in filename_unreg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_unreg and (bkt_id not in bkt_id_re_reg):
                    bkt_id_unreg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_unreg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if (
                    file == filename
                    and (bkt_id not in bkt_id_re_reg)
                    and (bkt_id not in bkt_id_unreg)
                    and (spk_per_bkt_storage_old[bkt_id] != 0)
                ):
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    model_storage[bkt_id].load_state_dict(checkpoint[hparams.model_str])

    return model_storage, start_epoch_storage


def cont_loss_loader_dynamic_reg(
    args,
    hparams,
    buckets,
    cont_loss_storage,
    filename_storage,
    filename_reg_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    bkt_id_reg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_reg in filename_reg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_reg:
                    bkt_id_reg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_reg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename and bkt_id not in bkt_id_reg:
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    return cont_loss_storage, start_epoch_storage


def cont_loss_loader_dynamic_unreg(
    args,
    hparams,
    buckets,
    cont_loss_storage,
    unreg_bkts_storage,
    filename_storage,
    filename_reg_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    bkt_id_reg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_reg in filename_reg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_reg:
                    bkt_id_reg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_reg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if (
                    file == filename
                    and (bkt_id not in bkt_id_reg)
                    and (bkt_id not in unreg_bkts_storage)
                ):
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    return cont_loss_storage, start_epoch_storage


def cont_loss_loader_dynamic_re_reg(
    args,
    hparams,
    buckets,
    cont_loss_storage,
    spk_per_bkt_storage_old,
    filename_storage,
    filename_unreg_storage,
    filename_re_reg_storage,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    start_epoch_storage = {bkt_ids: 0 for _, bkt_ids in enumerate(buckets)}

    bkt_id_re_reg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_re_reg in filename_re_reg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_re_reg:
                    bkt_id_re_reg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_re_reg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    bkt_id_unreg = []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename_unreg in filename_unreg_storage.items():
                start_epoch_storage[bkt_id] = 0
                if file == filename_unreg and (bkt_id not in bkt_id_re_reg):
                    bkt_id_unreg.append(bkt_id)
                    checkpoint = torch.load(checkpoint_dir_path / filename_unreg)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            for bkt_id, filename in filename_storage.items():
                start_epoch_storage[bkt_id] = 0
                if (
                    file == filename
                    and (bkt_id not in bkt_id_re_reg)
                    and (bkt_id not in bkt_id_unreg)
                    and (spk_per_bkt_storage_old[bkt_id] != 0)
                ):
                    checkpoint = torch.load(checkpoint_dir_path / filename)
                    start_epoch_storage[bkt_id] = checkpoint["epoch"]
                    cont_loss_storage[bkt_id].load_state_dict(
                        checkpoint[hparams.contloss_str]
                    )

    return cont_loss_storage, start_epoch_storage


def cont_loss_loader_dvec_latent_dynamic_reg(
    args,
    hparams,
    cont_loss,
    filename,
    filename_reg,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_path):
        start_epoch = 0

        for file in files:
            if file == filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename_reg)

                start_epoch = checkpoint["epoch"]
                cont_loss.load_state_dict(checkpoint[hparams.contloss_str])

            elif file == filename and file != filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename)

                start_epoch = checkpoint["epoch"]
                cont_loss.load_state_dict(checkpoint[hparams.contloss_str])

    return cont_loss, start_epoch


def model_loader_dvec_latent_dynamic_reg(
    args,
    hparams,
    model,
    filename,
    filename_reg,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_path):
        start_epoch = 0

        for file in files:
            if file == filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename_reg)

                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint[hparams.model_str])

            elif file == filename and file != filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename)

                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint[hparams.model_str])

    return model, start_epoch


def model_loader_dvec_per_bkt(hparams, model, args, filename):
    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    start_epoch = 0

    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                checkpoint = torch.load(checkpoint_dir_dvector_path / filename)
                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint[hparams.model_str])

    return model, start_epoch


def contloss_loader_per_bkt(hparams, contloss, args, filename):
    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    start_epoch = 0

    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                checkpoint = torch.load(checkpoint_dir_dvector_path / filename)
                start_epoch = checkpoint["epoch"]
                contloss.load_state_dict(checkpoint[hparams.contloss_str])

    return contloss, start_epoch


def opt_loader_dvec_per_bkt(hparams, opt_dvec, args, filename):
    checkpoint_dir_dvector = args.checkpoint_dir_dvector
    checkpoint_dir_dvector_path = Path(checkpoint_dir_dvector)
    checkpoint_dir_dvector_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_dvector_path):
        for file in files:
            if file == filename:
                checkpoint = torch.load(checkpoint_dir_dvector_path / filename)
                opt_dvec.load_state_dict(checkpoint[hparams.opt_str])

    return opt_dvec


def model_loader_dvec_dynamic_reg_per_bkt(
    args,
    hparams,
    model,
    filename,
    filename_reg,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_path):
        start_epoch = 0

        for file in files:
            if file == filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename_reg)

                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint[hparams.model_str])

            elif file == filename and file != filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename)

                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint[hparams.model_str])

    return model, start_epoch


def cont_loss_loader_dvec_dynamic_reg_per_bkt(
    args,
    hparams,
    cont_loss,
    filename,
    filename_reg,
):
    checkpoint_dir = args.checkpoint_dir_dvector
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for _, _, files in os.walk(checkpoint_dir_path):
        start_epoch = 0

        for file in files:
            if file == filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename_reg)

                start_epoch = checkpoint["epoch"]
                cont_loss.load_state_dict(checkpoint[hparams.contloss_str])

            elif file == filename and file != filename_reg:
                checkpoint = torch.load(checkpoint_dir_path / filename)

                start_epoch = checkpoint["epoch"]
                cont_loss.load_state_dict(checkpoint[hparams.contloss_str])

    return cont_loss, start_epoch


class DvecModel(BuildLoadModel):
    """Build and load dvectors."""

    def __init__(self, device, buckets, args):
        self.device = device
        self.buckets = buckets
        self.args = args

    def build_model(self, model):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)

        return model_dict

    def load_model(self, hparams, model, filename_storage):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage = self.build_model(model)
        model_storage, start_epoch_storage = dvec_model_loader(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            filename_storage,
        )

        return model_storage, start_epoch_storage


class DvecModelUnsupervised(BuildLoadModel):
    """Build and load dvectors."""

    def __init__(self, device, buckets, args):
        self.device = device
        self.buckets = buckets
        self.args = args

    def build_model(self, model, cont_loss):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        cont_loss_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)
            cont_loss_dict[bkt_id] = cont_loss(self.args).to(self.device)

        return model_dict, cont_loss_dict

    def load_model(self, hparams, model, cont_loss, filename_storage):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage, _cont_loss_storage = self.build_model(model, cont_loss)
        model_storage, start_epoch_storage = dvec_model_loader(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            filename_storage,
        )
        cont_loss_storage, _ = cont_loss_loader(
            self.args,
            hparams,
            self.buckets,
            _cont_loss_storage,
            filename_storage,
        )

        return model_storage, cont_loss_storage, start_epoch_storage


class DvecModelDynamicReg(BuildLoadModel):
    """Build and load dvectors for dynamic registrations."""

    def __init__(self, device, buckets, args):
        self.device = device
        self.buckets = buckets
        self.args = args

    def build_model(self, model):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)

        return model_dict

    def load_model(self, hparams, model, filename_storage, filename_storage_reg):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage = self.build_model(model)
        model_storage, start_epoch_storage = dvec_model_loader_dynamic_reg(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            filename_storage,
            filename_storage_reg,
        )

        return model_storage, start_epoch_storage


class DvecModelDynamicUnReg(BuildLoadModel):
    """Build and load dvectors for dynamic registrations."""

    def __init__(self, device, buckets, unreg_bkts_storage, args):
        self.device = device
        self.buckets = buckets
        self.unreg_bkts_storage = unreg_bkts_storage
        self.args = args

    def build_model(self, model):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)

        return model_dict

    def load_model(self, hparams, model, filename_storage, filename_storage_unreg):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage = self.build_model(model)
        model_storage, start_epoch_storage = dvec_model_loader_dynamic_unreg(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            self.unreg_bkts_storage,
            filename_storage,
            filename_storage_unreg,
        )

        return model_storage, start_epoch_storage


class DvecModelDynamicReReg(BuildLoadModel):
    """Build and load dvectors for dynamic re-registrations."""

    def __init__(self, device, buckets, spk_per_bkt_storage_old, args):
        self.device = device
        self.buckets = buckets
        self.spk_per_bkt_storage_old = spk_per_bkt_storage_old
        self.args = args

    def build_model(self, model):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)

        return model_dict

    def load_model(
        self,
        hparams,
        model,
        filename_storage,
        filename_storage_reg,
        filename_storage_re_reg,
    ):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage = self.build_model(model)
        model_storage, start_epoch_storage = dvec_model_loader_dynamic_re_reg(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            self.spk_per_bkt_storage_old,
            filename_storage,
            filename_storage_reg,
            filename_storage_re_reg,
        )

        return model_storage, start_epoch_storage


class DvecModelDynamicRegUnsupervised(BuildLoadModel):
    """Build and load dvectors for dynamic registrations."""

    def __init__(self, device, buckets, args):
        self.device = device
        self.buckets = buckets
        self.args = args

    def build_model(self, model, cont_loss):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        cont_loss_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)
            cont_loss_dict[bkt_id] = cont_loss(self.args).to(self.device)

        return model_dict, cont_loss_dict

    def load_model(
        self,
        hparams,
        model,
        cont_loss,
        filename_storage,
        filename_storage_reg,
    ):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage, _cont_loss_storage = self.build_model(model, cont_loss)
        model_storage, start_epoch_storage = dvec_model_loader_dynamic_reg(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            filename_storage,
            filename_storage_reg,
        )
        cont_loss_storage, _ = cont_loss_loader_dynamic_reg(
            self.args,
            hparams,
            self.buckets,
            _cont_loss_storage,
            filename_storage,
            filename_storage_reg,
        )

        return model_storage, cont_loss_storage, start_epoch_storage


class DvecModelDynamicUnRegUnsupervised(BuildLoadModel):
    """Build and load dvectors for dynamic registrations."""

    def __init__(self, device, buckets, unreg_bkts_storage, args):
        self.device = device
        self.buckets = buckets
        self.unreg_bkts_storage = unreg_bkts_storage
        self.args = args

    def build_model(self, model, cont_loss):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        cont_loss_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)
            cont_loss_dict[bkt_id] = cont_loss(self.args).to(self.device)

        return model_dict, cont_loss_dict

    def load_model(
        self, hparams, model, cont_loss, filename_storage, filename_storage_reg
    ):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage, _cont_loss_storage = self.build_model(model, cont_loss)
        model_storage, start_epoch_storage = dvec_model_loader_dynamic_unreg(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            self.unreg_bkts_storage,
            filename_storage,
            filename_storage_reg,
        )
        cont_loss_storage, _ = cont_loss_loader_dynamic_unreg(
            self.args,
            hparams,
            self.buckets,
            _cont_loss_storage,
            self.unreg_bkts_storage,
            filename_storage,
            filename_storage_reg,
        )

        return model_storage, cont_loss_storage, start_epoch_storage


class DvecModelDynamicReRegUnsupervised(BuildLoadModel):
    """Build and load dvectors for dynamic registrations."""

    def __init__(self, device, buckets, spk_per_bkt_storage_old, args):
        self.device = device
        self.buckets = buckets
        self.spk_per_bkt_storage_old = spk_per_bkt_storage_old
        self.args = args

    def build_model(self, model, cont_loss):
        """Build the dvectors."""
        model_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        cont_loss_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            model_dict[bkt_id] = model(self.args).to(self.device)
            cont_loss_dict[bkt_id] = cont_loss(self.args).to(self.device)

        return model_dict, cont_loss_dict

    def load_model(
        self,
        hparams,
        model,
        cont_loss,
        filename_storage,
        filename_storage_unreg,
        filename_storage_re_reg,
    ):
        """Load the dvectors from the corresponding checkpoints."""

        _model_storage, _cont_loss_storage = self.build_model(model, cont_loss)
        model_storage, start_epoch_storage = dvec_model_loader_dynamic_re_reg(
            self.args,
            hparams,
            self.buckets,
            _model_storage,
            self.spk_per_bkt_storage_old,
            filename_storage,
            filename_storage_unreg,
            filename_storage_re_reg,
        )
        cont_loss_storage, _ = cont_loss_loader_dynamic_re_reg(
            self.args,
            hparams,
            self.buckets,
            _cont_loss_storage,
            self.spk_per_bkt_storage_old,
            filename_storage,
            filename_storage_unreg,
            filename_storage_re_reg,
        )

        return model_storage, cont_loss_storage, start_epoch_storage


class DvecOptimizer(BuildOptimizer):
    """Build the d-vector optimizer with available checkpoints."""

    def __init__(self, device, buckets, args, hparams):
        self.device = device
        self.buckets = buckets
        self.args = args

        self.lr = hparams.lr
        self.momentum = hparams.momentum
        self.nesterov = hparams.nesterov
        self.dampening = hparams.dampening
        self.weight_decay = hparams.weight_decay

    def build_optimizer(self, model_dict, opt, contrastive_loss):
        """Build the d-vector optimizer."""

        criterion = contrastive_loss(self.args).to(self.device)

        opt_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            # SGD optimizer
            # opt_dict[bkt_id] = opt(
            #     [
            #         {
            #             "params": list(model_dict[bkt_id].parameters())
            #             + list(criterion.parameters()),
            #             "weight_decay": self.weight_decay,
            #         }
            #     ],
            #     self.lr,
            #     self.momentum,
            #     self.nesterov,
            #     self.dampening,
            # )

            # SophiaG optimizer
            opt_dict[bkt_id] = opt(
                list(model_dict[bkt_id].parameters()) + list(criterion.parameters()),
                lr=3e-4,
                betas=(0.9, 0.95),
                rho=0.03,
            )

        return opt_dict


class DvecOptimizerUnsupervised(BuildOptimizer):
    """Build the d-vector optimizer with available checkpoints."""

    def __init__(self, device, buckets, args, hparams):
        self.device = device
        self.buckets = buckets
        self.args = args

        self.lr = hparams.lr
        self.momentum = hparams.momentum
        self.nesterov = hparams.nesterov
        self.dampening = hparams.dampening
        self.weight_decay = hparams.weight_decay

    def build_optimizer(self, model_dict, opt, contrastive_loss):
        """Build the d-vector optimizer."""

        opt_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for _, bkt_id in enumerate(self.buckets):
            # opt_dict[bkt_id] = opt(
            #     [
            #         {
            #             "params": list(model_dict[bkt_id].parameters())
            #             + list(contrastive_loss[bkt_id].parameters()),
            #             "weight_decay": self.weight_decay,
            #         }
            #     ],
            #     self.lr,
            #     self.momentum,
            #     self.nesterov,
            #     self.dampening,
            # )

            # opt_dict[bkt_id] = opt(
            #     list(model_dict[bkt_id].parameters())
            #     + list(contrastive_loss[bkt_id].parameters()),
            #     lr=self.lr,
            #     amsgrad=True,
            # )

            # SophiaG optimizer
            opt_dict[bkt_id] = opt(
                list(model_dict[bkt_id].parameters())
                + list(contrastive_loss[bkt_id].parameters()),
                lr=3e-4,
                betas=(0.9, 0.95),
                rho=0.03,
            )

        return opt_dict


class DvecOptimizerUnRegUnsupervised(BuildOptimizer):
    """Build the d-vector optimizer with available checkpoints."""

    def __init__(self, device, buckets, args, unreg_bkts_storage, hparams):
        self.device = device
        self.buckets = buckets
        self.args = args
        self.unreg_bkts_storage = unreg_bkts_storage

        self.lr = hparams.lr
        self.momentum = hparams.momentum
        self.nesterov = hparams.nesterov
        self.dampening = hparams.dampening
        self.weight_decay = hparams.weight_decay

    def build_optimizer(self, model_dict, opt, contrastive_loss):
        """Build the d-vector optimizer."""

        opt_dict = {bkt_ids: [] for _, bkt_ids in enumerate(self.buckets)}
        for indx, bkt_id in enumerate(self.buckets):
            opt_dict[bkt_id] = opt(
                [
                    {
                        "params": list(model_dict[bkt_id].parameters())
                        + list(contrastive_loss[bkt_id].parameters()),
                        "weight_decay": self.weight_decay,
                    }
                ],
                self.lr,
                self.momentum,
                self.nesterov,
                self.dampening,
            )

        return opt_dict


class DvecGeneral(BuildLoadModelOptimizer):
    """Build and load the d-vector & optimizer."""

    def __init__(self, model_obj, opt_obj, opt, device, buckets, args):
        super().__init__(model_obj, opt_obj)

        self.opt = opt
        self.device = device
        self.buckets = buckets
        self.args = args

        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, model, contrastive_loss):
        """Build the encoder & optimizer."""
        model_dict = self.model_obj.build_model(model)
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, opt_dict

    def load_model_opt(self, hparams, model, contrastive_loss, filename_dict):
        """Load the encoder & optimizer from the corresponding checkpoints."""
        model_dict, start_epoch_dict = self.model_obj.load_model(
            hparams,
            model,
            filename_dict,
        )
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, opt_dict, start_epoch_dict


class DvecGeneralUnsupervised(BuildLoadModelOptimizer):
    """Build and load the d-vector & optimizer."""

    def __init__(self, model_obj, opt_obj, opt, device, buckets, args):
        super().__init__(model_obj, opt_obj)

        self.opt = opt
        self.device = device
        self.buckets = buckets
        self.args = args

        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, model, contrastive_loss):
        """Build the encoder & optimizer."""
        model_dict, cont_loss_dict = self.model_obj.build_model(model, contrastive_loss)
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, cont_loss_dict, opt_dict

    def load_model_opt(
        self,
        hparams,
        model,
        cont_loss,
        filename_dict,
    ):
        """Load the encoder & optimizer from the corresponding checkpoints."""
        model_dict, cont_loss_dict, start_epoch_dict = self.model_obj.load_model(
            hparams,
            model,
            cont_loss,
            filename_dict,
        )

        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, cont_loss_dict)

        return model_dict, cont_loss_dict, opt_dict, start_epoch_dict


class DvecGeneralDynamicReg(BuildLoadModelOptimizer):
    """Build and load the d-vector & optimizer."""

    def __init__(self, model_obj, opt_obj, opt, device, buckets, args):
        super().__init__(model_obj, opt_obj)

        self.opt = opt
        self.device = device
        self.buckets = buckets
        self.args = args

        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, model, contrastive_loss):
        """Build the encoder & optimizer."""
        model_dict = self.model_obj.build_model(model)
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, opt_dict

    def load_model_opt(
        self,
        hparams,
        model,
        contrastive_loss,
        filename_dict,
        filename_dict_reg,
    ):
        """Load the encoder & optimizer from the corresponding checkpoints."""
        model_dict, start_epoch_dict = self.model_obj.load_model(
            hparams,
            model,
            filename_dict,
            filename_dict_reg,
        )
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, opt_dict, start_epoch_dict


class DvecGeneralDynamicReReg(BuildLoadModelOptimizer):
    """Build and load the d-vector & optimizer."""

    def __init__(
        self,
        model_obj,
        opt_obj,
        opt,
        device,
        buckets,
        args,
    ):
        super().__init__(model_obj, opt_obj)

        self.opt = opt
        self.device = device
        self.buckets = buckets
        self.args = args

        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, model, contrastive_loss):
        """Build the encoder & optimizer."""
        model_dict = self.model_obj.build_model(model)
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, opt_dict

    def load_model_opt(
        self,
        hparams,
        model,
        contrastive_loss,
        filename_dict,
        filename_dict_reg,
        filename_dict_re_reg,
    ):
        """Load the encoder & optimizer from the corresponding checkpoints."""
        model_dict, start_epoch_dict = self.model_obj.load_model(
            hparams,
            model,
            filename_dict,
            filename_dict_reg,
            filename_dict_re_reg,
        )
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, opt_dict, start_epoch_dict


class DvecGeneralDynamicRegUnsupervised(BuildLoadModelOptimizer):
    """Build and load the d-vector & optimizer."""

    def __init__(self, model_obj, opt_obj, opt, device, buckets, args):
        super().__init__(model_obj, opt_obj)

        self.opt = opt
        self.device = device
        self.buckets = buckets
        self.args = args

        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, model, contrastive_loss):
        """Build the encoder & optimizer."""
        model_dict, cont_loss_dict = self.model_obj.build_model(model, contrastive_loss)
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, cont_loss_dict, opt_dict

    def load_model_opt(
        self,
        hparams,
        model,
        cont_loss,
        filename_dict,
        filename_dict_reg,
    ):
        """Load the encoder & optimizer from the corresponding checkpoints."""
        model_dict, cont_loss_dict, start_epoch_dict = self.model_obj.load_model(
            hparams,
            model,
            cont_loss,
            filename_dict,
            filename_dict_reg,
        )

        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, cont_loss_dict)

        return model_dict, cont_loss_dict, opt_dict, start_epoch_dict


class DvecGeneralDynamicReRegUnsupervised(BuildLoadModelOptimizer):
    """Build and load the d-vector & optimizer."""

    def __init__(self, model_obj, opt_obj, opt, device, buckets, args):
        super().__init__(model_obj, opt_obj)

        self.opt = opt
        self.device = device
        self.buckets = buckets
        self.args = args

        self.model_obj = model_obj
        self.opt_obj = opt_obj

    def build_model_opt(self, model, contrastive_loss):
        """Build the encoder & optimizer."""
        model_dict, cont_loss_dict = self.model_obj.build_model(model, contrastive_loss)
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, contrastive_loss)

        return model_dict, cont_loss_dict, opt_dict

    def load_model_opt(
        self,
        hparams,
        model,
        cont_loss,
        filename_dict,
        filename_dict_reg,
        filename_dict_re_reg,
    ):
        """Load the encoder & optimizer from the corresponding checkpoints."""
        model_dict, cont_loss_dict, start_epoch_dict = self.model_obj.load_model(
            hparams,
            model,
            cont_loss,
            filename_dict,
            filename_dict_reg,
            filename_dict_re_reg,
        )
        opt_dict = self.opt_obj.build_optimizer(model_dict, self.opt, cont_loss_dict)

        return model_dict, cont_loss_dict, opt_dict, start_epoch_dict
