import torch
import numpy as np

from utils import (
    label_normalizer_per_bucket,
    # progressive_indx_normalization,
    # Progressive_normalized_label,
)
from .train_strategy_new_reg_epoch_bkt_unsup import (
    train_strategy_new_reg_per_epoch_per_bkt_unsup,
)
from .train_strategy_new_reg_causal_epoch_bkt_unsup import (
    train_strategy_new_reg_causal_per_epoch_per_bkt_unsup,
)
from .train_strategy_sofar_reg_causal_epoch_bkt_unsup import (
    train_strategy_sofar_reg_causal_per_epoch_per_bkt_unsup,
)
from .strategy_old_reg_epoch_bkt_unsup import strategy_old_reg_per_epoch_per_bkt_unsup


def train_reg_per_round_per_epoch_per_bkt_unsup(
    hparams,
    args,
    device,
    outputs,
    bucket_id,
    opt_unique_bkt_sofar,
    indx_opt_unique_bkt_sofar,
    opt_unique_bkt,
    indx_opt_unique_bkt,
    epoch,
    early_stopping_status,
    agent_method,
    **kwargs_training,
):

    strategy_keys = {
        "strategy_1": (bucket_id in opt_unique_bkt)
        and (bucket_id not in opt_unique_bkt_sofar),
        "strategy_2": (bucket_id not in opt_unique_bkt)
        and (bucket_id in opt_unique_bkt_sofar),
        "strategy_3": (bucket_id in opt_unique_bkt)
        and (bucket_id in opt_unique_bkt_sofar),
        "strategy_4": (bucket_id not in opt_unique_bkt)
        and (bucket_id not in opt_unique_bkt_sofar),
    }

    # Select the strategy per round per registration
    if strategy_keys["strategy_1"]:
        spk_selected_strategy = opt_unique_bkt.index(bucket_id)

        num_spk_per_bkt = args.spk_per_bucket
        num_new_reg_bkt = 1

        train_strategy = train_strategy_new_reg_per_epoch_per_bkt_unsup(
            epoch,
            indx_opt_unique_bkt,
            spk_selected_strategy,
            kwargs_training["early_stop"][bucket_id],
        )

    elif strategy_keys["strategy_2"]:
        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        num_spk_selected_strategy_sofar = len(spk_selected_strategy_sofar)

        num_spk_per_bkt = args.spk_per_bucket + num_spk_selected_strategy_sofar
        num_new_reg_bkt = 0

        train_strategy = train_strategy_sofar_reg_causal_per_epoch_per_bkt_unsup(
            epoch,
            indx_opt_unique_bkt_sofar,
            spk_selected_strategy_sofar,
            kwargs_training["early_stop"][bucket_id],
        )

    elif strategy_keys["strategy_3"]:
        spk_selected_strategy = opt_unique_bkt.index(bucket_id)

        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        num_spk_selected_strategy_sofar = len(spk_selected_strategy_sofar)

        num_spk_per_bkt = args.spk_per_bucket + num_spk_selected_strategy_sofar
        num_new_reg_bkt = 1

        train_strategy = train_strategy_new_reg_causal_per_epoch_per_bkt_unsup(
            epoch,
            indx_opt_unique_bkt_sofar,
            spk_selected_strategy_sofar,
            indx_opt_unique_bkt,
            spk_selected_strategy,
            kwargs_training["early_stop"][bucket_id],
        )

    elif strategy_keys["strategy_4"]:
        spk_selected_strategy = -1

        num_spk_per_bkt = args.spk_per_bucket
        num_new_reg_bkt = 0

        train_strategy = strategy_old_reg_per_epoch_per_bkt_unsup(early_stopping_status)
    else:
        raise ValueError

    total_num_spk_per_bkt = num_spk_per_bkt + num_new_reg_bkt

    sup_reg = train_strategy(
        hparams,
        args,
        outputs,
        bucket_id,
        device,
        agent_method,
        total_num_spk_per_bkt,
        kwargs_training,
    )

    x = sup_reg["x"]
    spk = sup_reg["y"]

    xe = kwargs_training["dvectors"][bucket_id](x).detach()

    # Label normalization
    # spk_normalized = progressive_indx_normalization(
    #     spk.cpu(),
    #     args.spk_per_bucket,
    #     bucket_id,
    #     device,
    # )

    # spk_normalized_concat = Progressive_normalized_label(
    #     spk_normalized,
    #     bucket_id,
    # )

    spk_normalized = label_normalizer_per_bucket(spk)

    feat_props = {
        "feat_bkt": xe.view(-1, args.dim_emb),
        "label_bkt": spk_normalized.view(-1),
        "num_spk_per_bkt": num_spk_per_bkt,
        "num_new_reg_bkt": num_new_reg_bkt,
    }

    return feat_props
