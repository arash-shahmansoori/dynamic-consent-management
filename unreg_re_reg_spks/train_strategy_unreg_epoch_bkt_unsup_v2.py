from functools import partial
from .train_selective_unreg_inductive_bias_unsupervised_v2 import (
    train_selective_unreg_inductive_bias_unsup_v2,
)


def train_strategy_unreg_per_epoch_per_bkt_unsup_v2(
    epoch,
    early_stopping,
    agent_method,
    hparams,
):

    train_selective_unreg_inductive_bias_unsup_new_v2 = partial(
        train_selective_unreg_inductive_bias_unsup_v2,
        epoch,
        early_stopping,
        agent_method,
        hparams,
    )

    return train_selective_unreg_inductive_bias_unsup_new_v2