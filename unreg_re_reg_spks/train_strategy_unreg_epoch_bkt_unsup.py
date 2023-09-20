from functools import partial
from .train_selective_unreg_inductive_bias_unsupervised import (
    train_selective_unreg_inductive_bias_unsup,
)


def train_strategy_unreg_per_epoch_per_bkt_unsup(
    epoch,
    early_stopping,
    agent_method,
    hparams,
    wittness_outputs,
):

    train_selective_unreg_inductive_bias_unsup_new = partial(
        train_selective_unreg_inductive_bias_unsup,
        epoch,
        early_stopping,
        agent_method,
        hparams,
        wittness_outputs,
    )

    return train_selective_unreg_inductive_bias_unsup_new