from functools import partial
from .train_selective_re_reg_inductive_bias_unsupervised import (
    train_selective_re_reg_inductive_bias_unsup,
)


def train_strategy_re_reg_per_epoch_per_bkt_unsup(
    epoch,
    early_stopping,
    agent_method,
    hparams,
):

    train_selective_re_reg_inductive_bias_unsup_new = partial(
        train_selective_re_reg_inductive_bias_unsup,
        epoch,
        early_stopping,
        agent_method,
        hparams,
    )

    return train_selective_re_reg_inductive_bias_unsup_new