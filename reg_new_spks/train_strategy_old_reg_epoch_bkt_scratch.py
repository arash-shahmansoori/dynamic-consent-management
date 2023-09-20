from functools import partial
from .train_reg_supervised_old_scratch import (
    train_selective_reg_inductive_bias_old_sup_scratch,
)


def train_strategy_old_reg_per_epoch_per_bkt_scratch(epoch, early_stopping_status):

    reg_sup_old_partial_scratch = partial(
        train_selective_reg_inductive_bias_old_sup_scratch,
        epoch,
        early_stopping_status,
    )

    return reg_sup_old_partial_scratch