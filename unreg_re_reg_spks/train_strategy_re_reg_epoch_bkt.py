from functools import partial
from .train_selective_re_reg_inductive_bias_supervised import (
    train_selective_re_reg_inductive_bias_sup,
)


def train_strategy_re_reg_per_epoch_per_bkt(epoch, early_stopping):

    train_selective_re_reg_inductive_bias_sup_new = partial(
        train_selective_re_reg_inductive_bias_sup,
        epoch,
        early_stopping,
    )

    return train_selective_re_reg_inductive_bias_sup_new