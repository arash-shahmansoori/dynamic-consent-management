from functools import partial
from .train_selective_unreg_inductive_bias_supervised import (
    train_selective_unreg_inductive_bias_sup,
)


def train_strategy_unreg_per_epoch_per_bkt(epoch, early_stopping):

    train_selective_unreg_inductive_bias_sup_new = partial(
        train_selective_unreg_inductive_bias_sup,
        epoch,
        early_stopping,
    )

    return train_selective_unreg_inductive_bias_sup_new