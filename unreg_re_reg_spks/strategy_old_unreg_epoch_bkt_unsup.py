from functools import partial
from .unreg_unsupervised_old import unreg_unsup_old


def strategy_old_unreg_per_epoch_per_bkt_unsup(early_stopping_status):

    unreg_unsup_old_partial = partial(unreg_unsup_old, early_stopping_status)

    return unreg_unsup_old_partial