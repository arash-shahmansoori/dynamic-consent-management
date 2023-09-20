from functools import partial
from .unreg_supervised_old import unreg_sup_old


def strategy_old_unreg_per_epoch_per_bkt(early_stopping_status):

    unreg_sup_old_partial = partial(unreg_sup_old, early_stopping_status)

    return unreg_sup_old_partial