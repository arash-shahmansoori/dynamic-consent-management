from .strategy_old_unreg_epoch_bkt_unsup import (
    strategy_old_unreg_per_epoch_per_bkt_unsup,
)
from .train_strategy_unreg_epoch_bkt_unsup import (
    train_strategy_unreg_per_epoch_per_bkt_unsup,
)


def train_unreg_per_epoch_per_bkt_unsup(
    hparams,
    args,
    device,
    updated_outputs,
    outputs,
    bucket_id,
    epoch,
    early_stopping_status,
    agent_method,
    **kwargs_training,
):

    strategy_keys = {
        "strategy_1": len(updated_outputs[bucket_id]) == len(outputs[bucket_id]),
        "strategy_2": len(updated_outputs[bucket_id]) != len(outputs[bucket_id]),
        "strategy_3": (len(updated_outputs[bucket_id]) == 0)
        and (len(outputs[bucket_id]) != 0),
    }

    if strategy_keys["strategy_1"]:
        train_strategy = strategy_old_unreg_per_epoch_per_bkt_unsup(
            early_stopping_status
        )
    elif strategy_keys["strategy_2"]:
        train_strategy = train_strategy_unreg_per_epoch_per_bkt_unsup(
            epoch,
            kwargs_training["early_stop"][bucket_id],
            agent_method,
            hparams,
        )
    elif strategy_keys["strategy_3"]:
        train_strategy = train_strategy_unreg_per_epoch_per_bkt_unsup(
            epoch,
            kwargs_training["early_stop"][bucket_id],
            agent_method,
            hparams,
        )
    else:
        raise ValueError

    sup_unreg = train_strategy(
        args,
        updated_outputs,
        bucket_id,
        device,
        kwargs_training,
    )

    x = sup_unreg["x"]
    spk = sup_unreg["y"]

    xe = kwargs_training["dvectors"][bucket_id](x).detach()

    feat_props = {
        "feat_bkt": xe.view(-1, args.dim_emb),
        "label_bkt": spk.view(-1),
    }

    return feat_props
