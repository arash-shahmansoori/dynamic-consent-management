import numpy as np


def per_round_spks_per_bkt(
    args,
    bucket_id,
    opt_unique_bkt_sofar,
    opt_unique_bkt,
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

        num_spk_per_bkt = args.spk_per_bucket
        num_new_reg_bkt = 1

    elif strategy_keys["strategy_2"]:
        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        num_spk_selected_strategy_sofar = len(spk_selected_strategy_sofar)

        num_spk_per_bkt = args.spk_per_bucket + num_spk_selected_strategy_sofar
        num_new_reg_bkt = 0

    elif strategy_keys["strategy_3"]:

        spk_selected_strategy_sofar = np.where(
            np.array(opt_unique_bkt_sofar) == bucket_id
        )[0]

        num_spk_selected_strategy_sofar = len(spk_selected_strategy_sofar)

        num_spk_per_bkt = args.spk_per_bucket + num_spk_selected_strategy_sofar
        num_new_reg_bkt = 1

    elif strategy_keys["strategy_4"]:

        num_spk_per_bkt = args.spk_per_bucket
        num_new_reg_bkt = 0

    else:
        raise ValueError

    return num_spk_per_bkt, num_new_reg_bkt


def per_round_spks_per_bkts_storage(
    args,
    buckets,
    opt_unique_bkt_sofar,
    opt_unique_bkt,
):
    spk_per_bkt_storage = []
    spk_per_bkt_reg_storage = []
    for _, bucket_id in enumerate(buckets):
        num_spk_per_bkt, num_new_reg_bkt = per_round_spks_per_bkt(
            args,
            bucket_id,
            opt_unique_bkt_sofar,
            opt_unique_bkt,
        )

        spk_per_bkt_storage.append(num_spk_per_bkt)
        spk_per_bkt_reg_storage.append(num_new_reg_bkt)

    return spk_per_bkt_storage, spk_per_bkt_reg_storage
