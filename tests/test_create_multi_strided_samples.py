import torch


import pytest

from numpy.random import RandomState
from .multi_stride_sampler import TestCreateMultiStridedSamples


# Arguments required for tests
spk_per_bucket = 5

max_mem = 120
dim_emb = 256
n_utterances_labeled = 20


prng = RandomState(0)


@pytest.fixture
def create_multi_strided_samples():
    return TestCreateMultiStridedSamples(max_mem, dim_emb, n_utterances_labeled)


def test_num_per_spk_utts_progressive_mem_valid(
    create_multi_strided_samples: TestCreateMultiStridedSamples,
):
    utts_per_spk = create_multi_strided_samples.num_per_spk_utts_progressive_mem(
        [5, 6, 5],
        [0, 1, 0],
    )
    true_utts_per_spk = int(torch.floor(torch.tensor((max_mem / 17))))
    assert utts_per_spk == true_utts_per_spk


def test_num_per_spk_utts_progressive_mem_invalid(
    create_multi_strided_samples: TestCreateMultiStridedSamples,
):
    with pytest.raises(ZeroDivisionError):
        _ = create_multi_strided_samples.num_per_spk_utts_progressive_mem(
            [],
            [],
        )


# def test_num_per_spk_utts_per_epoch_mem_valid(
#     create_multi_strided_samples: TestCreateMultiStridedSamples,
# ):
#     utts_per_spk = create_multi_strided_samples.num_per_spk_utts_per_epoch_mem(
#         5,
#         0,
#         1,
#     )
#     true_utts_per_spk = torch.floor(torch.tensor((max_mem / (1 * (5 + 0)))))
#     assert utts_per_spk == true_utts_per_spk


# def test_num_per_spk_utts_per_epoch_mem_invalid(
#     create_multi_strided_samples: TestCreateMultiStridedSamples,
# ):
#     with pytest.raises(ZeroDivisionError):
#         _ = create_multi_strided_samples.num_per_spk_utts_per_epoch_mem(0, 0, 3)


def test_utt_index_per_bucket(
    create_multi_strided_samples: TestCreateMultiStridedSamples,
):
    lf = create_multi_strided_samples.utt_index_per_bucket(
        5,
        0,
        5,
        prng,
    )

    total_spk_per_bucket = 5 + 0
    l = [
        (
            torch.from_numpy(
                prng.choice(
                    range(
                        n_utterances_labeled * i,
                        n_utterances_labeled * (i + 1),
                    ),
                    5,
                    replace=True,
                )
            ).int()
        ).tolist()
        for i in range(total_spk_per_bucket)
    ]
    lf_true = [u for s in l for u in s]
    assert all(lf) == all(lf_true)


def test_utt_index_per_bucket_collection(
    create_multi_strided_samples: TestCreateMultiStridedSamples,
):
    lf_collection = create_multi_strided_samples.utt_index_per_bucket_collection(
        [5, 6],
        [0, 1],
        5,
        prng,
    )

    total_spk_per_bucket_0 = 5 + 0
    l_0 = [
        (
            torch.from_numpy(
                prng.choice(
                    range(
                        n_utterances_labeled * i,
                        n_utterances_labeled * (i + 1),
                    ),
                    5,
                    replace=True,
                )
            ).int()
        ).tolist()
        for i in range(total_spk_per_bucket_0)
    ]
    lf_true_0 = [u for s in l_0 for u in s]

    total_spk_per_bucket_1 = 6 + 1
    l_1 = [
        (
            torch.from_numpy(
                prng.choice(
                    range(
                        n_utterances_labeled * i,
                        n_utterances_labeled * (i + 1),
                    ),
                    5,
                    replace=True,
                )
            ).int()
        ).tolist()
        for i in range(total_spk_per_bucket_1)
    ]
    lf_true_1 = [u for s in l_1 for u in s]

    lf_true_collection = [lf_true_0, lf_true_1]
    lf_true_total = [u for s in lf_true_collection for u in s]

    lf_total = [u for s in lf_collection for u in s]

    assert all(lf_total) == all(lf_true_total)


# def test_inter_bucket_sample_single_bucket(
#     create_multi_strided_samples: TestCreateMultiStridedSamples,
# ):

#     utts_per_spk = create_multi_strided_samples.num_per_spk_utts_per_epoch_mem(
#         5,
#         0,
#         1,
#     )
#     lf = create_multi_strided_samples.utt_index_per_bucket(
#         5,
#         0,
#         utts_per_spk,
#     )
#     total_spks_per_bucket = 5 + 0
#     samples = torch.zeros(
#         (
#             total_spks_per_bucket * utts_per_spk * 1,
#             dim_emb,
#         )
#     )
#     labels = torch.tensor([[i] * utts_per_spk for i in range(spk_per_bucket)]).view(-1)

#     stacked_feats, stacked_labels = create_multi_strided_samples.inter_bucket_sample(
#         lf,
#         samples,
#         labels,
#         [],
#         [],
#         permute_samples=False,
#     )

#     assert all(stacked_labels.tolist()) == all(labels.tolist()) and all(
#         stacked_feats.tolist()
#     ) == all(samples.tolist())


# def test_inter_bucket_sample_multi_buckets_per_epoch_mem(
#     create_multi_strided_samples: TestCreateMultiStridedSamples,
# ):
#     spk_per_bkt = 5
#     spk_per_bkt_reg = 0
#     num_buckets = 1

#     utts_per_spk = create_multi_strided_samples.num_per_spk_utts_per_epoch_mem(
#         spk_per_bkt, spk_per_bkt_reg, num_buckets
#     )
#     lf = create_multi_strided_samples.utt_index_per_bucket(
#         spk_per_bkt, spk_per_bkt_reg, utts_per_spk
#     )

#     samples = torch.zeros(
#         ((spk_per_bkt + spk_per_bkt_reg) * utts_per_spk * num_buckets, dim_emb)
#     )
#     labels = torch.tensor(
#         [[i] * utts_per_spk for i in range(spk_per_bkt * num_buckets)]
#     ).view(-1)

#     spks_per_bkts_sofar = 0
#     feats_init, labels_init = [], []
#     for indx in range(num_buckets):
#         samples_bkt = torch.zeros(
#             ((spk_per_bkt + spk_per_bkt_reg) * utts_per_spk, dim_emb)
#         )
#         if indx > 0:
#             spks_per_bkts_sofar += spk_per_bkt
#             labels_bkt = torch.tensor(
#                 [
#                     [i + spks_per_bkts_sofar] * utts_per_spk
#                     for i in range(spk_per_bkt + spk_per_bkt_reg)
#                 ]
#             ).view(-1)
#         else:
#             labels_bkt = torch.tensor(
#                 [[i] * utts_per_spk for i in range(spk_per_bkt + spk_per_bkt_reg)]
#             ).view(-1)

#         (
#             stacked_feats,
#             stacked_labels,
#         ) = create_multi_strided_samples.inter_bucket_sample(
#             lf,
#             samples_bkt,
#             labels_bkt,
#             feats_init,
#             labels_init,
#             permute_samples=False,
#         )

#     assert all(stacked_labels.tolist()) == all(labels.tolist())


def test_inter_bucket_sample_multi_buckets_progressive_mem_static_no_permute(
    create_multi_strided_samples: TestCreateMultiStridedSamples,
):
    spk_per_bkt_storage = [5, 5]
    spk_per_bkt_reg_storage = [0, 0]

    utts_per_spk = create_multi_strided_samples.num_per_spk_utts_progressive_mem(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
    )
    lf_collection = create_multi_strided_samples.utt_index_per_bucket_collection(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
        utts_per_spk,
        prng,
    )

    spks_per_bkts_sofar = 0
    labels_bnchmark, samples_bnchmark = [], []
    feats_init, labels_init = [], []
    for indx in range(len(spk_per_bkt_storage)):

        samples_bkt = torch.zeros(
            (
                (spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx])
                * n_utterances_labeled,
                dim_emb,
            )
        )

        samples = torch.zeros(
            (
                (spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx])
                * utts_per_spk,
                dim_emb,
            )
        )

        if indx > 0:
            spks_per_bkts_sofar += spk_per_bkt_storage[indx - 1]

            labels = torch.tensor(
                [
                    [i + spks_per_bkts_sofar] * utts_per_spk
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)

            labels_bkt = torch.tensor(
                [
                    [i + spks_per_bkts_sofar] * n_utterances_labeled
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)
        else:
            labels = torch.tensor(
                [
                    [i] * utts_per_spk
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)

            labels_bkt = torch.tensor(
                [
                    [i] * n_utterances_labeled
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)

        labels_bnchmark.append(labels)
        samples_bnchmark.append(samples)

        (
            stacked_feats,
            stacked_labels,
        ) = create_multi_strided_samples.inter_bucket_sample(
            lf_collection[indx],
            samples_bkt,
            labels_bkt,
            feats_init,
            labels_init,
            permute_samples=False,
        )

    assert all(stacked_labels.tolist()) == all(
        torch.cat(labels_bnchmark, dim=0).view(-1).tolist()
    ) and all(stacked_feats.view(-1).tolist()) == all(
        torch.cat(samples_bnchmark, dim=0).view(-1).tolist()
    )


def test_inter_bucket_sample_multi_buckets_progressive_mem_dynamic_no_permute(
    create_multi_strided_samples: TestCreateMultiStridedSamples,
):
    spk_per_bkt_storage = [7, 7]
    spk_per_bkt_reg_storage = [0, 1]

    utts_per_spk = create_multi_strided_samples.num_per_spk_utts_progressive_mem(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
    )
    lf_collection = create_multi_strided_samples.utt_index_per_bucket_collection(
        spk_per_bkt_storage,
        spk_per_bkt_reg_storage,
        utts_per_spk,
        prng,
    )

    spks_per_bkts_sofar = 0
    labels_bnchmark, samples_bnchmark = [], []
    feats_init, labels_init = [], []
    for indx in range(len(spk_per_bkt_storage)):

        samples_bkt = torch.zeros(
            (
                (spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx])
                * n_utterances_labeled,
                dim_emb,
            )
        )

        samples = torch.zeros(
            (
                (spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx])
                * utts_per_spk,
                dim_emb,
            )
        )

        if indx > 0:
            spks_per_bkts_sofar += spk_per_bkt_storage[indx - 1]

            labels = torch.tensor(
                [
                    [i + spks_per_bkts_sofar] * utts_per_spk
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)

            labels_bkt = torch.tensor(
                [
                    [i + spks_per_bkts_sofar] * n_utterances_labeled
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)
        else:
            labels = torch.tensor(
                [
                    [i] * utts_per_spk
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)

            labels_bkt = torch.tensor(
                [
                    [i] * n_utterances_labeled
                    for i in range(
                        spk_per_bkt_storage[indx] + spk_per_bkt_reg_storage[indx]
                    )
                ]
            ).view(-1)

        labels_bnchmark.append(labels)
        samples_bnchmark.append(samples)

        (
            stacked_feats,
            stacked_labels,
        ) = create_multi_strided_samples.inter_bucket_sample(
            lf_collection[indx],
            samples_bkt,
            labels_bkt,
            feats_init,
            labels_init,
            permute_samples=False,
        )

    assert all(stacked_labels.tolist()) == all(
        torch.cat(labels_bnchmark, dim=0).view(-1).tolist()
    ) and all(stacked_feats.view(-1).tolist()) == all(
        torch.cat(samples_bnchmark, dim=0).view(-1).tolist()
    )
