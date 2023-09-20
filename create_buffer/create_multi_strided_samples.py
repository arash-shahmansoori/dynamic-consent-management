import torch
import random
import numpy as np


class CreateMultiStridedSamples:
    """
    Create multi-strided samples for the training.
    """

    def __init__(self, args):

        self.args = args

    def num_per_spk_utts_progressive_mem(
        self,
        spk_per_bucket_storage,
        spk_per_bucket_reg_storage,
    ):
        total_spks_per_bkts_storage = 0
        for spk_per_bucket, spk_per_bucket_reg in zip(
            spk_per_bucket_storage, spk_per_bucket_reg_storage
        ):
            total_spks_per_bkts_storage += spk_per_bucket + spk_per_bucket_reg

        utts_per_spk = torch.floor(
            torch.tensor((self.args.max_mem / (total_spks_per_bkts_storage)))
        )
        return int(utts_per_spk)

    def num_per_spk_utts_per_epoch_mem(
        self,
        spk_per_bucket,
        spk_per_bucket_reg,
        num_buckets,
    ):
        utts_per_spk = torch.floor(
            torch.tensor(
                (
                    self.args.max_mem
                    / (num_buckets * (spk_per_bucket + spk_per_bucket_reg))
                )
            )
        )
        return int(utts_per_spk)

    def utt_index_per_bucket(
        self,
        spk_per_bucket,
        spk_per_bucket_reg,
        num_utts,
        prng=None,
    ):
        # Randomly selects "num_utts" utterances per speaker per bucket.

        prng = prng if prng else np.random

        total_spk_per_bucket = spk_per_bucket + spk_per_bucket_reg

        l = [
            (
                torch.from_numpy(
                    prng.choice(
                        range(
                            self.args.n_utterances_labeled * i,
                            self.args.n_utterances_labeled * (i + 1),
                        ),
                        num_utts,
                        replace=True,
                    )
                ).int()
            ).tolist()
            for i in range(total_spk_per_bucket)
        ]
        lf = [u for s in l for u in s]
        return lf

    def utt_index_per_bucket_collection(
        self,
        spk_per_bucket_storage,
        spk_per_bucket_reg_storage,
        num_utts,
        prng=None,
    ):
        # Randomly selects "num_utts" utterances per speaker per bucket.

        prng = prng if prng else np.random

        lf_collection = []
        for spk_per_bucket, spk_per_bucket_reg in zip(
            spk_per_bucket_storage, spk_per_bucket_reg_storage
        ):
            total_spk_per_bucket = spk_per_bucket + spk_per_bucket_reg

            l = [
                (
                    torch.from_numpy(
                        prng.choice(
                            range(
                                self.args.n_utterances_labeled * i,
                                self.args.n_utterances_labeled * (i + 1),
                            ),
                            num_utts,
                            replace=True,
                        )
                    ).int()
                ).tolist()
                for i in range(total_spk_per_bucket)
            ]
            lf = [u for s in l for u in s]

            lf_collection.append(lf)

        return lf_collection

    def inter_bucket_sample(
        self,
        per_bkt_indices,
        per_bkt_samples,
        per_bkt_labels,
        feats_init,
        labels_init,
        permute_samples=True,
    ):
        # Inter bucket sampling
        if permute_samples:
            # if len(per_bkt_indices) <= len(per_bkt_samples):
            #     perm = random.sample(per_bkt_indices, len(per_bkt_indices))
            # else:
            #     perm = random.sample(per_bkt_samples, len(per_bkt_samples))
            # per_bkt_indices_selected = perm

            perm = random.sample(per_bkt_indices, len(per_bkt_indices))
            per_bkt_indices_selected = perm

        else:
            # if len(per_bkt_indices) <= len(per_bkt_samples):
            #     per_bkt_indices_selected = per_bkt_indices
            # else:
            #     per_bkt_indices_selected = per_bkt_samples

            per_bkt_indices_selected = per_bkt_indices

        feats_init.append(
            per_bkt_samples[per_bkt_indices_selected, :].view(-1, self.args.dim_emb)
        )
        labels_init.append(per_bkt_labels[per_bkt_indices_selected].view(-1))

        stacked_feats = torch.cat(feats_init, dim=0).view(-1, self.args.dim_emb)
        stacked_labels = torch.cat(labels_init, dim=0).view(-1)

        return stacked_feats, stacked_labels
