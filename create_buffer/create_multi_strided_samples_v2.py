import torch
import random
import numpy as np


class CreateMultiStridedSamplesV2:
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

    def inter_bucket_sample_v2(
        self,
        bucket,
        bkt_indices,
        bkt_samples,
        bkt_labels,
        device,
        permute_samples=True,
    ):
        if permute_samples:
            bkt_indices_progressive = [
                random.sample(
                    bkt_indices[str(bucket)][k], len(bkt_indices[str(bucket)][k])
                )
                for k in range(bucket + 1)
            ]
        else:
            bkt_indices_progressive = [
                bkt_indices[str(bucket)][k] for k in range(bucket + 1)
            ]

        bkt_samples_progressive = [bkt_samples[str(n)] for n in range(bucket + 1)]
        bkt_labels_progressive = [bkt_labels[str(n)] for n in range(bucket + 1)]

        samples_progressive = [
            torch.cat([bkt_samples_progressive[inds][k] for k in s], dim=0).view(
                -1, self.args.dim_emb
            )
            for inds, s in enumerate(bkt_indices_progressive)
        ]
        labels_progressive = [
            [bkt_labels_progressive[inds][k] for k in s]
            for inds, s in enumerate(bkt_indices_progressive)
        ]

        stacked_feats = (
            torch.cat(samples_progressive, dim=0)
            .view(
                -1,
                self.args.dim_emb,
            )
            .to(device)
        )
        stacked_labels = torch.tensor(labels_progressive).view(-1).to(device)

        return stacked_feats, stacked_labels

    def create_progressive_collect_indx(self, args, buckets):
        bkt_samples, bkt_labels = {}, {}
        lf_collect_progress = {}
        for indx, _ in enumerate(buckets):
            spk_per_bkt_storage = (indx + 1) * [args.spk_per_bucket]
            spk_per_bkt_reg_storage = (indx + 1) * [0]

            utts_per_spk = self.num_per_spk_utts_progressive_mem(
                spk_per_bkt_storage,
                spk_per_bkt_reg_storage,
            )
            lf_collect = self.utt_index_per_bucket_collection(
                spk_per_bkt_storage,
                spk_per_bkt_reg_storage,
                utts_per_spk,
            )

            lf_collect_progress[str(indx)] = [
                lf_collect[i] for i in range(len(lf_collect))
            ]

        return lf_collect_progress, bkt_samples, bkt_labels

    def create_collect_indx(
        self,
        args,
        buckets,
        spk_per_bkt_storage=None,
        spk_per_bkt_reg_storage=None,
    ):
        feats_init, labels_init = [], []
        if spk_per_bkt_storage == None and spk_per_bkt_reg_storage == None:
            spk_per_bkt_storage = len(buckets) * [args.spk_per_bucket]
            spk_per_bkt_reg_storage = len(buckets) * [0]

        utts_per_spk = self.num_per_spk_utts_progressive_mem(
            spk_per_bkt_storage,
            spk_per_bkt_reg_storage,
        )
        lf_collection = self.utt_index_per_bucket_collection(
            spk_per_bkt_storage,
            spk_per_bkt_reg_storage,
            utts_per_spk,
        )

        return lf_collection, feats_init, labels_init
