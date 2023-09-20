import torch

from torch.optim import SGD

from torch.utils.data import DataLoader
from preprocess_data import SubDatasetGdrSpk, collateGdrSpkr
from utils import compute_prototypes, pairwise_distances, SupConLoss


def train_dvector(model_dvector, bucket_id, x, y, device, args):
    criterion = SupConLoss(args).to(device)

    optimizer = SGD(
        [
            {
                "params": list(model_dvector[bucket_id].parameters())
                + list(criterion.parameters()),
                "weight_decay": 1e-4,
            }
        ],
        lr=1e-2,
        momentum=0.9,
        nesterov=True,
        dampening=0,
    )

    model_dvector[bucket_id].train()

    for _ in range(args.epochs_per_dvector):

        optimizer.zero_grad()

        output = model_dvector[bucket_id](x).view(-1, 1, args.dim_emb)

        loss = criterion(output, y)
        loss.backward()

        optimizer.step()


def compute_opt_bkt_final_v2(
    dvectors,
    bucket_id_chosen,
    round_num,
    number_of_bucket,
    outputs,
    removed_indices_flattened,
    opt_buckets_flattened,
    dataset,
    dataset_prev_other,
    dataset_other,
    device,
    args,
):
    distances_dict = {bucket_id: [] for bucket_id in range(number_of_bucket)}
    n_x_dic = {bucket_id: [] for bucket_id in range(number_of_bucket)}
    n_y_dic = {bucket_id: [] for bucket_id in range(number_of_bucket)}

    for bucket_id in range(number_of_bucket):

        sub_lbs_current = outputs[bucket_id]

        sub_lbs_current_other = [bucket_id_chosen]

        sub_dataset_current = SubDatasetGdrSpk(dataset, sub_lbs_current)

        sub_dataset_current_other = SubDatasetGdrSpk(
            dataset_other, sub_lbs_current_other
        )

        test_loader_current = DataLoader(
            sub_dataset_current,
            batch_size=len(sub_lbs_current),
            collate_fn=collateGdrSpkr,
            drop_last=True,
            pin_memory=True,
        )

        test_loader_current_other = DataLoader(
            sub_dataset_current_other,
            batch_size=len(sub_lbs_current_other),
            collate_fn=collateGdrSpkr,
            drop_last=True,
            pin_memory=True,
        )

        mel_db_batch = next(iter(test_loader_current))
        x, gdr, spk = mel_db_batch
        x, gdr, spk = x.to(device), gdr.to(device), spk.to(device)

        mel_db_batch_other = next(iter(test_loader_current_other))
        x_other, gdr_other, spk_other = mel_db_batch_other
        x_other, gdr_other, spk_other = (
            x_other.to(device),
            gdr_other.to(device),
            spk_other.to(device),
        )

        if round_num > 0:

            sub_lbs_previous_other = []
            for id_indx, bkt_id in enumerate(opt_buckets_flattened):
                if bkt_id == bucket_id:
                    sub_lbs_previous_other.append(removed_indices_flattened[id_indx])

            if sub_lbs_previous_other:

                sub_dataset_previous_other = SubDatasetGdrSpk(
                    dataset_prev_other, sub_lbs_previous_other
                )
                test_loader_previous_other = DataLoader(
                    sub_dataset_previous_other,
                    batch_size=len(sub_lbs_previous_other),
                    collate_fn=collateGdrSpkr,
                    drop_last=True,
                    pin_memory=True,
                )
                mel_db_batch_prev_other = next(iter(test_loader_previous_other))
                x_prev_other, gdr_prev_other, spk_prev_other = mel_db_batch_prev_other
                x_prev_other, gdr_prev_other, spk_prev_other = (
                    x_prev_other.to(device),
                    gdr_prev_other.to(device),
                    spk_prev_other.to(device),
                )

                x_cat = torch.cat((x, x_prev_other, x_other), dim=0)
                y_cat = torch.cat((spk, spk_prev_other, spk_other), dim=0)

                train_dvector(dvectors, bucket_id, x_cat, y_cat, device, args)

                x_ = torch.cat((x, x_prev_other), dim=0)
                y_ = torch.cat((spk, spk_prev_other), dim=0)

                emb_spk = dvectors[bucket_id](x_)
                # Compute pairwise distance
                prototypes = compute_prototypes(
                    emb_spk,
                    args.spk_per_bucket + len(sub_lbs_previous_other),
                    args.nt_utterances_labeled,
                )
                emb_spk_other = dvectors[bucket_id](x_other)

                distances = pairwise_distances(emb_spk_other, prototypes, "l2")

                distances_dict[bucket_id].append(distances.tolist())
                n_x_dic[bucket_id].append(emb_spk_other.shape[0])
                n_y_dic[bucket_id].append(prototypes.shape[0])
        elif round_num == 0:

            x_cat = torch.cat((x, x_other), dim=0)
            y_cat = torch.cat((spk, spk_other), dim=0)

            train_dvector(dvectors, bucket_id, x_cat, y_cat, device, args)

            emb_spk = dvectors[bucket_id](x)
            # Compute pairwise distance
            prototypes = compute_prototypes(
                emb_spk,
                args.spk_per_bucket,
                args.nt_utterances_labeled,
            )
            emb_spk_other = dvectors[bucket_id](x_other)

            distances = pairwise_distances(emb_spk_other, prototypes, "l2")

            distances_dict[bucket_id].append(distances.tolist())
            n_x_dic[bucket_id].append(emb_spk_other.shape[0])
            n_y_dic[bucket_id].append(prototypes.shape[0])
        else:
            raise ValueError

    return distances_dict, n_x_dic, n_y_dic
