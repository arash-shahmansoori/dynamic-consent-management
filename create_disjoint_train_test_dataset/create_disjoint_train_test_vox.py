import os
import json
import torch
import librosa
import numpy as np
import itertools

from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from torch.utils.data import DataLoader


from .create_custom_vox_celeb_dataset import VoxCeleb1IdentificationCustom


def extract_features(audio, args):
    """Returns a np.array with size (args.feature_dim,'n') where n is the number of audio frames."""

    # yt, _ = librosa.effects.trim(audio, top_db=args.top_db)
    # yt, _ = librosa.effects.trim(audio)
    yt = audio

    yt = normalize(yt)
    ws = int(args.sample_rate * 0.001 * args.window_size)
    st = int(args.sample_rate * 0.001 * args.stride)

    if args.feature == "fbank":
        feat = librosa.feature.melspectrogram(
            y=audio,
            sr=args.sample_rate,
            n_mels=args.feature_dim,
            n_fft=ws,
            hop_length=st,
        )
        feat = np.log(feat + 1e-6)
    elif args.feature == "mfcc":
        feat = librosa.feature.mfcc(
            y=audio, sr=args.sample_rate, n_mfcc=args.feature_dim
        )
    else:
        raise ValueError("Unsupported Acoustic Feature: " + args.feature)

    feat = [feat]
    if args.delta:
        feat.append(librosa.feature.delta(feat[0]))
    if args.delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    return feat


def normalize(yt):
    yt = np.array(yt)

    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt


def create_file_path_list(file_id):
    fid = file_id[0].split("-")

    # if len(fid) == 3:
    #     fid_n = fid
    # elif len(fid) == 4:
    #     fid_n = [fid[0], fid[1] + "-" + fid[2], fid[-1]]
    # elif len(fid) == 5:
    #     fid_n = [fid[0], fid[1] + "-" + fid[2] + "-" + fid[3], fid[-1]]
    # elif len(fid) == 6:
    #     fid_n = [fid[0], fid[1] + "-" + fid[2] + "-" + fid[3] + "-" + fid[4], fid[-1]]
    # else:
    #     raise ValueError

    fid_m = []
    if len(fid) > 3:
        for i, _ in enumerate(fid):
            if i > 0 and i <= (len(fid) - 3):
                fid_m.append(fid[i] + "-")
            fid_m.append(fid[len(fid) - 2])
        fid_n = [fid[0], fid_m, fid[-1]]
    elif len(fid) == 3:
        fid_n = fid
    else:
        raise ValueError

    return fid_n


# def subset_logics(utts_counts, logic_name):
#     """Limit the number of old utterances as follows.

#     Args:
#         - utts_counts: the number of utterances in different folders of dataset.
#         - logic_name: dev/train.

#     Returns:
#         subset_name: logic for dev/train.

#     """

#     set_logics = {
#         "train": utts_counts > 1,  # for full use of old training utterances
#         "test": utts_counts <= 1,  # for use of testing/evaluation utterances
#         "dev": utts_counts <= 1,  # for use of testing/evaluation utterances
#     }

#     return set_logics[logic_name]


def commpute_utt_id_per_spk(args, data_loader):
    utt_id_spk = {
        str(i): []
        for i in range(
            args.agnt_num * args.n_speakers + 1,
            (args.agnt_num + 1) * args.n_speakers + 1,
        )
    }
    for _, _, spk_id, file_id in tqdm(data_loader):
        fid_n = create_file_path_list(file_id)
        speaker_name = str(spk_id[0].tolist())

        if (
            int(speaker_name) <= (args.agnt_num + 1) * args.n_speakers
            and int(speaker_name) >= args.agnt_num * args.n_speakers + 1
        ):
            utt_id = int(fid_n[-1])
            utt_id_spk[speaker_name].append(utt_id)
        elif int(speaker_name) <= args.agnt_num * args.n_speakers:
            pass
        else:
            return utt_id_spk


def commpute_folder_id_per_spk(args, data_loader):
    folder_id_spk = {
        str(i): []
        for i in range(
            args.agnt_num * args.n_speakers + 1,
            (args.agnt_num + 1) * args.n_speakers + 1,
        )
    }
    for _, _, spk_id, file_id in tqdm(data_loader):
        fid_n = create_file_path_list(file_id)
        speaker_name = str(spk_id[0].tolist())

        if (
            int(speaker_name) <= (args.agnt_num + 1) * args.n_speakers
            and int(speaker_name) >= args.agnt_num * args.n_speakers + 1
        ):
            folder_id_spk[speaker_name].append(fid_n[1])
        elif int(speaker_name) <= args.agnt_num * args.n_speakers:
            pass
        else:
            return folder_id_spk


def convert_utt_id_spk(utt_id_spk):
    utt_id_spk_converted = {spk_name: [] for spk_name, _ in utt_id_spk.items()}
    for spk_name, utt_ids in utt_id_spk.items():
        for indx, _ in enumerate(utt_ids):
            utt_id_spk_converted[spk_name].append(indx)
    return utt_id_spk_converted


def compute_max_utt_per_spk(utt_id_spk):
    utt_max_spk = {}
    for spk_name, utt_ids in utt_id_spk.items():
        utt_max_spk[spk_name] = torch.tensor(utt_ids).view(-1).max()

    return utt_max_spk


def compute_val_logic(utt_id, spk_id, utt_max_spk):
    val_logic_spk = {}
    for spk_name, utt_max in utt_max_spk.items():
        if (
            utt_id <= utt_max.item()
            and utt_id >= utt_max.item() - 1
            # utt_id == utt_max.item()
            and spk_id == spk_name
        ):
            val_logic_spk[spk_name] = True
        else:
            val_logic_spk[spk_name] = False

    return val_logic_spk


def compute_val_logic_v2(utt_id_spk, eval_utts):
    val_logic_spk = {}
    for spk_name, utt_ids in utt_id_spk.items():
        for utt_id in utt_ids:
            if utt_id <= eval_utts:
                val_logic_spk[spk_name] = True
            else:
                val_logic_spk[spk_name] = False

    return val_logic_spk


def compute_val_logic_v3(utt_id, eval_utts):
    if utt_id <= eval_utts:
        val_logic_spk = True
    else:
        val_logic_spk = False

    return val_logic_spk


def compute_val_logic_v4(utt_id_list_spk, eval_utts, indx):
    if utt_id_list_spk[indx] <= eval_utts:
        val_logic_spk = True
    else:
        val_logic_spk = False

    return val_logic_spk


def compute_unique_list(folder_ids):
    res = []
    for item in folder_ids:
        if item not in res:
            res.append(item)
    return res


def compute_val_logic_v5(utt_id, folder_ids, eval_utts_per_folder, eval_utts=4):
    folder_ids_unique = compute_unique_list(folder_ids)
    val_logic_spk_collect = []
    count = 0
    for _ in folder_ids_unique:
        if utt_id <= eval_utts_per_folder and count <= eval_utts:
            val_logic_spk_collect.append(True)
            count += 1
        else:
            val_logic_spk_collect.append(False)

    return all(val_logic_spk_collect)


def compute_list_indx(converted_spk_utt_indx):
    x = []
    for indx, _ in enumerate(converted_spk_utt_indx):
        x.append(indx)

    return x[-1]


def compute_list_indx_folder(converted_spk_utt_indx):
    x = []
    for _, utt_indx in enumerate(converted_spk_utt_indx):
        x.append(utt_indx)

    return x[-1]


def vox_preprocess(args, root_dir, output_dir, logic_name):
    """Preprocess audio files into features for training."""

    output_dir_path = Path(f"{output_dir}_{args.agnt_num}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    voxceleb = VoxCeleb1IdentificationCustom(root_dir)
    data_loader = DataLoader(voxceleb, batch_size=1)

    utt_id_spk = commpute_utt_id_per_spk(args, data_loader)
    folder_id_spk = commpute_folder_id_per_spk(args, data_loader)
    # utt_id_spk_converted = convert_utt_id_spk(utt_id_spk)

    tester_dict = {spk_name: [] for spk_name, _ in utt_id_spk.items()}
    folder_dict = {spk_name: [] for spk_name, _ in folder_id_spk.items()}

    # utt_max_spk = compute_max_utt_per_spk(utt_id_spk)
    # val_logic_spk = compute_val_logic(utt_max_spk["5"].item(), "5", utt_max_spk)

    spks = (
        int(str(spk_id[0].tolist()))
        for _, _, spk_id, _ in iter(data_loader)
        if int(str(spk_id[0].tolist())) >= (args.agnt_num * args.n_speakers + 1)
    )

    infos = {
        "n_mels": args.feature_dim,
        "speakers": {
            str(spks): []
            for spks in itertools.takewhile(
                lambda x: (x <= (args.agnt_num + 1) * args.n_speakers),
                spks,
            )
        },
    }

    for waveform, _, spk_id, file_id in tqdm(data_loader):
        fid_n = create_file_path_list(file_id)
        speaker_name = str(spk_id[0].tolist())

        if (
            int(speaker_name) <= (args.agnt_num + 1) * args.n_speakers
            and int(speaker_name) >= args.agnt_num * args.n_speakers + 1
        ):
            tester_dict[speaker_name].append(int(fid_n[-1]))
            folder_dict[speaker_name].append(fid_n[1])

            converted_spk_utt_indx = compute_list_indx(tester_dict[speaker_name])
            # converted_spk_utt_indx_folder = compute_list_indx_folder(
            #     tester_dict[speaker_name]
            # )

            #     # val_logic_spk = compute_val_logic(int(fid_n[-1]), speaker_name, utt_max_spk)
            val_logic_spk = compute_val_logic_v3(converted_spk_utt_indx, 5)
            # val_logic_spk = compute_val_logic_v5(
            #     converted_spk_utt_indx_folder,
            #     folder_dict[speaker_name],
            #     1,
            # )

            # if val_logic_spk[speaker_name] == True and (
            #     logic_name == "test" or logic_name == "dev"
            # ):
            if val_logic_spk == True and (logic_name == "test" or logic_name == "dev"):
                feature = extract_features(
                    np.array(waveform).reshape(
                        -1,
                    ),
                    args,
                ).swapaxes(0, 1)

                # zero mean and unit variance
                mel_tensor = (feature - feature.mean()) / feature.std()

                random_file_path = output_dir_path / f"uttr-{uuid4().hex}.pt"
                torch.save(mel_tensor, random_file_path)

                infos["speakers"][speaker_name].append(
                    {
                        "feature_path": random_file_path.name,
                        "mel_len": mel_tensor.shape[0],
                    }
                )

            # if val_logic_spk[speaker_name] == False and logic_name == "train":
            if val_logic_spk == False and logic_name == "train":
                feature = extract_features(
                    np.array(waveform).reshape(
                        -1,
                    ),
                    args,
                ).swapaxes(0, 1)

                # zero mean and unit variance
                mel_tensor = (feature - feature.mean()) / feature.std()

                random_file_path = output_dir_path / f"uttr-{uuid4().hex}.pt"
                torch.save(mel_tensor, random_file_path)

                infos["speakers"][speaker_name].append(
                    {
                        "feature_path": random_file_path.name,
                        "mel_len": mel_tensor.shape[0],
                    }
                )

        elif int(speaker_name) <= args.agnt_num * args.n_speakers:
            pass

        else:
            break

    with open(output_dir_path / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)
