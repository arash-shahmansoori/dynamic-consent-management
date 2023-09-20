import argparse


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_args():
    # Commandline arguments
    parser = argparse.ArgumentParser(
        description="Arguments for Consent Management by Contrastive Embedding Replay"
    )
    ################################ Acoustic feature parameters ###########################
    parser.add_argument("--feature", default="fbank", type=str, help="acoustic feature")
    parser.add_argument(
        "--sample_rate", default=16000, type=int, help="sample rate of audio signal"
    )
    parser.add_argument(
        "--top_db", default=20, type=int, help="voice acticity detection"
    )
    parser.add_argument("--window_size", default=25, type=int, help="window size in ms")
    parser.add_argument(
        "--feature_dim", default=40, type=int, help="input acoustic feature dimension"
    )
    parser.add_argument(
        "--delta", default=False, type=bool, help="acoustic delta feature"
    )
    parser.add_argument(
        "--delta_delta", default=False, type=bool, help="acoustic d2elta feature"
    )
    ######################## Model #############################
    parser.add_argument("--dim_emb", default=256, type=int)
    parser.add_argument("--dim_emb_backend", default=128, type=int)
    parser.add_argument("--latent_dim", default=64, type=int)
    parser.add_argument("-n", "--n_speakers", type=int, default=40)
    parser.add_argument("--n_attributes", type=int, default=2)
    parser.add_argument("-n_other", "--n_speakers_other", type=int, default=20)
    parser.add_argument("--dim_cell", default=256, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--e_dim", default=32, type=int)
    parser.add_argument("--gp_norm_dvector", default=4, type=int)
    parser.add_argument("--gp_norm_cls", default=2, type=int)
    parser.add_argument("--context_size", default=5, type=int)
    parser.add_argument("--stride_conv", default=1, type=int)
    parser.add_argument("--dilation", default=1, type=int)
    parser.add_argument("--batch_norm", default=True, type=bool)
    parser.add_argument("--dropout_p", default=0.0, type=float)
    parser.add_argument(
        "--backbone", default="resnet18", type=str, help="ResNet Backbone"
    )
    ######################## Training setup ##########################
    parser.add_argument(
        "--epoch",
        dest="epoch",
        default=200,
        type=int,
        help="The number of epochs used for one bucket. (default: %(default)s)",
    )
    parser.add_argument(
        "--epoch_test",
        dest="epoch_test",
        default=1,
        type=int,
        help="The number of epochs used for one bucket. (default: %(default)s)",
    )
    parser.add_argument("--epochs_per_dvector", type=int, default=5)
    parser.add_argument("--epochs_per_cls", type=int, default=2)
    parser.add_argument("--epochs_per_dvector_latent", type=int, default=1)
    ######################## Data #########################
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data\\train_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_ablation",
        type=str,
        default="data\\train_dir_ablation_agnt",
    )
    parser.add_argument(
        "--data_dir_other",
        type=str,
        default=f"data\\nonreduced_train_dir_agnt_other",
    )

    parser.add_argument(
        "--data_dir_ninty_pcnt",
        type=str,
        default="data\\train_reduced_ninty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_eighty_pcnt",
        type=str,
        default="data\\train_reduced_eighty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_seventy_pcnt",
        type=str,
        default="data\\train_reduced_seventy_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_sixty_pcnt",
        type=str,
        default="data\\train_reduced_sixty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_fifty_pcnt",
        type=str,
        default="data\\train_reduced_fifty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_forty_pcnt",
        type=str,
        default="data\\train_reduced_forty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_thirty_pcnt",
        type=str,
        default="data\\train_reduced_thirty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_twenty_pcnt",
        type=str,
        default="data\\train_reduced_twenty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "--data_dir_ten_pcnt",
        type=str,
        default="data\\train_reduced_ten_pcnt_dir_agnt",
    )

    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default="data\\val_dir_agnt",
    )
    parser.add_argument(
        "--validation_data_dir_ablation",
        type=str,
        default="data\\val_dir_ablation_agnt",
    )
    parser.add_argument(
        "--validation_data_dir_other",
        type=str,
        default=f"data\\val_dir_agnt_other",
    )

    parser.add_argument(
        "--data_dir_vox_train",
        type=str,
        default=f"data\\train_vox_agnt",
    )
    parser.add_argument(
        "--data_dir_vox_test",
        type=str,
        default=f"data\\test_vox_agnt",
    )
    parser.add_argument(
        "--data_dir_vox_dev",
        type=str,
        default=f"data\\dev_vox_agnt",
    )

    parser.add_argument("-agt_num", "--agnt_num", type=int, default=0)

    parser.add_argument("-m_labeled", "--n_utterances_labeled", type=int, default=20)
    parser.add_argument(
        "-m_unlabeled", "--n_utterances_unlabeled", type=int, default=20
    )

    parser.add_argument(
        "-m_labeled_reg", "--n_utterances_labeled_reg", type=int, default=20
    )
    parser.add_argument(
        "-m_labeled_old", "--n_utterances_labeled_old", type=int, default=20
    )
    parser.add_argument("-m_labeled_", "--n_utterances_labeled_", type=int, default=20)

    parser.add_argument("-t_labeled", "--nt_utterances_labeled", type=int, default=6)
    parser.add_argument(
        "-t_labeled_other", "--nt_utterances_labeled_other", type=int, default=2
    )
    parser.add_argument(
        "-v_labeled", "--nv_utterances_labeled", type=int, default=6
    )  # Set to 5 for N=200 speakers/agent
    parser.add_argument(
        "-v_unlabeled", "--nv_utterances_unlabeled", type=int, default=6
    )
    parser.add_argument(
        "-v_labeled_prev_other",
        "--nv_utterances_labeled_prev_other",
        type=int,
        default=6,
    )

    parser.add_argument("-dev_vox_utt", "--n_dev_vox_utts", type=int, default=4)
    parser.add_argument("-test_vox_utt", "--n_test_vox_utts", type=int, default=4)
    parser.add_argument("-train_vox_utt", "--n_train_vox_utts", type=int, default=20)

    parser.add_argument("-n_utts_select", "--n_utts_selected", type=int, default=20)
    parser.add_argument("-n_utts_select_", "--n_utts_selected_", type=int, default=20)
    parser.add_argument(
        "-n_utts_select_reg", "--n_utts_selected_reg", type=int, default=20
    )
    parser.add_argument(
        "-n_utts_select_old", "--n_utts_selected_old", type=int, default=20
    )

    parser.add_argument("--seg_len", type=int, default=160)
    parser.add_argument("--spk_per_bucket", type=int, default=5)
    parser.add_argument("--stride_per_bucket", type=int, default=5)
    parser.add_argument("--stride", default=5, type=int, help="stride size")
    parser.add_argument("--valid_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    ######################## ER #########################
    parser.add_argument(
        "--max_mem",
        dest="max_mem",
        default=120,
        type=int,
        help="Memory buffer size (default: %(default)s)",
    )

    parser.add_argument(
        "--max_mem_unsup",
        dest="max_mem_unsup",
        default=120,
        type=int,
        help="Memory buffer size (default: %(default)s)",
    )

    parser.add_argument(
        "--buffer_sampling_mode",
        dest="buffer_sampling_mode",
        default="per_spk_random_max_mem",
        type=str,
        choices=["per_spk_random_max_mem", "recent_max_mem"],
        help="Sampling types from embedding buffer when the buffer is full",
    )
    ################################## Checkpoints ####################################
    parser.add_argument(
        "-cp_dvector",
        "--checkpoint_dir_dvector",
        type=str,
        default="checkpoints\dvectors",
    )

    parser.add_argument(
        "-cp_d", "--checkpoint_dir", type=str, default="checkpoints\cls"
    )
    parser.add_argument(
        "-cp_d_m",
        "--checkpoint_dir_modular",
        type=str,
        default="checkpoints\cls_modular",
    )
    parser.add_argument("-ckpt_cls", default=None, type=str)
    parser.add_argument("-ckpt_scratch", default=None, type=str)
    ################################### Logging #######################################
    parser.add_argument("--log_training", default=False, type=bool)
    parser.add_argument("--log_validation", default=True, type=bool)
    ################################### LRScheduler ###################################
    parser.add_argument(
        "--lr_scheduler",
        dest="lr_scheduler",
        default=True,
        type=boolean_string,
        help="To use the learning rate scheduler",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        default=5,
        type=int,
        help="Number of events to wait if no improvement and then stop the training.",
    )
    parser.add_argument(
        "--min_lr",
        dest="min_lr",
        default=1e-6,
        type=float,
        help="least lr value to reduce to while updating",
    )
    parser.add_argument(
        "--factor",
        dest="factor",
        default=0.5,
        type=float,
        help="factor by which the lr should be updated",
    )
    #################### Early Stopping ######################
    parser.add_argument(
        "--early_stopping",
        dest="early_stopping",
        default=True,
        type=boolean_string,
        help="To use the early stopping",
    )
    parser.add_argument(
        "--min_delta",
        dest="min_delta",
        default=0,
        type=float,
        help="A minimum increase in the score to qualify as an improvement",
    )
    parser.add_argument(
        "--min_delta_loss",
        dest="min_delta_loss",
        default=0.12,
        type=float,
        help="A minimum increase in the score to qualify as an improvement",
    )
    parser.add_argument(
        "--min_delta_unreg",
        dest="min_delta_unreg",
        default=5,
        type=float,
        help="A minimum increase in the score to qualify as an improvement for unregistering",
    )
    parser.add_argument(
        "--threshold_val_acc",
        dest="threshold_val_acc",
        default=96,
        type=int,
        help="Threshold validation accuracy for early stopping.",
    )
    parser.add_argument(
        "--threshold_val_loss",
        dest="threshold_val_loss",
        default=0.12,
        type=int,
        help="Threshold validation loss for early stopping.",
    )
    parser.add_argument(
        "--verbose_stopping",
        dest="verbose_stopping",
        default=False,
        type=boolean_string,
        help="Print the early stopping message during iterations",
    )
    parser.add_argument(
        "--patience_stopping",
        dest="patience_stopping",
        default=5,
        type=int,
        help="Number of events to wait if no improvement and then stop the training.",
    )
    ############################# Stochastic weight averaging (SWA) #####################
    parser.add_argument(
        "--swa_start",
        type=float,
        default=50,  # Set to 20 for new registrations
        help="SWA start epoch number",
    )
    parser.add_argument(
        "--swa_lr",
        type=float,
        default=0.005,
        help="SWA LR (default: 0.005)",
    )

    ############################# Output path for the data ############################
    parser.add_argument(
        "-o_train",
        "--output_dir_train",
        type=str,
        default=f"data\\train_dir_agnt",
    )
    parser.add_argument(
        "-o_train_ablation",
        "--output_dir_train_ablation",
        type=str,
        default=f"data\\train_dir_ablation_agnt",
    )
    parser.add_argument(
        "-o_nonreduced_train_other",
        "--output_dir_nonreduced_train_other",
        type=str,
        default=f"data\\nonreduced_train_dir_agnt_other",
    )

    parser.add_argument(
        "-o_train_ninty_pcnt",
        "--output_dir_train_ninty_pcnt",
        type=str,
        default="data\\train_reduced_ninty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_eighty_pcnt",
        "--output_dir_train_eighty_pcnt",
        type=str,
        default="data\\train_reduced_eighty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_seventy_pcnt",
        "--output_dir_train_seventy_pcnt",
        type=str,
        default="data\\train_reduced_seventy_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_sixty_pcnt",
        "--output_dir_train_sixty_pcnt",
        type=str,
        default="data\\train_reduced_sixty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_fifty_pcnt",
        "--output_dir_train_fifty_pcnt",
        type=str,
        default="data\\train_reduced_fifty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_forty_pcnt",
        "--output_dir_train_forty_pcnt",
        type=str,
        default="data\\train_reduced_forty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_thirty_pcnt",
        "--output_dir_train_thirty_pcnt",
        type=str,
        default="data\\train_reduced_thirty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_twenty_pcnt",
        "--output_dir_train_twenty_pcnt",
        type=str,
        default="data\\train_reduced_twenty_pcnt_dir_agnt",
    )
    parser.add_argument(
        "-o_train_ten_pcnt",
        "--output_dir_train_ten_pcnt",
        type=str,
        default="data\\train_reduced_ten_pcnt_dir_agnt",
    )

    parser.add_argument(
        "-o_val",
        "--output_dir_val",
        type=str,
        default="data\\val_dir_agnt",
    )
    parser.add_argument(
        "-o_val_ablation",
        "--output_dir_val_ablation",
        type=str,
        default="data\\val_dir_ablation_agnt",
    )
    parser.add_argument(
        "-o_val_other",
        "--output_dir_val_other",
        type=str,
        default=f"data\\val_dir_agnt_other",
    )

    parser.add_argument(
        "-o_train_dir_vox",
        "--output_dir_vox_train",
        type=str,
        default=f"data\\train_vox_agnt",
    )
    parser.add_argument(
        "-o_test_dir_vox",
        "--output_dir_vox_test",
        type=str,
        default=f"data\\test_vox_agnt",
    )
    parser.add_argument(
        "-o_dev_dir_vox",
        "--output_dir_vox_dev",
        type=str,
        default=f"data\\dev_vox_agnt",
    )

    ############################# Output path for the results ############################
    parser.add_argument(
        "-r_td",
        "--result_dir_td",
        type=str,
        default="results\\elapsed_time",
    )

    parser.add_argument(
        "-r_acc_train",
        "--result_dir_acc_train",
        type=str,
        default="results\\train_acc",
    )
    parser.add_argument(
        "-r_acc_val",
        "--result_dir_acc_val",
        type=str,
        default="results\\val_acc",
    )
    parser.add_argument(
        "-r_tsne",
        "--result_dir_tsne",
        type=str,
        default="results\\tsne",
    )

    parser.add_argument(
        "-r_loss_train",
        "--result_dir_loss_train",
        type=str,
        default="results\\train_loss",
    )
    parser.add_argument(
        "-r_loss_val",
        "--result_dir_loss_val",
        type=str,
        default="results\\val_loss",
    )

    parser.add_argument(
        "-r_td_modular",
        "--result_dir_modular_td",
        type=str,
        default="results\\elapsed_time_modular",
    )

    parser.add_argument(
        "-r_acc_train_modular",
        "--result_dir_acc_train_modular",
        type=str,
        default="results\\train_acc_modular",
    )
    parser.add_argument(
        "-r_acc_val_modular",
        "--result_dir_acc_val_modular",
        type=str,
        default="results\\val_acc_modular",
    )

    parser.add_argument(
        "-r_loss_train_modular",
        "--result_dir_loss_train_modular",
        type=str,
        default="results\\train_loss_modular",
    )
    parser.add_argument(
        "-r_loss_val_modular",
        "--result_dir_loss_val_modular",
        type=str,
        default="results\\val_loss_modular",
    )

    parser.add_argument(
        "-o_eer",
        "--result_dir_eer",
        type=str,
        default="results\\EERs",
    )

    parser.add_argument(
        "-out_result",
        "--output_dir_results",
        type=str,
        default="results\\plots",
    )

    args = parser.parse_args()
    return args
