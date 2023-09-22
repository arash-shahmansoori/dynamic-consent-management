# Dynamic Consent Management of Speakers in Voice Assistant Systems by Contrastive Embedding Replay

PyTorch implementation of: ``**Dynamic Recognition of Speakers for Consent Management by Contrastive Embedding Replay''** ([arXiv](https://arxiv.org/abs/2205.08459)).

## Installation

```angular2
pip install -r requirements.txt
```

## Data

To create speakers per agent use the following steps:

Make sure that the LibriSpeech dataset is downloaded and follows the following tree structure in the data folder.

```angular2
data ---.
        ¦---> LibriSpeech --> train-clean-100 --> Speakers' folders
                          --> Books.txt
                          --> Chapters.txt
                          --> License.txt
                          --> ReadMe.txt
                          --> Speakers.txt

```

Similarly, for the noisy utterances, make sure to download the LibriSpeech dataset with the following data structure.

```angular2
data ---.
        ¦---> LibriSpeechOther --> train-other-500 --> Speakers' folders
                               --> Books.txt
                               --> Chapters.txt
                               --> License.txt
                               --> ReadMe.txt
                               --> Speakers.txt

```

Choose the appropriate root directory "root_dir" and the destination directory "dest_dir" in the "create_spks_per_agnt.py". For instance, the following can be used for the "LibriSpeech" dataset:

```angular2
root_dir = "data\\LibriSpeech"
dest_dir = f"data\\LibriSpeech_modular\\agnt_{args.agnt_num}_spks_{num_spks_per_agnt}"
```

Once the aformentioned directories are selected, use the following command to create speakers per agent.

```angular2
python create_spks_per_agnt.py
```

## Preprocessing

To create datasets for training-testing using disjoint utterences of different speakers, use the following steps:

Choose the appropriate "root_name" and "file_name" in "main_preprocess.py" in the "plugins" folder. For instance, in the case of clean utterances, the following names can be used:

```angular2
root_name = f"./data/LibriSpeech_modular/agnt_{args.agnt_num}_spks_{args.n_speakers}"
file_name = (
        f"LibriSpeech_modular/agnt_{args.agnt_num}_spks_{args.n_speakers}/Speakers.txt"
    )
```

For the case of noisy utterances, the following names can be used:

```angular2
root_name = f"./data/LibriSpeech_train_other_500/train-other-500_agnt_{args.agnt_num}"
file_name = "data/LibriSpeech_train_other_500/Speakers.txt"
```

Set the "output_dir" in "main_preprocess.py" to the appropriate name:

e.g., for training with clean data and the entire utterances:

```angular2
output_dir = args.output_dir_train
pcnt_old = "full"
```

e.g., for testing with clean data:

```angular2
output_dir = args.output_dir_val
pcnt_old = "eval"
```

Similarly, choose the appropriate name for the "output_dir" for the case of training with reduced utterances and training and testing for noisy utterances. The appropriate names for the aforementioned "output_dir" for different cases are provides in "parse_args()" function.

<!-- For the ease of use, the "output_dir" files for an Agent is created: "train_dir_agnt_0", "train_reduced_fifty_pcnt_dir_agnt_0", "nonreduced_train_dir_agnt_other_0", "val_dir_agnt_0", and "val_dir_agnt_other_0" for some of the aforementioned cases. So, one can just go to the next step and run the training scripts. -->

Make sure you follow the instructions in "DisjointTrainTest" class used in the function "preprocess" from "main_preprocess.py" to create disjoint utterances for training and testing. Also, follow the commented instructions in "DisjointTrainTest" class to create "x % use of training data" for the case of training with clean dataset using "x %" of clean utterances.

Once the aformentioned steps are completed, choose the appropriate "PLUGIN_NAME" in the "main_executable.py" and use the following command to create each dataset.

```angular2
python main_executable.py
```

This would create an "output_dir" with the corresponding name for each dataset that can be used during the training/testing accordingly.

## Training

The training process can be divided to two categories: proposed, and literature (from scratch).

### Proposed

To run the simulations for training according to the proposed method, first set the number of buckets in the "utils_hyper_params.py" file in the folder "utils_final" to "8" as: "num_of_buckets: int = 8", e.g., for the case of 40 total speakers distributed among 8 buckets each of which containing 5 speakers. Then, set the number of speaker per bucket and the stride to the number of speakrs that need to be registered in each bucket, e.g., 5. This can be achieved by setting the following arguments in the "utils_args.py" in the folder "utils_final" as:

```angular2
parser.add_argument("--spk_per_bucket", type=int, default=5)
parser.add_argument("--stride_per_bucket", type=int, default=5)
parser.add_argument("--stride", default=5, type=int, help="stride size")
```

For the case of supervised contrastive training, choose  PLUGIN_NAME="plugins.main_train_cont_sup" in the "main_executable.py" and use the following command.

```angular2
python main_executable.py
```

For the case of unsupervised contrastive training, choose  PLUGIN_NAME="plugins.main_train_cont_unsup" in the "main_executable.py" and run "main_executable.py" as mentioned above.

### Literature (From Scratch)

To run the simulations for training according to the literature, first set the number of buckets in the "utils_hyper_params.py" file in the folder "utils_final" to "1" as: "num_of_buckets: int = 1". Then, set the number of speaker per bucket and the stride to the total number of speakrs that need to be registered, e.g., 40. This can be achieved by setting the following arguments in the "utils_args.py" in the folder "utils_final" as:

```angular2
parser.add_argument("--spk_per_bucket", type=int, default=40)
parser.add_argument("--stride_per_bucket", type=int, default=40)
parser.add_argument("--stride", default=40, type=int, help="stride size")
```

Finally, choose  PLUGIN_NAME="plugins.main_train_scratch_unsup" in the "main_executable.py" and run "main_executable.py".

### Other Dataset

The extension of the training process to VoxCeleb dataset is provided as follows. In the plugins folder, choose logic_name = "train" and output_dir = args.output_dir_vox_train for training set, and logic_name = "test" and output_dir = args.output_dir_vox_test for the testing set. Select the args.agnt_num accordingly. Then, set PLUGIN_NAME="plugins.main_vox_preprocess" in the "main_executable.py" and use the following command.

```angular2
python main_executable.py
```

The training process follows a similar process as for the LibriSpeech dataset. The corresponding files for training and evaluation on the VoxCeleb dataset are shown by the "_vox" in the corresponding folders.

### Dynamic Registrations

For the dynamic registration using the default initial optimal buckets in the paper, use "create_unique_opt_bkt_spks_existing" in the folder "compute_optimal_buckets_final". To use another optimal initial bucket according to a fresh run of the L2 Euclidean distance use "create_unique_opt_bkts_spks" in the same folder.

Then, choose  PLUGIN_NAME="plugins.main_dynamic_reg_sup" in the "main_executable.py" and run "main_executable.py" for each new registration round for the supervised case.

Use the same process for the unsupervised dynamic registration by choosing  PLUGIN_NAME="plugins.main_dynamic_reg_unsup" in the "main_executable.py" and running "main_executable.py" for each new registration round.

Note: For the case of using 10% of old utterances set: n_utterances_labeled, n_utterances_unlabeled, n_utterances_labeled_reg, n_utterances_labeled_old, n_utterances_labeled_ to: 3 in the utils/utils_args to avoid StopIteration error during the training. For the case of using 30% of old utterances set the aformentioned parameters to:10.

### Dynamic Removal

For the dynamic removal of previously registered speakers in the buckets, choose  PLUGIN_NAME="plugins.main_unreg_sup" in the "main_executable.py" and run "main_executable.py" for the supervised case. Similarly, choose PLUGIN_NAME="plugins.main_unreg_unsup" and run "main_executable.py" for the unsupervised case.

### Dynamic Re-Registration

For the dynamic re-registration of previously unregistered speakers, choose  PLUGIN_NAME="plugins.main_re_reg_sup" in the "main_executable.py" and run "main_executable.py" for the supervised case. Similarly, choose PLUGIN_NAME="plugins.main_re_reg_unsup" and run "main_executable.py" for the unsupervised case.

### Saved Checkpoints

The checipoints for the contrastive feature extraction (i.e., dvectors) and classification (i.e., cls) should be stored in the folder "checkpoints" as the sub-folders with the corresponding names.

## Test

Run the test from the "tests" folder for the multi-strided random sampling proposed in the paper. One can simply run the test from the aforementioned folder with the coverage report using the following command.

```angular2
pytest --cov
```

Subsequently, the report is obtained by the following command.

```angular2
coverage report -m
```

The aforementioned test in the "tests" folder provides 100% coverage.

### Verification

To perform verification and obtain the necessary metrics, e.g., EER, CLLR, and DCF, choose PLUGIN_NAME="plugins.main_verification_sup" and run "main_executable.py" for the supervised case. Similarly, choose PLUGIN_NAME="plugins.main_verification_unsup" and run "main_executable.py" for the unsupervised case.

### Plots

To create each plot from the paper, select the plot of interest by setting RESULT_NAME in the "main_plot_results.py", choose the corresponding function name by setting "plt_fn_name" in the "main_plot_results.py", and run the following command.

```angular2
python main_plot_results.py
```

## References

The proposed consent management implemented here is from the following publication:

- [Dynamic Recognition of Speakers for Consent
Management by Contrastive Embedding Replay](https://arxiv.org/abs/2205.08459)

Note: The paper mentioned above has been recently accepted for publication in IEEE Transactions on Neural Networks and Learning Systems (TNNLS) with the "DOI:10.1109/TNNLS.2023.3317493". Please cite the paper using the following format in the future if you are using the current repository:

Arash Shahmansoori and Utz Roedig."Dynamic Recognition of Speakers for Consent
Management by Contrastive Embedding Replay." IEEE Transactions on Neural Networks and Learning Systems, vol. [Volume], no. [Issue], pp. [Page range], [Month] [2023]. DOI:10.1109/TNNLS.2023.3317493 (corresponding author: Arash Shahmansoori)

- Cite this repository

To support differential privacy for speaker recognition in voice assistant systems using the private data and a small portion of publicly available data, please refer to the following repository:

- <https://github.com/arash-shahmansoori/differentially-private-consent-management.git>

## License

[MIT License](LICENSE)

---
***Contact the Author***

The author "Arash Shahmansoori" e-mail address: <arash.mansoori65@gmail.com>
