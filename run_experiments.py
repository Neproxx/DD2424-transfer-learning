from copy import deepcopy
from core.training import train_convnext

"""
Explanation of the config:
    - finetune.use_fraction: How much of the dataset to use for finetuning
        (applied before splitting into train, val, test)
    - finetune.epochs: Number of epochs to train for during finetuning
    - finetune.lr: Learning rate for finetuning
    - finetune.train_size: Fraction of the finetune dataset to use for training.
        The rest is split equally into a validation and test set.
    - finetune.batch_size: Batch size for finetuning.

    - pretrain.use_fraction: How much of the dataset to use for pretraining
    - pretrain.task: Which task to use for pretraining. Can be one of:
        - "rotated": Use the rotated images for pretraining
        - "grayscale": Use the grayscale images for pretraining
        - "rotated_grayscale": Use the rotated grayscale images for pretraining
        - None: Don't pretrain, just finetune
    - pretrain.epochs: Number of epochs to train for during pretraining
    - pretrain.lr: Learning rate for pretraining
    - pretrain.batch_size: Batch size for pretraining
"""
train_config_base = {
    "run_label": "MyExperimentName",
    "finetune": {
        # A fraction of 1 corresponds to 10,000 training samples (and 5,000 for validation test each)
        "use_fraction": 1,
        "epochs": 12,
        "lr": 0.0001,
        "train_size": 1 / 3,
        "batch_size": 16,
    },
    "pretrain": {
        # Note from Marcel: An epoch on the full dataset takes roughly 1h on my RTX 2070s
        # and a batchsize of 16 takes up 7GB of memory on my GPU (every additional sample adds ~200MB of memory usage)
        # That is only for the rotation dataset though, the grayscale dataset we need to half the batchsize for a similar memory usage
        # In general, the pretraining is much more demanding than the finetuning (due to the larger dataset size)
        # In total there are 171960 samples, so fractions correspond like this
        # (make sure to adapt the number of epochs accordingly for a fair comparison):
        "use_fraction": 0.072691,  # 10,000 samples
        # "use_fraction": 0.174461, # 30,000 samples
        # "use_fraction": 0.348922, # 60,000 samples
        # "use_fraction": 0.581531, # 100,000 samples
        "task": None,
        "epochs": 3,
        "lr": 0.0001,
        "batch_size": 12,
    },
}

tc_no_pretrain = deepcopy(train_config_base)
tc_no_pretrain["run_label"] = "without_pretraining"
tc_no_pretrain["pretrain"]["task"] = None

tc_rotated = deepcopy(train_config_base)
tc_rotated["run_label"] = "rotated"
tc_rotated["pretrain"]["task"] = "rotated"
tc_rotated["pretrain"]["epochs"] = 3

tc_grayscale = deepcopy(train_config_base)
tc_grayscale["run_label"] = "grayscale"
tc_grayscale["pretrain"]["task"] = "grayscale"
tc_grayscale["pretrain"]["epochs"] = 3

tc_rotated_grayscale = deepcopy(train_config_base)
tc_rotated_grayscale["run_label"] = "rotated_grayscale"
tc_rotated_grayscale["pretrain"]["task"] = "rotated_grayscale"
tc_rotated_grayscale["pretrain"]["epochs"] = 3

# NOTE: Uncomment the experiment you want to run
# train_convnext(train_config=tc_no_pretrain)
# train_convnext(train_config=tc_rotated)
# train_convnext(train_config=tc_grayscale)
# train_convnext(train_config=tc_rotated_grayscale)
