import os
import pickle
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss, Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split

from core.dataset import VGGFace2Dataset
from core.models import ConvNeXt
from core.constants import device

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5)

# Specify the path to the folders containing the images
train_path = os.path.join("data", "preprocessed", "train")
test_path = os.path.join("data", "preprocessed", "test")

allowed_tasks = ["rotated", "grayscale", "rotated_grayscale"]


def load_dataset(train_config):
    """
    This method returns two dictionaries. The first one contains up to four different dataset loaders:
    - pretrain_loader: used to pretrain the model
    - train_loader: used to train the model during fine-tuning
    - val_loader: used to validate the model during fine-tuning
    - test_loader: used to test the model after fine-tuning
    The second dictionary contains meta data about the datasets, e.g. the number of classes.
    """
    pretrain_task = train_config["pretrain"]["task"]
    if pretrain_task is not None:
        assert pretrain_task in ["rotated", "grayscale", "rotated_grayscale"]
        pt_dataset = VGGFace2Dataset(root_dir=train_path, version=pretrain_task)
    ft_dataset = VGGFace2Dataset(root_dir=test_path, version="original")

    # Reduce overall size of dataset according to config
    ft_frac = train_config["finetune"]["use_fraction"]
    ft_size = int(ft_frac * len(ft_dataset))
    ft_dataset_subset, _ = random_split(
        dataset=ft_dataset,
        lengths=[ft_size, len(ft_dataset) - ft_size],
        generator=torch.Generator().manual_seed(42),
    )

    if pretrain_task is not None:
        pt_frac = train_config["pretrain"]["use_fraction"]
        pt_train_size = int(pt_frac * len(pt_dataset))
        pt_dataset_subset, _ = random_split(
            dataset=pt_dataset,
            lengths=[pt_train_size, len(pt_dataset) - pt_train_size],
            generator=torch.Generator().manual_seed(42),
        )

    # Determine split sizes
    ft_train_size = train_config["finetune"]["train_size"]
    ft_train_size = int(ft_train_size * len(ft_dataset_subset))
    ft_val_test_size = len(ft_dataset_subset) - ft_train_size
    ft_val_size = int(ft_val_test_size / 2)
    ft_test_size = int(ft_val_test_size - ft_val_size)

    # Split dataset
    ft_train_dataset, ft_val_test_dataset = random_split(
        dataset=ft_dataset_subset,
        lengths=[ft_train_size, ft_val_test_size],
        generator=torch.Generator().manual_seed(42),
    )
    ft_val_dataset, ft_test_dataset = random_split(
        dataset=ft_val_test_dataset,
        lengths=[ft_val_size, ft_test_size],
        generator=torch.Generator().manual_seed(42),
    )

    ft_bs = train_config["finetune"]["batch_size"]
    pt_bs = train_config["pretrain"]["batch_size"]
    if pretrain_task is not None:
        pt_loader = DataLoader(pt_dataset_subset, batch_size=pt_bs, shuffle=True)
    ft_train_loader = DataLoader(ft_train_dataset, batch_size=ft_bs, shuffle=True)
    ft_val_loader = DataLoader(ft_val_dataset, batch_size=ft_bs, shuffle=False)
    ft_test_loader = DataLoader(ft_test_dataset, batch_size=ft_bs, shuffle=False)

    loaders = {
        "train_loader": ft_train_loader,
        "val_loader": ft_val_loader,
        "test_loader": ft_test_loader,
    }
    meta = {
        "num_ft_train_samples": ft_train_size,
        "num_ft_val_samples": ft_val_size,
        "num_ft_test_samples": ft_test_size,
        "num_ft_classes": ft_dataset.n_classes_orig,
        "pretrain_task": pretrain_task,
    }
    if pretrain_task is not None:
        loaders["pretrain_loader"] = pt_loader
        meta["num_pt_samples"] = pt_train_size
        meta["pt_output_size"] = pt_dataset.output_size

    return loaders, meta


def train_convnext(train_config):
    """
    Initializes and trains a ConvNeXt model.
    """
    dataloaders, meta = load_dataset(train_config)
    print(f"Meta information on run: {meta}")

    ### Save training config and meta data for later reference
    base_path = generate_results_path(train_config)
    with open(os.path.join(base_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)
    with open(os.path.join(base_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    ### Pretrain the model
    pretrain_task = train_config["pretrain"]["task"]
    if pretrain_task is not None:
        assert pretrain_task in allowed_tasks
        model = pretrain_convnext(
            train_config,
            dataloaders["pretrain_loader"],
            meta,
        )
    else:
        model = ConvNeXt(
            meta["num_ft_classes"],
            device=device,
            fresh_init=train_config["fresh_init"],
        )

    ### Fine-tune the model
    model = finetune_convnext(model, train_config, dataloaders)

    return model


def pretrain_convnext(
    train_config: dict,
    dataloader: DataLoader,
    meta: dict,
):
    """
    Initializes and pretrains a ConvNeXt model on a given pretraining task.
    At the end, the output layers are replaced by a fully connected layer
    for later fine-tuning.

    Args:
        train_config (dict): A dictionary containing training arguments.
        dataloader (DataLoader): The dataloader to use for pretraining.
        meta (dict): A dictionary containing meta data about the dataset.
            For example the output dimensions of the pretraining task
            and the input dimension for the downstream task.
    """
    n_samples = meta["num_pt_samples"]
    print(f"Starting pretraining on {n_samples} samples...")

    # All pretraining tasks have a single output layer, except for rotated_grayscale
    # which has two output layers (rotation output with size 4 and grayscale output with size 224*224)
    output_sizes = meta["pt_output_size"]
    pretrain_task = train_config["pretrain"].get("task", None)
    if pretrain_task == "rotated_grayscale":
        output_sizes = [4, output_sizes - 4]

    model = ConvNeXt(
        output_sizes, device=device, fresh_init=train_config["fresh_init"]
    )

    criterion = get_loss_func(pretrain_task)
    optimizer = optim.Adam(model.parameters(), train_config["pretrain"]["lr"])
    n_epochs = train_config["pretrain"]["epochs"]
    train_losses_raw = []
    # for epoch in tqdm(range(n_epochs)):
    for epoch in range(n_epochs):
        model.train()
        # for batch in dataloader:
        for batch in tqdm(dataloader):
            inputs = batch["image"]
            labels = batch["label"]
            loss = perform_step(model, criterion, inputs, labels, optimizer)
            train_losses_raw.append(loss)
        print(f"Epoch: {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

    # Replace output layers with a fully connected layer for the fine-tuning task
    model.replace_fc(meta["num_ft_classes"])

    # Save model
    base_path = generate_results_path(train_config)
    torch.save(model.state_dict(), os.path.join(base_path, "model_pretrained.pt"))

    # Create and save plots
    fname = "pretraining_train_loss_curve.png"
    plot_results(
        train=get_smooth_loss(train_losses_raw),
        val=[],
        title=None,
        # title="Training loss during pre-training",
        xlabel="Update step",
        ylabel="Loss (smoothed)",
        save_path=os.path.join(base_path, fname),
    )

    # Save raw results for potential later re-plotting
    fname = "pretraining_train_loss_curve.pkl"
    with open(os.path.join(base_path, fname), "wb") as f:
        pickle.dump(train_losses_raw, f)

    print("Pretraining finished.")
    return model


def finetune_convnext(
    model: ConvNeXt,
    train_config: dict,
    dataloaders: dict,
):
    """
    Fine-tunes a ConvNeXt model on the given dataset.

    Args:
        model (ConvNeXt): The model to fine-tune.
        train_config (dict): A dictionary containing training arguments.
        dataloaders (dict): A dictionary containing the dataloaders for
            training, validation and testing.
    """
    print("Starting fine-tuning...")
    criterion = get_loss_func("classification")
    optimizer = optim.Adam(model.parameters(), train_config["finetune"]["lr"])
    n_epochs = train_config["finetune"]["epochs"]

    loss_train_per_update = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []
    acc_test_per_epoch = []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for batch in dataloaders["train_loader"]:
            inputs = batch["image"]
            labels = batch["label"]
            loss = perform_step(model, criterion, inputs, labels, optimizer)
            loss_train_per_update.append(loss)

        model.eval()
        with torch.no_grad():
            acc_train_per_epoch.append(
                compute_accuracy(model, dataloaders["train_loader"])
            )
            acc_val_per_epoch.append(compute_accuracy(model, dataloaders["val_loader"]))
            acc_test_per_epoch.append(
                compute_accuracy(model, dataloaders["test_loader"])
            )

        print(
            f"Epoch: {epoch + 1}/{n_epochs}, loss: {loss:.4f}, accuracy train: {acc_train_per_epoch[-1]:.4f}, accuracy val: {acc_val_per_epoch[-1]:.4f}"
        )

    # Compute accuracy on test set
    model.eval()
    with torch.no_grad():
        acc_test = compute_accuracy(model, dataloaders["test_loader"])
    print(f"Accuracy achieved on test set: {acc_test:.6f}")

    # save model
    fname = "model_finetuned.pt"
    base_path = generate_results_path(train_config)
    torch.save(model.state_dict(), os.path.join(base_path, fname))

    # Create and save plots
    fname = f"finetuning_train_val_acc_curves__testAcc={acc_test:.4f}.png"
    plot_results(
        train=acc_train_per_epoch,
        val=acc_val_per_epoch,
        title=None,
        # title="Accuracy after fine-tuning",
        xlabel="Epoch",
        ylabel="Accuracy",
        save_path=os.path.join(base_path, fname),
    )

    fname = f"finetuning_train_loss_curve.png"
    plot_results(
        train=loss_train_per_update,
        val=[],
        title=None,
        # title="Training loss during fine-tuning",
        xlabel="Update step",
        ylabel="Loss",
        save_path=os.path.join(base_path, fname),
    )

    # Save raw results for possible later plotting
    fname = f"finetuning_curves.pkl"
    with open(os.path.join(base_path, fname), "wb") as f:
        pickle.dump(
            {
                "train_acc": acc_train_per_epoch,
                "val_acc": acc_val_per_epoch,
                "test_acc": acc_test_per_epoch,
                "train_loss": loss_train_per_update,
            },
            f,
        )

    print("Fine-tuning finished.")


def perform_step(model, criterion, inputs, labels, optimizer=None):
    """
    Performs a single step of training on the given model.
    If no optimizer is given, the model is only evaluated.
    """
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    return loss.item()


def plot_results(train, val, title, xlabel, ylabel, save_path, logscale=False):
    """
    Plots the training and validation curves for a given model.

    Args:
        train (list): A list containing the training values.
        val (list): A list containing the validation values.
        title (str): The title of the plot.
        what (str): The y-axis label (e.g. 'loss' or 'accuracy').
        save_path (str): The path to save the plot to.
    """
    plt.figure(figsize=(9, 6))
    plt.plot(train, label="Training")
    if len(val) > 0:
        plt.plot(val, label="Validation")
    if logscale:
        plt.yscale("log")
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_results_path(train_config):
    """
    Generates a folder name for the results based on the training configuration.
    """
    if not "date" in train_config:
        train_config["date"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = train_config["date"] + "__" + train_config["run_label"]
    path = os.path.join("results", folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def compute_accuracy(model, dataloader):
    n_correct = 0
    n_samples = 0

    for batch in dataloader:
        inputs = batch["image"].to(device)
        targets = batch["label"].to(device)
        pred_logits = model(inputs)
        pred_labels = pred_logits.argmax(dim=1)
        target_labels = targets.argmax(dim=1)
        n_correct += (pred_labels == target_labels).sum().item()
        n_samples += target_labels.shape[0]

    return n_correct / n_samples


def get_loss_func(task):
    if task in ["classification", "rotated"]:
        return CrossEntropyLoss()
    elif task == "grayscale":
        return MSELoss()
    elif task == "rotated_grayscale":
        return MultiTaskLoss(clf_size=4, strategy="equal")


class MultiTaskLoss(Module):
    """
    A loss function that combines classification and regression losses.
    Assumes that the first n outputs are for classification and the remaining
    outputs are for regression.
    """

    def __init__(self, clf_size, strategy):
        """
        Args:
            clf_size (int): The number of outputs for classification.
            strategy (str): The strategy to use for combining the losses.
        """
        super().__init__()
        self.clf_size = clf_size
        self.strategy = strategy

        self.classification_loss = CrossEntropyLoss()
        self.regression_loss = MSELoss()

    def forward(self, pred, target):
        clf_pred = pred[0]
        reg_pred = pred[1]

        clf_target = target[:, : self.clf_size]
        reg_target = target[:, self.clf_size :]

        class_loss = self.classification_loss(clf_pred, clf_target)
        regression_loss = self.regression_loss(reg_pred, reg_target)

        # NOTE: As additional experiments, we could explore different strategies for combining the losses.
        # E.g. implement the approach from paper "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
        # and compare it with a plain 1:1 weighting of the losses
        # The idea of the paper is to let the network learn the loss
        if self.strategy == "equal":
            total_loss = (class_loss + regression_loss) / 2
        else:
            raise NotImplementedError("Only equal strategy is implemented")

        return total_loss


def get_smooth_loss(losses):
    # Applies an exponential moving average to the losses for smoothing
    alpha = 0.9
    smooth_loss = [losses[0]]
    for i, loss in enumerate(losses[1:]):
        smooth_loss.append(alpha * smooth_loss[i] + (1 - alpha) * loss)
    return smooth_loss
