import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm import tqdm


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits.

    Args:
        images (torch.Tensor): The images to plot.
        titles (list): The titles of the images.
        h (int): The height of the images.
        w (int): The width of the images.
        n_row (int): The number of rows in the plot.
        n_col (int): The number of columns in the plot.

    Returns:
            None
    """

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def show_errors(num_errors: int, images, labels, predictions, out_dir: str) -> None:
    """Display the first `num_errors` errors in the dataset.

    Args:
        num_errors (int): The number of errors to display.
        images (torch.Tensor): The images in the dataset.
        labels (torch.Tensor): The ground truth labels.
        predictions (torch.Tensor): The predicted labels.
        out_dir (str): The directory to save the images.

    Returns:
        None
    """
    errors = []
    for i in range(len(images)):
        if labels[i] != predictions[i]:
            errors.append(i)
        if len(errors) == num_errors:
            break

    for i in errors:
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f"Label: {labels[i]}, Prediction: {predictions[i]}")
        plt.savefig(f"{out_dir}/error_{i}.png")
        plt.show()


def evaluate(model, data_loader, device):
    """Evaluate the model on the dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for the dataset.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def cer(preds, target):
    """Calculate the Character Error Rate (CER) between the predicted and target strings.

    Args:
        preds (list): The predicted strings.
        target (list): The target strings.

    Returns:
        float: The Character Error Rate (CER) between the predicted and target strings.
    """
    cer = CharErrorRate()
    return cer(preds, target)


def wer(preds, target):
    """Calculate the Word Error Rate (WER) between the predicted and target strings.

    Args:
        preds (list): The predicted strings.
        target (list): The target strings.

    Returns:
        float: The Word Error Rate (WER) between the predicted and target strings.
    """
    wer = WordErrorRate()
    return wer(preds, target)
