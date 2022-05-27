import warnings
from tqdm import tqdm
import numpy as np
import torch
from deep_utils import color_str

warnings.filterwarnings("ignore")

# def accuracy(y_true, y_predict):



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(net,
          dataloader,
          criterion,
          optimizer,
          scaler,
          epoch,
          config
          ):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=False,
                        desc=color_str(f"Epoch: {epoch}/{config.epochs} Training", color='blue'), position=0,
                        colour='blue'):
        inputs, labels = data
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        # Train with FP16
        with torch.cuda.amp.autocast():
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_tr += loss.item()
        _, preds = torch.max(outputs.data, dim=1)
        correct_count += (preds == torch.max(labels, dim=1)[1]).sum().item()
        n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples
    return acc, loss


def evaluate(net, dataloader, criterion, epoch, config):
    net = net.eval()
    with torch.no_grad():
        loss_, correct_count, n_samples = 0.0, 0.0, 0.0
        for data in tqdm(dataloader, total=len(dataloader), leave=False,
                         desc=color_str(f"Epoch {epoch}/{config.epochs} Evaluation", color='blue'),
                         position=0, colour='blue'):
            inputs, labels = data
            inputs, labels = inputs.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_ += loss.item()
            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == torch.max(labels, dim=1)[1]).sum().item()
            n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_ / n_samples

    return acc, loss
