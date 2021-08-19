import os
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from data.fer2013 import get_dataloaders
from tqdm import tqdm
from utils.hparams import setup_hparams
from utils.setup_network import setup_network
import torch.nn.functional as F
from models.resnet50_cbam import CbamBottleNeck


def correct_count(output, target, topk=(1,)):
    """Computes the top k corrrect count for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


def evaluate(net, dataloader, criterion, device, name, save_path, Ncrop):
    net = net.eval()
    if name == "Test":
        net.drop_prob = 0

    loss_tr, n_samples = 0.0, 0.0

    y_pred = []
    y_gt = []

    correct_count1 = 0
    correct_count2 = 0
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), leave=False, desc='Evaluation:'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # fuse crops and batchsize
            if Ncrop:
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)
            else:
                bs, c, h, w = inputs.shape
                ncrops = 1

            # forward
            outputs = net(inputs)

            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops

            loss = criterion(outputs, labels)

            # calculate performance metrics
            loss_tr += loss.item()

            # accuracy
            counts = correct_count(outputs, labels, topk=(1, 2))
            correct_count1 += counts[0].item()
            correct_count2 += counts[1].item()

            _, preds = torch.max(outputs.data, 1)
            preds = preds.to("cpu")
            labels = labels.to("cpu")
            n_samples += labels.size(0)

            y_pred.extend(pred.item() for pred in preds)
            y_gt.extend(y.item() for y in labels)

    acc1 = 100 * correct_count1 / n_samples
    acc2 = 100 * correct_count2 / n_samples

    loss = loss_tr / n_samples
    print_text = f"{name}\n"
    print_text += "--------------------------------------------------------\n"
    print_text += "Top 1 Accuracy: %2.6f %%" % acc1 + "\n"
    print_text += "Top 2 Accuracy: %2.6f %%" % acc2 + "\n"

    print_text += "Loss: %2.6f" % loss + "\n"
    print_text += "Precision: %2.6f" % precision_score(y_gt, y_pred, average='micro') + "\n"
    print_text += "Recall: %2.6f" % recall_score(y_gt, y_pred, average='micro') + "\n"
    print_text += "F1 Score: %2.6f" % f1_score(y_gt, y_pred, average='micro') + "\n"
    print_text += "Confusion Matrix:\n" + str(confusion_matrix(y_gt, y_pred)) + '\n' + "\n"
    print(print_text)
    with open(os.path.join(save_path, name + '.txt'), mode='w') as f:
        f.write(print_text)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hps = setup_hparams('vgg_cbam', network='vgg_cbam',
                        crop_size=40, data_path='../data',
                        cbam_blocks=(1, 2, 3), residual_cbam=False,
                        gussian_blur=False, rotation_range=20,
                        combine_val_train=False)

    # build network
    logger, net, optimizer, scheduler = setup_network(hps, get_best=True, device=device)
    net = net.to(device)

    print(net)

    criterion = nn.CrossEntropyLoss()

    # Get data with no augmentation
    trainloader, valloader, testloader = get_dataloaders(augment=False,
                                                         bs=hps['bs'],
                                                         num_workers=0,
                                                         crop_size=40,
                                                         path=hps['data_path'],
                                                         gussian_blur=hps['gussian_blur'],
                                                         rotation_range=hps['rotation_range'],
                                                         combine_val_train=hps['combine_val_train'])

    evaluate(net,
             trainloader,
             criterion,
             device,
             name='Train',
             save_path=hps['model_save_dir'])

    evaluate(net,
             valloader,
             criterion,
             device,
             name='Val',
             save_path=hps['model_save_dir'])

    evaluate(net,
             testloader,
             criterion,
             device,
             name='Test',
             save_path=hps['model_save_dir'])
