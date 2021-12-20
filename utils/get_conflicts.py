import os
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

from data.fer2013 import get_dataloaders, load_data, RESCostumDataset, CustomDataset, prepare_data
from tqdm import tqdm
from utils.hparams import setup_hparams
from utils.setup_network import setup_network
import cv2
from argparse import ArgumentParser
from deep_utils import tensor_to_image
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from models.resnet50_cbam import CbamBottleNeck
from deep_utils import remove_create

emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


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


def evaluate(net, dataloader, device, name, save_path, thresh, Ncrop=False, ):
    net = net.eval()
    net.drop_prob = 0
    k = 0
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), leave=False, desc='Evaluation:'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.unsqueeze(0)
            # labels = labels.item()
            # forward
            # fuse crops and batchsize
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
            outputs = F.softmax(outputs, dim=1)
            for i in range(len(outputs)):
                acc, pred = outputs[i].topk(1)
                acc = acc.item()
                pred = pred[0].item()
                if pred != labels[i].item() and acc > thresh:
                    path = os.path.join(save_path, str(pred))

                    if not os.path.exists(path):
                        os.mkdir(path)
                    # no need to apply std=255, cause it do the same when converting to uint8
                    img = tensor_to_image(inputs[i])
                    cv2.imwrite(
                        os.path.join(path,
                                     f"prd_{emotion_mapping[pred]}_lbl_{emotion_mapping[labels[i].item()]}_{acc:2f}_{k}.jpg"),
                        img)
                    k += 1


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--model-name", default='vgg', type=str, help='The model that you want to run')
    parser.add_argument("--batch-size", default=2, type=int, help='batch size')
    parser.add_argument("--cut-mix", action='store_true', help='uses cut-mix augmentation, default is false')
    parser.add_argument("--residual-cbam", action='store_true', help='Adds residual cbam blocks to the architecture')
    parser.add_argument('--augment', action='store_true', help='applies augmentation methods, default is false')
    parser.add_argument('--n-epochs', type=int, default=100, help='How many epochs for training')
    parser.add_argument('--dataset-dir', type=str, default='datasets/fer2013.csv', help='path to the dataset')
    parser.add_argument('--n-workers', type=int, default=4, help="number of workers for dataloader")
    parser.add_argument('--crop-size', type=int, default=40, help="crop size, for vgg use 40")
    parser.add_argument('--model-path', type=str, default='checkpoints', help='model-path directory.')
    parser.add_argument('--label-path', type=str, default='vgg-labels', help='label_path')
    parser.add_argument('--restore-epoch', type=int, default=0, help='restore model trained before')
    parser.add_argument('--restore-path', type=str, default='vgg-best/vgg_best.pt', help='path to restore ')
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hps = setup_hparams(name=args.model_name,
                        network=args.model_name,
                        crop_size=args.crop_size,
                        bs=args.batch_size,
                        cbam_blocks=(0, 1, 2, 3, 4),
                        residual_cbam=args.residual_cbam,
                        gussain_blur=False,
                        rotation_range=20,
                        augment=args.augment,
                        combine_val_train=False,
                        cutmix=args.cut_mix,
                        cutmix_prop=0.5,
                        restore_epoch=args.restore_epoch,
                        beta=1,
                        data_path=args.dataset_dir,
                        model_save_dir='..',
                        n_epochs=args.n_epochs,
                        n_workers=args.n_workers,
                        restore_path=args.restore_path
                        )

    # build network
    logger, net, optimizer, scheduler, monitor_val = setup_network(hps, device=device)
    net = net.to(device)

    fer2013, emotion_mapping = load_data("../datasets/fer2013.csv")
    mu, st = 0, 255

    test_transform = transforms.Compose([
        # transforms.Pad(2 if crop_size == 48 else 0),
        transforms.CenterCrop(args.crop_size),
        transforms.PILToTensor(),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(
        #     lambda tensors: torch.stack([)(t) for t in tensors])),
        transforms.Normalize(mean=(mu,), std=(st,))
    ])
    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    train = CustomDataset(xtrain, ytrain, test_transform)
    trainloader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    remove_create(args.label_path)
    evaluate(net,
             trainloader,
             device,
             name='Train',
             save_path=args.label_path,
             Ncrop=False,
             thresh=0.8)
