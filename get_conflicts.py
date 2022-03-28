import os
import torch
from torch.utils.data import DataLoader
from data.fer2013 import get_data_loaders, load_data, CustomDataset, prepare_data
from tqdm import tqdm

from models import get_model
from utils.config_utils import Config
import cv2
from argparse import ArgumentParser
from deep_utils import tensor_to_image, log_print, easy_argparse
from torchvision import transforms
import torch.nn.functional as F
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


def evaluate(net, dataloader, device, save_path):
    net = net.eval()
    n_correct = 0
    for data in tqdm(dataloader, total=len(dataloader), leave=False, desc='Evaluation:'):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        with torch.no_grad():
            outputs = net(inputs)
        cls_accuracies, cls_indices = torch.max(F.softmax(outputs, dim=1), dim=1)
        n_correct += (cls_indices == labels).sum()
        for i, (acc, cls_index) in enumerate(zip(cls_accuracies, cls_indices)):
            acc = acc.item()
            cls_index = cls_index.item()

            path = os.path.join(save_path, str(cls_index))

            os.makedirs(path, exist_ok=True)
            # no need to apply std=255, cause it do the same when converting to uint8
            if cls_index == labels[i]:
                img = tensor_to_image(inputs[i])
                cv2.imwrite(os.path.join(path,
                                         f"prd_{emotion_mapping[cls_index]}_lbl_{emotion_mapping[labels[i].item()]}_{acc:2f}.jpg"),
                            img)
    print(f"accuracy: {n_correct.item() / len(dataloader.dataset)}")


def parser_args():
    args = easy_argparse(
        dict(name="--config_path", default='configs/vgg.yml', type=str, help='the path to config file!'),
        dict(name="--model_path", default="configs/vgg/vgg_main/weights/model_last.pt", required=True,
             help="path to the model"),
        dict(name="--dataset", default="test", type=str, help="dataset which results should be saved!"),
        dict(name="--output_dir", default="datasets/out", help="path to the output directory"),
        dict(name="--num_workers", default=0, help="number of dataloader workers, default is 0")
    )
    return args


if __name__ == "__main__":
    args = parser_args()

    # load model
    config = Config.load_config(args.config_path)
    config.dataset.num_workers = args.num_workers

    model = get_model(model_name=config.model.name, device=config.device, model_config=config.model, weight_path=args.model_path)

    # get data_loader
    data_loader = get_data_loaders(config, dataloader_name=args.dataset)

    remove_create(args.output_dir)
    evaluate(net=model, dataloader=data_loader, device=config.device, save_path=args.output_dir)
