from os.path import join
from argparse import ArgumentParser
import warnings
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import numpy as np
from data.fer2013 import get_dataloaders
from models.resnet50_cbam import CbamBottleNeck
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.loops import train, evaluate
from utils.setup_network import setup_network
from deep_utils import ModelCheckPoint

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net, logger, hps, optimizer, scheduler, num_workers, apply_class_weights, model_path, monitor_val):
    model_checkpointer = ModelCheckPoint(join(model_path, hps['network'], hps['network'] + ".pt"),
                                         net,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         monitor_val=monitor_val
                                         )
    trainloader, valloader, testloader = get_dataloaders(path=hps['data_path'],
                                                         bs=hps['bs'],
                                                         num_workers=num_workers,
                                                         crop_size=hps['crop_size'],
                                                         augment=hps['augment'],
                                                         gussian_blur=hps['gussain_blur'],
                                                         rotation_range=hps['rotation_range'],
                                                         combine_val_train=hps['combine_val_train'],
                                                         cutmix=hps['cutmix'],
                                                         network=hps['network'] if hps[
                                                                                       'network'] == 'resnet50_cbam' else False,
                                                         imagesize=hps[
                                                             'imagesize'] if 'imagesize' in hps else False,
                                                         NoF=hps['NoF'] if 'NoF' in hps else False
                                                         )
    if 'use_dropblock' in hps:
        net.n_steps = len(trainloader) * hps['n_epochs']
        net.dropblock.drop_values = np.linspace(start=0.1, stop=net.drop_prob, num=net.n_steps)

    net = net.to(device)
    scaler = GradScaler()

    if apply_class_weights:
        class_weights = [
            1.02660468,
            9.40661861,
            1.00104606,
            0.56843877,
            0.84912748,
            1.29337298,
            0.82603942,
        ]
        class_weights = torch.FloatTensor(np.array(class_weights)).to(device)
        criterion = nn.CrossEntropyLoss(class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = hps['restore_epoch'] if hps['restore_epoch'] else hps['start_epoch']
    print("[INFO] Training", hps['name'], "on", device, "restore_epoch: ", start_epoch)

    for epoch in range(start_epoch, hps['n_epochs']):

        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler, cutmix_prop=hps['cutmix_prop'],
                                beta=hps['beta'], Ncrop=hps['Ncrop_train'] if 'Ncrop_train' in hps else True)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        if valloader is not None:
            acc_v, loss_v = evaluate(net, valloader, criterion, Ncrop=hps['Ncrop_val'] if 'Ncrop_val' in hps else True)
            model_checkpointer(loss_v)
            logger.loss_val.append(loss_v)
            logger.acc_val.append(acc_v)

            # Update learning rate
            if hps['network'] == 'resnet50_cbam':
                scheduler.step(100 - acc_v)
            else:
                scheduler.step(acc_v)

            if acc_v > logger.best_acc:
                logger.best_acc = acc_v
                logger.best_loss = loss_v
                logger.save_plt(hps)

            if (epoch + 1) % hps['save_freq'] == 0:
                logger.save_plt(hps)
            learning_rate = optimizer.param_groups[0]['lr']
            print('Epoch %2d' % (epoch + 1),
                  'Train Accuracy: %2.4f %%' % acc_tr,
                  'Val Accuracy: %2.4f/%2.4f %%' % (acc_v, logger.best_acc),
                  'Train Loss: %2.4f ' % loss_tr,
                  'Val Loss: %2.4f/%2.4f ' % (loss_v, logger.best_loss),
                  "LR: %2.6f " % learning_rate,
                  sep='\t')
        else:
            if epoch >= 20 and epoch % 10 == 0:
                optimizer.param_groups[0]['lr'] /= 10

            learning_rate = optimizer.param_groups[0]['lr']

            logger.save_plt(hps)
            if (epoch + 1) % hps['save_freq'] == 0:
                logger.save_plt(hps)

            print('Epoch %2d' % (epoch + 1),
                  'Train Accuracy: %2.4f %%' % acc_tr,
                  'Train Loss: %2.4f ' % loss_tr,
                  "LR: %2.6f " % learning_rate,
                  sep='\t')
            model_checkpointer(loss_tr)
    # Calculate performance on test set
    acc_test, loss_test = evaluate(net, testloader, criterion)
    print('Test Accuracy: %2.4f %%' % acc_test,
          'Test Loss: %2.6f' % loss_test,
          sep='\t\t')


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
    parser.add_argument('--restore-epoch', type=int, default=0, help='restore model trained before, '
                                                                     'default is 0 which indicates no restoring')
    parser.add_argument('--restore-path', type=str, default=None, help='restore model path, default is None!')
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
                        restore_path=args.restore_path,
                        beta=1,
                        data_path=args.dataset_dir,
                        model_save_dir='.',
                        n_epochs=args.n_epochs,
                        n_workers=args.n_workers
                        )
    logger, net, optimizer, scheduler, monitor_val = setup_network(hps, device=device)

    run(
        net,
        logger,
        hps,
        optimizer,
        scheduler,
        num_workers=args.n_workers,
        apply_class_weights=True,
        model_path=args.model_path,
        monitor_val=monitor_val
    )
