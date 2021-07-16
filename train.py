import warnings
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import numpy as np
from data.fer2013 import get_dataloaders
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.loops import train, evaluate
from utils.setup_network import setup_network

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net, logger, hps, optimizer, scheduler, num_workers, apply_class_weights):
    trainloader, valloader, testloader = get_dataloaders(path=hps['data_path'],
                                                         bs=hps['bs'],
                                                         num_workers=num_workers,
                                                         crop_size=hps['crop_size'],
                                                         augment=hps['augment'],
                                                         gussian_blur=hps['gussain_blur'],
                                                         rotation_range=hps['rotation_range'],
                                                         combine_val_train=hps['combine_val_train']
                                                         )

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

    start_epoch = hps['restore_epoch'] if hps['restore_epoch'] is not None else hps['start_epoch']
    print("Training", hps['name'], "on", device, " start_epoch: ", start_epoch)

    for epoch in range(start_epoch, hps['n_epochs']):

        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate
        scheduler.step(acc_v)

        if acc_v > logger.best_acc:
            logger.best_acc = acc_v
            logger.best_loss = loss_v
            save(net, logger, hps, optimizer, scheduler, name='best')
            logger.save_plt(hps)

        if (epoch + 1) % hps['save_freq'] == 0:
            # save(net, logger, hps, epoch + 1, optimizer, scheduler)
            logger.save_plt(hps)
        learning_rate = optimizer.param_groups[0]['lr']
        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f/%2.4f %%' % (acc_v, logger.best_acc),
              'Train Loss: %2.4f ' % loss_tr,
              'Val Loss: %2.4f/%2.4f ' % (loss_v, logger.best_loss),
              "LR: %2.6f " % learning_rate,
              sep='\t')
        save(net, logger, hps, optimizer, scheduler, name='last')

    # Calculate performance on test set
    acc_test, loss_test = evaluate(net, testloader, criterion)
    print('Test Accuracy: %2.4f %%' % acc_test,
          'Test Loss: %2.6f' % loss_test,
          sep='\t\t')


if __name__ == "__main__":
    hps = setup_hparams(name='vgg_cbam_extended',
                        restore_epoch=0,
                        network='vgg_cbam_extended',
                        crop_size=40,
                        cbam_blocks=(1, 2, 3, 4),
                        residual_cbam=True,
                        gussain_blur=False,
                        rotation_range=10,
                        augment=True,
                        combine_val_train=True)
    logger, net, optimizer, scheduler = setup_network(hps, get_best=False, device=device)
    run(net, logger, hps, optimizer, scheduler, num_workers=0, apply_class_weights=True)
