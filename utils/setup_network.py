import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import vgg, efn
from utils.checkpoint import restore
from utils.logger import Logger

nets = {
    'vgg': vgg.Vgg,
    'efn': efn.EfficientNet
}


def setup_network(hps):
    net = nets[hps['network']]()
    optimizer = torch.optim.SGD(net.parameters(), lr=hps['lr'], momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps, optimizer, scheduler)

    return logger, net, optimizer, scheduler
