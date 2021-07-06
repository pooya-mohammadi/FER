import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import vgg, efn, vgg_attention
from utils.checkpoint import restore
from utils.logger import Logger

nets = {
    'vgg': vgg.Vgg,
    'efn': efn.EfficientNet,
    'vgg_attention': vgg_attention.Vgg
}


def setup_network(hps, get_best, device):
    net = nets[hps['network']]()
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=hps['lr'], momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=hps['lr_decay'], patience=5, verbose=True)

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch'] or get_best:
        restore(net, logger, hps, optimizer, scheduler, get_best)

    return logger, net, optimizer, scheduler
