import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import vgg, efn, vgg_attention, vgg_bam, vgg_cbam, vgg_cbam_modified, vgg_cbam_extended, resnet50_cbam, \
    resnet50
from utils.checkpoint import restore
from utils.logger import Logger
from utils.radam import RAdam

nets = {
    'vgg': vgg.Vgg,
    'efn': efn.EfficientNet,
    'vgg_attention': vgg_attention.Vgg,
    'vgg_bam': vgg_bam.VggBAM,
    'vgg_cbam': vgg_cbam.VggCBAM,
    'vgg_cbam_modified': vgg_cbam_modified.VggCBAM,
    'vgg_cbam_extended': vgg_cbam_extended.VggCBAM,
    'resnet50': resnet50.Resnet,
    'resnet50_cbam': resnet50_cbam.Resnet
}


def setup_network(hps, get_best, device):
    net = nets[hps['network']](crop=hps['crop_size'], **hps)
    if hps['name'] == 'resnet50_cbam':
        net = nets[hps['network']](**hps)
    else:
        net = net.to(device)
    if 'optim' in hps and hps['optim'] == 'radam':
        optimizer = RAdam(net.parameters(), lr=hps['lr'], weight_decay=hps['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-6, patience=2, verbose=True)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=hps['lr'], momentum=0.9, nesterov=True, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=hps['lr_decay'], patience=5, verbose=True)

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch'] or get_best:
        restore(net, logger, hps, optimizer, scheduler, get_best)

    return logger, net, optimizer, scheduler
