from models import vgg, efn, vgg_attention, vgg_bam, vgg_cbam, vgg_cbam_modified, vgg_cbam_extended, resnet50_cbam, \
    resnet50

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


def get_model(model_name, model_config, device=None):
    model = nets[model_name](model_config)
    if device is not None:
        model = model.to(device)
    return model
