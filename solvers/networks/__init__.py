from .base7 import base7

def create_model(opt):
    which_model = opt['which_model']
    scale = opt['scale']
    in_channels = opt['in_channels']
    out_channels = opt['out_channels']

    if which_model == 'base7':
        model = base7(scale, in_channels, opt['num_fea'], opt['m'], out_channels)
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))

    return model
