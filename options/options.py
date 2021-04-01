import yaml
import os
import os.path as osp
import logging
import sys
sys.path.append('../')
from utils import logger
import shutil

def parse(opt):
    path, name, resume = opt.opt, opt.name, opt.resume

    with open(path, 'r') as fp:
        args = yaml.full_load(fp.read())
    lg = logger(name, 'log/{}.log'.format(name), resume)

    # general settings
    args['name'] = name

    # dataset settings
    for phase, dataset_opt in args['datasets'].items():
        dataset_opt['scale'] = opt.scale
        dataset_opt['split'] = phase
        dataset_opt['patch_size'] = opt.ps
        dataset_opt['batch_size'] = opt.bs
        dataset_opt['dataroot_LR'] = dataset_opt['dataroot_LR'].replace('N', str(opt.scale))

    # network settings
    args['networks']['scale'] = opt.scale

    # create experiment root
    args['solver']['resume'] = resume
    args['solver']['qat'] = opt.qat
    root = osp.join(args['paths']['experiment_root'], name)
    args['paths']['root'] = root
    args['paths']['ckp'] = osp.join(root, 'best_status')
    args['paths']['visual'] = osp.join(root, 'visual')
    args['paths']['state'] = osp.join(root, 'state.pkl')

    if osp.exists(root) and resume==False:
        lg.info('Remove dir: [{}]'.format(root))
        shutil.rmtree(root, True)
    for name, path in args['paths'].items(): 
        if name == 'state':
            continue
        if not osp.exists(path):
            os.mkdir(path)
            lg.info('Create directory: {}'.format(path)) 
    
    # solver
    args['solver']['lr'] = opt.lr
    args['solver']['qat_path'] = opt.qat_path
    args['solver']['resume_path'] = opt.resume_path

    # GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    lg.info('Available gpu: {}'.format(opt.gpu_ids))
    
    return dict_to_nonedict(args), lg


class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        for k,v in opt.items():
            opt[k] = dict_to_nonedict(v)
        return NoneDict(**opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt
