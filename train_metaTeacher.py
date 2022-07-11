import numpy as np
import torch
import torch.optim
import os
import random

from methods.backbone import model_dict
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.gnnnet import GnnNet
from options import parse_args, get_resume_file, load_warmup_state


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(model.parameters())
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc = 0
  total_it = 0

  # start
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    total_it = model.train_loop(epoch, base_loader,  optimizer, total_it) 
    model.eval()

    acc = model.test_loop( val_loader)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model



# --- main function ---
if __name__=='__main__':
  # set numpy random seed
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # parser argument
  params = parse_args('train')
  print(params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader and model
  print('\n--- prepare dataloader ---')
  val_file   = os.path.join(params.data_dir, 'miniImagenet', 'val.json')
  
  image_size  = 224
  if params.modelType == 'pretrain':
    print('  pre-training the feature encoder {}'.format(params.model))
    assert(params.dataset == 'miniImagenet')
    base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
    base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    training_loader     = base_datamgr.get_data_loader( base_file , aug=params.train_aug )
    val_datamgr     = SimpleDataManager(image_size, batch_size=64)
    val_loader      = val_datamgr.get_data_loader(val_file, aug=False)
    model           = BaselineTrain(model_dict[params.model], params.num_classes, tf_path=params.tf_dir)

  elif params.modelType == 'St-Net' or params.modelType == 'Tt-Net':
    if(params.modelType == 'St-Net'):
      print('meta-training the St-Net using {}'.format(params.dataset))
      assert(params.dataset == 'miniImagenet')
      training_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
    elif(params.modelType == 'Tt-Net'):
      print('meta-training the Tt-Net using {} with num_target as {}'.format(params.dataset, params.target_num_label))
      assert(params.dataset in ['cub', 'cars', 'places', 'plantae'])
      training_file =  'output/labled_base_' + params.dataset + '_' + str(params.target_num_label) + '.json'
    n_query = max(1, int(16* params.test_n_way/params.train_n_way))
    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
    base_datamgr  = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    training_loader = base_datamgr.get_data_loader( training_file , aug = params.train_aug )
    test_few_shot_params  = dict(n_way = params.test_n_way, n_support = params.n_shot)
    val_datamgr   = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_loader    = val_datamgr.get_data_loader( val_file, aug = False)
    model = GnnNet( model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)
    
  model = model.cuda()

  #load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  elif params.modelType != 'pretrain':
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup))
    model.feature.load_state_dict(state, strict=False)

  # training
  print('\n--- start the training ---')
  model = train(training_loader, val_loader,  model, start_epoch, stop_epoch, params)
