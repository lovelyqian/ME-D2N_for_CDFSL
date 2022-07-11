import torch
import os
import h5py

from methods.backbone import model_dict
from data.datamgr import SimpleDataManager
from options import parse_args, get_best_file, get_assigned_file

from methods.student_MED2N import GnnNetStudent
import data.feature_loader as feat_loader
import random
import numpy as np

import torch.nn.functional as F

# extract and save image features
def save_features(model, model2, data_loader, featurefile, featurefile2, data_flag):
  f = h5py.File(featurefile, 'w')
  f2 = h5py.File(featurefile2, 'w')
  max_count = len(data_loader)*data_loader.batch_size
  all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
  all_labels2 = f2.create_dataset('all_labels',(max_count,), dtype='i')
  all_feats=None
  all_feats2=None
  count=0
  count2 =0
  for i, (x,y) in enumerate(data_loader):
    if (i % 10) == 0:
      print('    {:d}/{:d}'.format(i, len(data_loader)))
    x = x.cuda()
    x_fea_block1 = model.feature.forward_block1(x)
    x_fea_block2 = model.feature.forward_block2(x_fea_block1)
    x_fea_block3 = model.feature.forward_block3(x_fea_block2)
    x_fea_block3 = model.mask_layer3(x_fea_block3, data_flag)
    x_fea_block4 = model.feature.forward_block4(x_fea_block3)
    x_fea_block4 = model.mask_layer4(x_fea_block4, data_flag)
    feats = model.feature.forward_rest(x_fea_block4)

    # model2
    feats2 = model2(x)

    if all_feats is None:
      all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
    all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
    all_labels[count:count+feats.size(0)] = y.cpu().numpy()
    count = count + feats.size(0)

    if all_feats2 is None:
      all_feats2 = f2.create_dataset('all_feats', [max_count] + list( feats2.size()[1:]) , dtype='f')
    all_feats2[count2:count2+feats2.size(0)] = feats2.data.cpu().numpy()
    all_labels2[count2:count2+feats2.size(0)] = y.cpu().numpy()
    count2 = count2 + feats2.size(0)

  count_var = f.create_dataset('count', (1,), dtype='i')
  count_var[0] = count
  f.close()

  count_var2 = f2.create_dataset('count', (1,), dtype='i')
  count_var2[0] = count2
  f2.close()

# evaluate using features
def feature_evaluation(cl_data_file, cl_data_file2, model, model2, n_way = 5, n_support = 5, n_query = 15):
  #print(cl_data_file.keys(), cl_data_file.keys())
  class_list = cl_data_file.keys()
  select_class = random.sample(class_list,n_way)
  z_all  = []
  z_all2 = []
  for cl in select_class:
    img_feat = cl_data_file[cl]
    img_feat2 = cl_data_file2[cl]
    perm_ids = np.random.permutation(len(img_feat)).tolist()
    z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )
    z_all2.append( [ np.squeeze( img_feat2[perm_ids[i]]) for i in range(n_support+n_query) ] )

  z_all = torch.from_numpy(np.array(z_all) )
  z_all2 = torch.from_numpy(np.array(z_all2) )
  model.n_query = n_query
  model2.n_query = n_query
  scores  = model.set_forward(z_all, is_feature = True)
  scores2 = model2.set_forward(z_all2, is_feature = True)
  scores = (scores + scores2)/2.0
  pred = scores.data.cpu().numpy().argmax(axis = 1)
  y = np.repeat(range( n_way ), n_query )
  acc = np.mean(pred == y)*100
  return acc

# --- main ---
if __name__ == '__main__':
  # seed
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # parse argument
  params = parse_args('test')
  print('Testing! {} shots on {} dataset with {} epochs of {}'.format(params.n_shot, params.dataset, params.save_epoch, params.name))
  remove_featurefile = True

  print('\nStage 1: saving features')
  # dataset
  print('  build dataset')
  image_size = 224
  split = params.split
  loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
  print('load file:', loadfile)
  datamgr         = SimpleDataManager(image_size, batch_size = 64)
  data_loader      = datamgr.get_data_loader(loadfile, aug = False)

  print('  build feature encoder')
  checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if params.save_epoch != -1:
    modelfile   = get_assigned_file(checkpoint_dir,params.save_epoch)
  else:
    modelfile   = get_best_file(checkpoint_dir)
  
  # feature encoder
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
  model = GnnNetStudent(model_dict[params.model], target_set = params.target_set, **few_shot_params)
  model = model.cuda()
  tmp = torch.load(modelfile)
  state = tmp['state']
  model.load_state_dict(state)
  model.eval()

  # model2
  model2 = GnnNetStudent(model_dict[params.model], target_set = params.target_set, **few_shot_params)
  model2 = model2.cuda()
  model2.load_state_dict(state)
  model2.eval()

  # save feature file
  print('  extract and save features...')
  if params.save_epoch != -1:
    featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_epoch)+ ".hdf5")
    featurefile2 = os.path.join( checkpoint_dir.replace("checkpoints","features"),  split + "_" + str(params.save_epoch)+ "2.hdf5") 
  else:
    featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")
  dirname = os.path.dirname(featurefile)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)

  if(params.dataset=='miniImagenet'):
    domain_flag = 'S'
  else:
    domain_flag = 'A'
  save_features(model, model2, data_loader, featurefile, featurefile2, domain_flag)

  print('\nStage 2: evaluate')
  acc_all = []
  iter_num = 1000
  # load feature file
  print('  load saved feature file')
  cl_data_file = feat_loader.init_loader(featurefile)
  cl_data_file2 = feat_loader.init_loader(featurefile2)

  # start evaluate
  print('  evaluate')
  for i in range(iter_num):
    acc = feature_evaluation(cl_data_file, cl_data_file2, model,model2, n_query=15, **few_shot_params)
    acc_all.append(acc)

  # statics
  print('  get statics')
  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

  # remove feature files [optional]
  if remove_featurefile:
    os.remove(featurefile)
