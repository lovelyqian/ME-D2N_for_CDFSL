# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
identity = lambda x:x


class SimpleDataset:
  def __init__(self, data_file, transform, target_transform=identity):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    
    image_path = image_path[:12]+'2'+image_path[13:]
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])


# added by fuyuqian in 20210108
class RandomLabeledTargetDataset:
  def __init__(self, data_file,data_file_miniImagenet, transform, target_transform=identity):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    with open(data_file_miniImagenet, 'r') as f_miniI:
      self.meta_miniImagenet = json.load(f_miniI)
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self,i):
    idx = random.randint(0, len(self.meta['image_names'])-1)
    image_path = self.meta['image_names'][idx]
   
    image_path = image_path[:12]+'2'+image_path[13:] 
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][idx])
    return img, target

  def __len__(self):
    #return len(self.meta['image_names'])
    return len(self.meta_miniImagenet['image_names'])


class SetDataset:
  def __init__(self, data_file, batch_size, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

    self.cl_list = np.unique(self.meta['image_labels']).tolist()
    #print('dataset:', 'SetDataset:', 'cl_list:', self.cl_list)

    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []

    for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
      self.sub_meta[y].append(x)

    #print('dataset:', 'SetDataset:', 'sub_meta:', len(self.sub_meta))
    #for i in range(len(self.sub_meta)):
        #print(i, len(self.sub_meta[i]))


    self.sub_dataloader = []
    sub_data_loader_params = dict(batch_size = batch_size,
        shuffle = True,
        num_workers = 0, #use main thread only or may receive multiple batches
        pin_memory = False)
    for cl in self.cl_list:
      #print('dataset:', 'SetDataset:', 'cl:', cl)
      sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
      self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    #print('dataset:', 'SetDataset:', 'len:', len(self.cl_list))
    return len(self.cl_list)


class MultiSetDataset:
  def __init__(self, data_files, batch_size, transform):
    self.cl_list = np.array([])
    self.sub_dataloader = []
    self.n_classes = []
    for data_file in data_files:
      with open(data_file, 'r') as f:
        meta = json.load(f)
      cl_list = np.unique(meta['image_labels']).tolist()
      self.cl_list = np.concatenate((self.cl_list, cl_list))

      sub_meta = {}
      for cl in cl_list:
        sub_meta[cl] = []

      for x,y in zip(meta['image_names'], meta['image_labels']):
        sub_meta[y].append(x)

      sub_data_loader_params = dict(batch_size = batch_size,
          shuffle = True,
          num_workers = 0, #use main thread only or may receive multiple batches
          pin_memory = False)
      for cl in cl_list:
        sub_dataset = SubDataset(sub_meta[cl], cl, transform = transform, min_size=batch_size)
        self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
      self.n_classes.append(len(cl_list))

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)

  def lens(self):
    return self.n_classes


class SubDataset:
  def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform
    self.target_transform = target_transform
    #print('dataset:', 'SubDatset:', 'sub_meta:', self.sub_meta)
    if len(self.sub_meta) < min_size:
      #print('dataset:', 'SubDataset:', 'len of self_meta:', len(self.sub_meta),' < 50')
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      #print('dataset:', 'SubDataset:', 'idxs:', idxs)
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()
      #print('dataset:', 'SubDataset:', 'sub_meat:', self.sub_meta)

  def __getitem__(self,i):
    #print('sub dataset:')
    image_path = os.path.join( self.sub_meta[i])
    #print(image_path)
    image_path = image_path[:12]+'2'+image_path[13:] 

    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.cl)
    #print('img:',img.size(), 'target:', target)
    return img, target

  def __len__(self):
    return len(self.sub_meta)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_classes)[:self.n_way]


class MultiEpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes
    self.n_domains = len(n_classes)

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    domain_list = [i%self.n_domains for i in range(self.n_episodes)]
    random.shuffle(domain_list)
    for i in range(self.n_episodes):
      domain_idx = domain_list[i]
      start_idx = sum(self.n_classes[:domain_idx])
      yield torch.randperm(self.n_classes[domain_idx])[:self.n_way] + start_idx
