import torch
import torch.nn as nn
import numpy as np
from methods.meta_template_student_MED2N import MetaTemplate
from methods.gnn import GNN_nl
from methods import backbone
from methods.learnablemask import LearnableMaskLayer
 
class_categories={}
class_categories['source']=64
class_categories['cub']=99
class_categories['cars']=97
class_categories['places']=182
class_categories['plantae']=99
EPS=0.00001


class GnnNetStudent(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support, tf_path=None, target_set='None'):
    super(GnnNetStudent, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'GnnNet'

    # define global fc classifiers
    self.classifier_source = nn.Linear(self.feat_dim, class_categories['source'])
    self.classifier_target = nn.Linear(self.feat_dim, class_categories[target_set])

    # define learnablMaskLayers
    mask_thred = 0.0
    self.mask_layer1 = LearnableMaskLayer(64)
    self.mask_layer2 = LearnableMaskLayer(128)
    self.mask_layer3 = LearnableMaskLayer(256)
    self.mask_layer4 = LearnableMaskLayer(512)

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.support_label = self.support_label.cuda()
    self.classifier_source.cuda()
    self.classifier_target.cuda()
    self.mask_layer1.cuda()
    self.mask_layer2.cuda()
    self.mask_layer3.cuda()
    self.mask_layer4.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))
    #print('z:', z.size())
    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    #print('z_stack:', 'len:', len(z_stack), 'z_stack[0]:', z_stack[0].size())
    scores = self.forward_gnn(z_stack)
    return scores


  def get_classification_scores(self, z, classifier):
    z_norm = torch.norm(z, p=2, dim=1).unsqueeze(1).expand_as(z)
    z_normalized = z.div(z_norm + EPS)
    L_norm = torch.norm(classifier.weight.data, p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
    classifier.weight.data = classifier.weight.data.div(L_norm + EPS)
    cos_dist = classifier(z_normalized)
    cos_fac = 1.0
    scores = cos_fac * cos_dist
    return scores


  def set_forward_withGlobalCls(self,x,data_flag, is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      x_fea_block1 = self.feature.forward_block1(x)
      x_fea_block2 = self.feature.forward_block2(x_fea_block1)
      x_fea_block3 = self.feature.forward_block3(x_fea_block2)
      #mask
      x_fea_block3 = self.mask_layer3(x_fea_block3, data_flag)
      x_fea_block4 = self.feature.forward_block4(x_fea_block3)
      #mask
      x_fea_block4 = self.mask_layer4(x_fea_block4, data_flag)
      x_fea = self.feature.forward_rest(x_fea_block4)
      z = self.fc(x_fea)
      z = z.view(self.n_way, -1, z.size(1))
    # for FSL- GNN classifer
    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    fsl_scores = self.forward_gnn(z_stack)
 
    # for FC - global FC classifier
    classifier_mode = 'v2'
    if(data_flag=='S'):
      if(classifier_mode == 'v1'):
        cls_scores = self.classifier_source(x_fea)
      elif(classifier_mode == 'v2'):
       cls_scores = self.get_classification_scores(x_fea, self.classifier_source)
    elif(data_flag=='A'):
      if(classifier_mode == 'v1'):
        cls_scores = self.classifier_target(x_fea)
      elif(classifier_mode == 'v2'):
        cls_scores = self.get_classification_scores(x_fea, self.classifier_target)
    return fsl_scores, cls_scores


  def set_forward_withGlobalCls_STD(self,x, data_flag, is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      x_fea = self.feature(x)
      z = self.fc(x_fea)
      z = z.view(self.n_way, -1, z.size(1))
    #for FSL- GNN classifer
    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    fsl_scores = self.forward_gnn(z_stack)

    # for FC - global FC classifier
    classifier_mode = 'v2'
    if(data_flag=='S'):
      if(classifier_mode == 'v1'):
        cls_scores = self.classifier_source(x_fea)
      elif(classifier_mode == 'v2'):
       cls_scores = self.get_classification_scores(x_fea, self.classifier_source)
    elif(data_flag=='A'):
      if(classifier_mode == 'v1'):
        cls_scores = self.classifier_target(x_fea)
      elif(classifier_mode == 'v2'):
        cls_scores = self.get_classification_scores(x_fea, self.classifier_target)
    return fsl_scores, cls_scores


  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)
    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

  def set_forward_loss_withGlobalCls(self, x, y_cls, data_flag):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    fsl_scores, cls_scores = self.set_forward_withGlobalCls(x, data_flag)
    fsl_loss = self.loss_fn(fsl_scores, y_query)
    y_cls = y_cls.view(cls_scores.size()[0]).cuda()
    cls_loss = self.loss_fn(cls_scores, y_cls)
    return fsl_scores, fsl_loss, cls_loss

  def set_forward_loss_withGlobalCls_STD(self, x, y_cls, data_flag):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    fsl_scores, cls_scores = self.set_forward_withGlobalCls_STD(x, data_flag)
    fsl_loss = self.loss_fn(fsl_scores, y_query)
    y_cls = y_cls.view(cls_scores.size()[0]).cuda()
    cls_loss = self.loss_fn(cls_scores, y_cls)
    return fsl_scores, fsl_loss, cls_loss


