import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
import torch.nn.functional as F 

class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
    super(MetaTemplate, self).__init__()
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.change_way = change_way  #some methods allow different_way classification during training and test
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  @abstractmethod
  def set_forward(self,x,is_feature):
    pass

  @abstractmethod
  def set_forward_loss(self, x):
    pass

  def forward(self,x):
    out  = self.feature.forward(x)
    return out

  def parse_feature(self,x,is_feature):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature.forward(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]

    return z_support, z_query

  def correct(self, x):
    scores, loss = self.set_forward_loss(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)

  def train_loop(self, epoch, S_train_loader, A_train_loader, Expert_S, Expert_A, optimizer, total_it):
    print_freq = len(S_train_loader) // 10
    avg_loss=0
    for ((i, (S_x, S_y_global)), (i, (A_x, A_y_global))) in zip(enumerate(S_train_loader), enumerate(A_train_loader)):
      self.n_query = S_x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = S_x.size(0)
      optimizer.zero_grad()

      #hyper-parameters
      k1 = 0.2      #S:A
      k2_fsl = 0.05 #loss_fsl
      k3_cls = 0.05 #loss-global-cls
      k4_std = 0.2  #loss-STD

      # forward Teachers
      Expert_S.n_way = self.n_way
      Expert_S.n_query = self.n_query
      Expert_A.n_way = self.n_way
      Expert_A.n_query = self.n_query
      Expert_S_scores, Expert_S_loss = Expert_S.set_forward_loss(S_x)
      Expert_A_scores, Expert_A_loss = Expert_A.set_forward_loss(A_x)
     
      # DSG: Forward Student wth Mask
      S_scores, S_loss_fsl, S_loss_cls = self.set_forward_loss_withGlobalCls(S_x, S_y_global, data_flag='S')
      A_scores, A_loss_fsl, A_loss_cls = self.set_forward_loss_withGlobalCls(A_x, A_y_global, data_flag='A')

      # DSG: KD loss
      T = 5.0
      lossKD = nn.KLDivLoss()
      S_loss_KD = lossKD(F.log_softmax(S_scores / T, dim=1), F.softmax(Expert_S_scores / T, dim=1))
      A_loss_KD = lossKD(F.log_softmax(A_scores / T, dim=1), F.softmax(Expert_A_scores / T, dim=1))
     
      # DSG: loss
      S_loss = S_loss_KD + k2_fsl * S_loss_fsl + k3_cls * S_loss_cls
      A_loss = A_loss_KD + k2_fsl * A_loss_fsl + k3_cls * A_loss_cls
      loss_Masked = k1 * S_loss + (1-k1) * A_loss

      # STD: Forward Student with STD (not masked)
      S_scores_STD, S_loss_fsl_STD, S_loss_cls_STD = self.set_forward_loss_withGlobalCls_STD(S_x, S_y_global, data_flag='S')
      A_scores_STD, A_loss_fsl_STD, A_loss_cls_STD = self.set_forward_loss_withGlobalCls_STD(A_x, A_y_global, data_flag='A')
  
      # STD: KD loss
      T = 5.0
      lossKD = nn.KLDivLoss()
      S_loss_KD_STD = lossKD(F.log_softmax(S_scores_STD / T, dim=1), F.softmax(Expert_S_scores / T, dim=1))
      A_loss_KD_STD = lossKD(F.log_softmax(A_scores_STD / T, dim=1), F.softmax(Expert_A_scores / T, dim=1))

      # STD: loss
      S_loss_STD = S_loss_KD_STD + k2_fsl * S_loss_fsl_STD + k3_cls * S_loss_cls_STD
      A_loss_STD = A_loss_KD_STD + k2_fsl * A_loss_fsl_STD + k3_cls * A_loss_cls_STD
      loss_STD = k1 * S_loss_STD + (1-k1) * A_loss_STD

      # final loss
      loss = loss_Masked + k4_std * loss_STD 
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(S_train_loader), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/total_loss', loss.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/loss_Masked', loss_Masked.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/loss_STD', loss_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss', S_loss.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss', A_loss.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_FSL', S_loss_fsl.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_KD', S_loss_KD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_cls', S_loss_cls.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_FSL', A_loss_fsl.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_KD', A_loss_KD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_cls', A_loss_cls.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_STD', S_loss_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_STD', A_loss_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_FSL_STD', S_loss_fsl_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_KD_STD', S_loss_KD_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/S_loss_cls_STD', S_loss_cls_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_FSL_STD', A_loss_fsl_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_KD_STD', A_loss_KD_STD.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/A_loss_cls_STD', A_loss_cls_STD.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_loop(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,_) in enumerate(test_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct(x)
      acc_all.append(correct_this/ count_this*100  )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    return acc_mean
