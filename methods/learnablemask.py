import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim):
        super(LearnableMaskLayer, self).__init__()
        self.mask = torch.nn.Parameter(torch.randn((2, feature_dim, 1, 1)))

    def forward(self, x, domain_flag):
        #soft mask to hard mask[0,1]
        if self.training:
            hard_mask = F.gumbel_softmax(self.mask, hard=True, dim=0)
        else:
            hard_mask = F.softmax(self.mask, dim=0)
            hard_mask = (hard_mask>0.5).float()    
        #print('S:', torch.sum(hard_mask[0]))
        #print('A:', torch.sum(hard_mask[1]))
        if(domain_flag=='S'):
            hard_mask = hard_mask[0]
        elif(domain_flag=='A'):
            hard_mask = hard_mask[1]
      
        hard_mask =  hard_mask.unsqueeze(0)
        x = x * hard_mask

        return x



if __name__ == '__main__':
    myLearnableMaskLayer = LearnableMaskLayer(feature_dim = 8)
    x = torch.randn(2,8,64, 64)
    out_x, out_loss = myLearnableMaskLayer(x, domain_flag='S')
    print(out_x, out_loss)
