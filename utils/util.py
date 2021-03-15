from torch.autograd import Variable
import torch
class Helper():
    """
    工具类： 用来提供各种工具函数
    """
    def __init__(self):
        self.a = None

    def to_longtensor(self,x,use_gpu):
        x = torch.LongTensor(x)
        if use_gpu:
            x = x.cuda()
        return x

    def to_floattensor(self,x,use_gpu):
        x = torch.FloatTensor(x)
        if use_gpu:
            x = x.cuda()
        return x

    def comput_distance_loss(self,smi_common,fas_common,tag):

        #dg-pt interaction
        dg_pt_temp1 = torch.pow(torch.sub(smi_common,fas_common),2)
        dg_pt_temp2 = torch.sum(dg_pt_temp1,dim=1)
        loss = torch.sum(torch.mul(dg_pt_temp2,tag))

        return loss






