import time
import datetime

import numpy as np
import torch.optim as optim
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from config import Config
from utils.util import Helper
from model.CommonModel import Common_model
from model.PredictModel import Predict_model
from dataset import HetrDataset

def train_common_model(config,helper,model,hetrdataset,repeat_nums,flod_nums):

    optimizer = optim.Adam(model.parameters(),config.common_learn_rate)
    model.train()

    print("common model begin training----------",datetime.datetime.now())

    #common_loss
    for e in range(config.common_epochs):

        common_loss = 0
        begin_time = time.time()

        for i, (dg,pt,tag) in enumerate(hetrdataset.get_train_batch(repeat_nums,flod_nums,config.batch_size)):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            tag = helper.to_floattensor(tag,config.use_gpu)

            optimizer.zero_grad()

            smi_common, fas_common = model(dg,pt)

            distance_loss = helper.comput_distance_loss(smi_common,fas_common,tag)
            common_loss += distance_loss

            distance_loss.backward()
            optimizer.step()

        #end a epech
        print("the loss of common model epoch[%d / %d]:is %4.f, time:%d s" % (e+1,config.common_epochs,common_loss,time.time()-begin_time))

def train_predict_model(config,helper,predict_model,common_model,hetrdataset,repeat_nums,flod_nums):

    optimizer1 = optim.Adam(predict_model.parameters(),config.pre_learn_rate)
    optimizer2 = optim.Adam(common_model.parameters(),config.common_learn_rate)

    predict_model.train()
    common_model.train()

    print("predict model begin training----------",datetime.datetime.now())

    #tag_loss
    for e in range(config.predict_epochs):

        epoch_loss = 0
        begin_time = time.time()

        for i, (dg,pt,tag) in enumerate(hetrdataset.get_train_batch(repeat_nums,flod_nums,config.batch_size)):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            tag = helper.to_floattensor(tag,config.use_gpu)

            smi_common,fas_common = common_model(dg,pt)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            predict, tag  = predict_model(smi_common,fas_common,tag)

            tag_loss = F.binary_cross_entropy(predict,tag)
            epoch_loss += tag_loss

            tag_loss.backward()

            optimizer1.step()
            optimizer2.step()

        # end a epech
        print("the loss of predict model epoch[%d / %d]:is %4.f, time:%d s" %  (e+1, config.predict_epochs, epoch_loss, time.time() - begin_time))

        evaluation_model(config,helper,predict_model,common_model,hetrdataset,repeat_nums,flod_nums)

def evaluation_model(config,helper,predict_model,common_model,hetrdataset,repeat_nums,flod_nums):
    predict_model.eval()
    common_model.eval()
    print("evaluate the model")

    begin_time = time.time()
    loss = 0
    avg_acc = []
    avg_aupr = []
    with torch.no_grad():
        for i,(dg,pt,tag) in enumerate(hetrdataset.get_test_batch(repeat_nums,flod_nums,config.batch_size)):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            tag = helper.to_floattensor(tag,config.use_gpu)

            smi_common,fas_common = common_model(dg,pt)
            predict, tag = predict_model(smi_common, fas_common, tag)

            tag_loss = F.binary_cross_entropy(predict,tag)
            loss +=tag_loss

            auc = roc_auc_score(tag.cpu(),predict.cpu())
            aupr = average_precision_score(tag.cpu(),predict.cpu())

            avg_acc.append(auc)
            avg_aupr.append(aupr)

    print("the total_loss of test model:is %4.f, time:%d s" % (loss, time.time() - begin_time))
    print("avg_acc:",np.mean(avg_acc),"avg_aupr:",np.mean(avg_aupr))


if __name__=='__main__':

    # initial parameters class
    config = Config()

    # initial utils class
    helper = Helper()

    #initial data
    hetrdataset = HetrDataset()

    #torch.backends.cudnn.enabled = False 把

    model_begin_time = time.time()
    for i in range(config.repeat_nums):
        print("repeat:",str(i),"+++++++++++++++++++++++++++++++++++")
        for j in range(config.fold_nums):
            print(" crossfold:", str(j), "----------------------------")

            #initial presentation model
            c_model = Common_model(config)
            p_model = Predict_model()
            if config.use_gpu:
                c_model = c_model.cuda()
                p_model = p_model.cuda()

            for epoch in range(config.num_epochs):
                print("         epoch:",str(epoch),"zzzzzzzzzzzzzzzz")
                train_common_model(config,helper,c_model,hetrdataset,i,j)
                train_predict_model(config,helper,p_model,c_model,hetrdataset,i,j)

    print("Done!")
    print("All_training time:",time.time()-model_begin_time)