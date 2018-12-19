# try to build a lstm network
import scipy.io as sio
import random
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import my2module2rnn
import my2linear
from torch.nn import functional as F

torch.manual_seed(1)
# writer = SummaryWriter('run/LSTM_129-129(L)-256-256-129(L)')

class LSTMNet(nn.Module):

    def __init__(self, featDim, hidden_dim,batch_size):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # self.linear1 = nn.Linear(featDim, hidden_dim[0])
        self.lstm1 = nn.LSTM(hidden_dim[0], hidden_dim[1], 2, batch_first=True, bidirectional=True)
        # self.lstm1 = nn.LSTM(hidden_dim[0], hidden_dim[1], 1, batch_first=True, bidirectional=False)
        self.lstm = my2module2rnn.sparseLSTM(hidden_dim[0], hidden_dim[3], 1, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_dim[1]*2, hidden_dim[0])


    def init_hidden_state1(self):
        # self.h_c_lstm1 = autograd.Variable(torch.zeros(2, 1, self.batch_size, self.hidden_dim[1]).cpu())
        self.h_c_lstm1 = (Variable(torch.zeros(4, self.batch_size, self.hidden_dim[1]).cpu()),
                    Variable(torch.zeros(4, self.batch_size, self.hidden_dim[1]).cpu()))
        return self.h_c_lstm1

    def init_hidden_state(self):
        self.h_c_lstm = autograd.Variable(torch.zeros(2, 1, self.batch_size, self.hidden_dim[3]).cpu())
        # self.h_c_lstm = (Variable(torch.zeros(4, self.batch_size, self.hidden_dim[3]).cpu()),
        #                   Variable(torch.zeros(4, self.batch_size, self.hidden_dim[3]).cpu()))

        return self.h_c_lstm

    def forward(self, input, sequence, dic):
        # layer1 = self.linear1(input)
        # layer1 = F.elu(layer1, alpha=1.0, inplace=False)

        packed_layer1 = nn.utils.rnn.pack_padded_sequence(input, sequence, batch_first=True)
        layer2, _ = self.lstm(packed_layer1, self.h_c_lstm)
        unpack_layer2 = nn.utils.rnn.pad_packed_sequence(layer2, batch_first=True)
        unpack_layer2 = unpack_layer2[0]
        dic = torch.from_numpy(dic).float()
        dic = Variable(dic)
        unpack_layer3 = torch.matmul(unpack_layer2, dic)
        packed_layer3 = nn.utils.rnn.pack_padded_sequence(unpack_layer3, sequence, batch_first=True)
        layer3, _ = self.lstm1(packed_layer3, self.h_c_lstm1)

        unpack_layer4 = nn.utils.rnn.pad_packed_sequence(layer3, batch_first=True)
        unpack_layer4 = unpack_layer4[0]
        layer4 = self.linear(unpack_layer4)


        #
        # packed_layer1 = nn.utils.rnn.pack_padded_sequence(input, sequence, batch_first=True)
        # layer2, _ = self.lstm1(packed_layer1, self.h_c_lstm1)
        # layer3, _ = self.lstm(layer2, self.h_c_lstm)
        #
        # unpack_layer3 = nn.utils.rnn.pad_packed_sequence(layer3,batch_first=True)
        # unpack_layer3 = unpack_layer3[0]
        #
        # dic = torch.from_numpy(dic).float()
        # dic = Variable(dic)
        # unpack_layer4 = torch.matmul(unpack_layer3, dic)
        #
        # layer4 = self.linear(unpack_layer4)

        return layer4


def shuffle_data(AC,BC,AC_or,trainNum):
    index = list(range(trainNum))
    random.shuffle(index)
    AC = AC[index]
    BC = BC[index]
    AC_or = AC_or[index]
    return AC,BC,AC_or

def prepare_data(bc,ac,ac_orig,batchsize,featDim):
    DATA = np.zeros((batchsize,2000,featDim))
    LABEL = np.zeros((batchsize,2000,featDim))
    LABEL_de_log_norm = np.zeros((batchsize,2000,featDim))
    Masking = np.zeros((batchsize,2000,featDim))
    TrueSequence = []
    maxSequence = 1
    for i in range(ac.shape[0]):
            LABEL[i,:ac[i].shape[0],:] = ac[i]
            LABEL_de_log_norm[i,:ac[i].shape[0],:] = ac_orig[i]
            DATA[i,:bc[i].shape[0],:] = bc[i]
            Masking[i,:bc[i].shape[0],:] = np.ones(bc[i].shape)
            TrueSequence.append(ac[i].shape[0])
            if ac[i].shape[0]>maxSequence:
                maxSequence = ac[i].shape[0]
    paixu = np.argsort(TrueSequence)
    paixu = paixu[::-1]
    DATA = DATA[paixu,:maxSequence,:]
    LABEL = LABEL[paixu,:maxSequence,:]
    LABEL_de_log_norm = LABEL_de_log_norm[paixu,:maxSequence,:]
    Masking = Masking[paixu,:maxSequence,:]
    Sequence = np.array(TrueSequence)[paixu]
    return DATA,LABEL,LABEL_de_log_norm,Sequence,Masking

def log_and_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.shape[1]):
        temp = np.log(data[0][i])
        temp = (temp-mean)/std
        #plt.imshow(temp.T,origin='lower')
        log_norm_data.append(temp)
    return np.array(log_norm_data)
def log_and_normCOMB(data,mean,std):
    log_norm_data = []
    for i in range(data.data.shape[0]):
            temp = torch.log(data[i])
            temp = temp.data.numpy()
            temp = (temp-mean)/std
            #plt.imshow(temp.T,origin='lower')
            log_norm_data.append(temp)
    result = np.array(log_norm_data)
    return result


def normalize_AC(data, mean, std):
    log_norm_data = []
    for i in range(data.shape[1]):
        temp = data[0][i]
        temp = (temp - mean) / std
        log_norm_data.append(temp)
    return  np.array(log_norm_data)

def normalize_Dic(data):
    #log_norm_data = np.log(data)
    log_norm_data = data
    dic_num = log_norm_data.shape[0]
    sum1 = log_norm_data.sum( axis=0)
    mean1 = sum1/dic_num
    narray2 = log_norm_data * log_norm_data
    sum2 = narray2.sum(axis=0)
    var = sum2 / dic_num - mean1 ** 2
    log_norm_data = (log_norm_data - mean1) / (var** (1./2))
    log_norm_data = np.array(log_norm_data)

    #normlize it to 1 length
    # norm_dic = np.linalg.norm(log_norm_data, axis=1)
    # for i in range(log_norm_data.shape[0]):
    #     log_norm_data[i, :] = log_norm_data[i, :] / norm_dic[i]

    #log_norm_data = (log_norm_data-mean)/std
    return log_norm_data


def de_log_and_normalize(data,mean,std):

    log_norm_data = []
    for i in range(data.data.shape[0]):
            temp = data[i]
            temp = temp.data.numpy()
            temp = temp * std + mean
            temp = np.exp(temp)
            #plt.imshow(temp.T,origin='lower')
            log_norm_data.append(temp)
    result = np.array(log_norm_data)
    return result
def de_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.data.shape[0]):
            temp = data[i]
            temp = temp.data.numpy()
            temp = temp * std + mean
            log_norm_data.append(temp)
    result = np.array(log_norm_data)
    return result

def my_lstm_mse_loss(output,target,sequence,masking):

    error = ((output - target))*masking
    mm = int(sum(sequence))
    vv = torch.sum(error ** 2)
    error = vv / mm
    return error

def my_lstm_l1_loss(output,target,sequence,masking):

    target_abs = torch.abs(target)+1
    error = torch.abs((output - target)/target_abs*masking)
    mm = int(sum(sequence)) * 129
    vv = torch.sum(error)/mm
    if vv >= 1.0:
        less_than_one = 0
        # print('vv==', vv)
    else:
        less_than_one = 1.0
        # print('vv=', vv)
    l1_loss = ((less_than_one*0.5*torch.sum(error**2))+(1-less_than_one)*(torch.sum(error-0.5)))/mm

    return l1_loss

# load data and split to train and val dataset
TRAIN = sio.loadmat('8km001/m001_STFT_TRAINSET')
TRAIN_DIC = sio.loadmat('8km001/m001_sparse_Dic')# f001_STFT_Dic f001_Dic128
# f001_STFT_TRAINDic 240
# f001_STFT_Dic 200 not sparse

AC = TRAIN['STFT_ac']  # change stft to log
BC = TRAIN['STFT_bc']
AC_Dic = TRAIN_DIC['dic']
AC_Dic = normalize_Dic(AC_Dic)
AC_orig_Train = AC

dataInfo = sio.loadmat('8km001/m001_datainfo.mat')
AC_mean, AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
AC_mean_nolog, AC_std_nolog = dataInfo['STFT_ac_mean'],dataInfo['STFT_ac_var']
BC_mean, BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']
BC_mean_nolog, BC_std_nolog = dataInfo['STFT_bc_mean'],dataInfo['STFT_bc_var']

# normalize data
featDim = 129

AC = log_and_normalize(AC, AC_mean, AC_std)
AC_orig_Train = normalize_AC(AC_orig_Train, AC_mean_nolog, AC_std_nolog)

## compasating dicinary
# Comps_Dic = 1e-1*np.eye(featDim)
# Comps_Dic_Neg = (-1e-1)*np.eye(featDim)
# AC_Dic = np.vstack((AC_Dic, Comps_Dic, Comps_Dic_Neg))


# BC = log_and_normalize(BC, BC_mean, BC_std)
BC = normalize_AC(BC, BC_mean_nolog, BC_std_nolog)


testdata = sio.loadmat('8km001/m001_STFT_TESTSET')
t_ac,t_bc = np.array(testdata['STFT_ac']),np.array(testdata['STFT_bc'])
AC_orig_Val = t_ac
t_ac = log_and_normalize(t_ac, AC_mean, AC_std)
# t_bc = log_and_normalize(t_bc, BC_mean, BC_std)
t_bc = normalize_AC(t_bc, BC_mean_nolog, BC_std_nolog)

AC_orig_Val = normalize_AC(AC_orig_Val, AC_mean_nolog,AC_std_nolog)

Num = AC.shape[0]
train_ac,train_bc = AC[:],BC[:]
val_ac,val_bc = t_ac,t_bc
train_num = train_ac.shape[0]
val_num = val_ac.shape[0]

train_batchsize = 8
val_batchsize = 8
num_epochs = 100  # <---

hidden_dim = [129, 256, 256, 200]
num_train_batch = int(train_num/train_batchsize)
num_val_batch = int(val_num/val_batchsize)

LSTMModel = LSTMNet(featDim, hidden_dim,train_batchsize)
# initial weight
for name, param in LSTMModel.named_parameters():
    if 'bias' in name:
        nn.init.constant_(param, 0.0)
    elif 'weight' in name:
        nn.init.xavier_normal_(param)

# LSTMModel.load_state_dict(torch.load('data/params.pkl'))

LSTMModel.cpu()
criterion = nn.SmoothL1Loss()    #
# optimizer = optim.SGD(LSTMModel.parameters(), lr=0.1)

optimizer = optim.Adam(LSTMModel.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.RMSprop(LSTMModel.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=6, factor=0.5,min_lr=0.000001)
num_iteration_train = 0
num_iteration_test = 0
best_model_wts = copy.deepcopy(LSTMModel.state_dict())
best_loss = 1000

notimproveNum = 0
for epoch in range(num_epochs):
    # shuffle the dataorder
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    if notimproveNum > 16:
        print('Valloss do not improve at {} epochs,so break'.format(notimproveNum))
        break
    for phase in ['train', 'val']:
        if phase == 'train':
            #scheduler.step()
            LSTMModel.train()  # Set model to training mode
            LSTMModel.batch_size = train_batchsize
            num_batch = num_train_batch
            batchsize = train_batchsize
            AC,BC,AC_orig = shuffle_data(train_ac,train_bc,AC_orig_Train,train_num)

        else:
            LSTMModel.eval() # Set model to evaluate mode
            LSTMModel.batch_size = val_batchsize
            num_batch = num_val_batch
            batchsize = val_batchsize
            AC,BC,AC_orig = shuffle_data(val_ac,val_bc,AC_orig_Val,val_num)

        running_loss = 0.0
        for j in range(num_batch):
            DATA,LABEL,LABEL_de_log_norm,Sequence,Masking = prepare_data(BC[j*batchsize:(j+1)*batchsize],AC[j*batchsize:(j+1)*batchsize],AC_orig[j*batchsize:(j+1)*batchsize],batchsize,featDim) #prepare_data(bc,ac,batchsize,featDim):
            DATA,LABEL,LABEL_de_log_norm,Masking = torch.from_numpy(DATA).float(),torch.from_numpy(LABEL).float(),torch.from_numpy(LABEL_de_log_norm).float(),torch.from_numpy(Masking).float() # Pa
            DATA,LABEL,LABEL_de_log_norm,Masking= Variable(DATA.cpu()),Variable(LABEL.cpu()),Variable(LABEL_de_log_norm.cpu()),Variable(Masking.cpu())

            LSTMModel.zero_grad()
            LSTMModel.hidden1 = LSTMModel.init_hidden_state1()
            LSTMModel.hidden = LSTMModel.init_hidden_state()
            output = LSTMModel(DATA,Sequence, AC_Dic)

            #######################
            output = output*Masking
            LABEL_de_log_norm = LABEL_de_log_norm*Masking
            loss = criterion(output, LABEL_de_log_norm)
            #############
            # loss = my_lstm_l1_loss(output, LABEL_de_log_norm, Sequence,Masking)
            #######################

            if phase == 'train':
                loss.backward()
                optimizer.step()
                num_iteration_train = num_iteration_train+1
                # writer.add_scalar('TrainLoss', loss.data[0], num_iteration_train)
            else:
                num_iteration_test = num_iteration_test+1
                # writer.add_scalar('VALLoss', loss.data[0], num_iteration_test)
            running_loss += loss.data[0]
            batch_average_loss = running_loss/(j+1)

        epoch_loss = running_loss/(num_batch)
        if phase == 'val':
            former_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Epoch_VALLoss', epoch_loss, epoch)
            print('learning rate is {}'.format(optimizer.param_groups[0]['lr']))
            if  epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(LSTMModel.state_dict())
                LSTMModel.load_state_dict(best_model_wts)
                print('BestLoss: {:.4f} is Epoch{} '.format(best_loss,epoch+1))
                notimproveNum = 0
            else:
                notimproveNum = notimproveNum +1
            # if former_lr !=current_lr:
            #     notimproveNum = 0
        # else:
            # writer.add_scalar('Epoch_TrainLoss', epoch_loss, epoch)
        print('{} EpochLoss: {:.4f} '.format(phase,epoch_loss))

LSTMModel.load_state_dict(best_model_wts)
torch.save(LSTMModel.state_dict(),'params_nodic.pkl')


# ##------------evaluate----------------------------------------------##
# load data
TRAIN1 = sio.loadmat('8km001/m001_STFT_TESTSET')
AC,BC = TRAIN1['STFT_ac'],TRAIN1['STFT_bc']
dataInfo = sio.loadmat('8km001/m001_datainfo.mat')
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
BC_mean_nolog,BC_std_nolog = dataInfo['STFT_bc_mean'],dataInfo['STFT_bc_var']
AC_mean_nolog,AC_std_nolog = dataInfo['STFT_ac_mean'],dataInfo['STFT_ac_var']
# normalize data
BC = normalize_AC(BC, BC_mean_nolog, BC_std_nolog)

# load model
LSTMModel.load_state_dict(torch.load('params_nodic.pkl'))

# start to evaluate
testnum = BC.shape[0]
result = []

dic_val = torch.from_numpy(AC_Dic).float()
dic_val = Variable(dic_val)

for i in range(testnum):
    LSTMModel.batch_size = 1  # this should be write before hidden_init
    LSTMModel.hidden1 = LSTMModel.init_hidden_state1()
    LSTMModel.hidden = LSTMModel.init_hidden_state()
    DATA = BC[i]
    sequence = DATA.shape[0]
    DATA = DATA[np.newaxis,:,:]
    DATA = torch.from_numpy(DATA).float()
    DATA = Variable(DATA.cpu())
    predict = LSTMModel(DATA,[sequence], AC_Dic)


    # predict_val = torch.matmul(predict, dic_val)

    # denormalize
    # predict_val = de_normalize(predict_val.data[0,:,:].numpy(),AC_mean_nolog,AC_std_nolog)
    # result.append(predict_val.data[0,:,:].numpy())
    predict = de_normalize(predict.data[0,:,:],AC_mean_nolog,AC_std_nolog)
    predict = torch.from_numpy(predict).permute(1,0,2)
    predict = np.array(predict)
    predict_val = predict[0,:,:]
    result.append(predict_val)

sio.savemat('8km001/result/demo4/result4demo4.mat',{'result':result})