import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from wav2vec_mean import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

from compute import ComputeAccuracy, ComputeCavg, ComputeEER
from outputlib import WriteConfusionSeaborn
from hparams import hparams, hparams_debug_string

import datetime
from time import strftime
from utils import  create_logger
# from tqdm import tqdm

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print('Device: {}'.format(device))

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu' )


def evaluate_accuracy(dev_loader, model, device,logger):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)

    # criterion = nn.CrossEntropyLoss()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:

            # batch_size = batch_x.size(0)

            # num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # print("batch_y=", batch_y)
            batch_out = model(batch_x)

            # batch_loss = criterion(batch_out, batch_y)
            # val_loss += (batch_loss.item() * batch_size)

            all_outputs.append(batch_out)
            # print("all_output=",all_outputs)
            all_targets.append(torch.tensor([batch_y]))


        # val_loss /= num_total
        all_outputs = torch.cat(all_outputs).cpu().data.numpy()
        all_targets = torch.cat(all_targets).cpu().data.numpy()


        np.save("egs/wav2vec_output.npy", all_outputs)
        np.save("egs/wav2vec_targets.npy", all_targets)
        np.savetxt("egs/wav2vec_output.txt", all_outputs, delimiter=' ')
        np.savetxt("egs/wav2vec_targets.txt", all_targets, delimiter=' ')


        all_predict = np.argmax(all_outputs, axis = 1)

        acc, class_accs, confu_mat = ComputeAccuracy(all_outputs, all_predict, all_targets)
        cavg = ComputeCavg(all_outputs, all_targets)
        eer, thd = ComputeEER(all_outputs, all_targets)

        #添加个混淆矩阵
        WriteConfusionSeaborn(
            confu_mat,
            hparams.lang,
            os.path.join("/home/lqb/project/SSL_Anti-spoofing-main", 'confmat_mean.png')
        )

        class_total = np.sum(confu_mat, axis=1)

        # 自己

        for i in range(len(class_accs)):
            # print('* Accuracy of {:6s} ........... {:6.2f}% {:4d}/{:<4d}'.format(
            #     hparams.lang[i], 100 * class_accs[i], confu_mat[i][i], class_total[i]))

            logger.info('* Accuracy of {:6s} ........... {:6.2f}% {:4d}/{:<4d}'.format(
                hparams.lang[i], 100*class_accs[i], confu_mat[i][i], class_total[i]))

        logger.info(": Acc:{:.04f}% Cavg: {}  EER: {}%  threshold: {}  Eval Loss:{:0.4f}".format(
                acc*100, cavg, eer, thd, val_loss))
        print(": Acc:{:.04f}% Cavg: {}  EER: {}%  threshold: {}  Eval Loss:{:0.4f}".format(
                acc*100, cavg, eer, thd, val_loss))

        return acc, confu_mat, cavg, eer, thd



def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)
        # print("batch_out=",batch_out)

        #本来的
        # batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        #
        #自己
        batch_score = (batch_out).data.cpu().numpy()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device,epoch,logger):
    running_loss = 0
    num_total = 0.0
    acc = 0.
    total = 0.
    model.train()

    #ZJ
    criterion = nn.CrossEntropyLoss()

    for idx, (batch_x, batch_y) in enumerate(train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        #zj
        _, pred = (batch_out).max(1)
        acc += pred.eq(batch_y).sum().item()
        total += batch_y.size(0)

        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if idx % hparams.print_intervals == 0 and idx != 0:
        # if idx % hparams.print_intervals == 0 :

            # logger = create_logger(model_save_path)
            logger.info('[Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(epoch,
                                                                                                       running_loss /num_total,
                                                                                                       acc / total * 100.,
                                                                                                       acc, total))
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset  数据
    parser.add_argument('--database_path', type=str, default='/home/lqb/project/SSL_Anti-spoofing-main/Dataset-AP20-dialect/IEMOCAP/Audio_16k/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

    '''

    parser.add_argument('--protocols_path', type=str, default='Dataset-AP20-dialect/IEMOCAP/labels/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters  超参数
    parser.add_argument('--batch_size', type=int, default=16)  #14
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model  模型
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 1234)')  #1234
    
    parser.add_argument('--model_path', type=str,
                        default="models/model_wav2vec_mean_2_weighted_CCE_100_16_1e-06/epoch_15.pth", help='Model checkpoint')  #default=None  这是用来评估模型?
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments  辅助参数
    parser.add_argument('--track', type=str, default='wav2vec_mean_2',choices=['wav2Raw2','wav2vec_mean_2','wav2Raw3','wav2vec_mean','RawNet3'], help='task')
    parser.add_argument('--eval_output', type=str, default="scores_1.txt",
                        help='Path to save the evaluation result') #default=False
    parser.add_argument('--eval', action='store_true', default=True, help='eval mode')  #default=False
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database') #default=False
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options  后端选项
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation  Rawboost数据增强 ======================================================================#
    #之前训练时定义是default=5
    parser.add_argument('--algo', type=int, default=5,
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters    LnL_卷积_噪声参数
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters  ISD 加性噪声参数
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters   SSI 加性噪声参数
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible  使实验可重复
    set_random_seed(args.seed, args)
    
    track = args.track

    #define model saving path  定义模型保存路径
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)

    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory  设置模型保存目录
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    

    model = Model(args,device)
    print(model)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    #evaluation 
    if args.eval:
        d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(args.protocols_path + 'test_utt2spk.txt'),is_train=False, is_eval=False)
        print('no. of validation trials', len(file_dev))
        dev_set = Dataset_ASVspoof2021_eval(list_IDs=file_dev,labels=d_label_dev,base_dir=os.path.join(args.database_path + 'test/'))
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, drop_last=False)
        evaluate_accuracy(dev_loader, model, device, model_save_path)
        sys.exit(0)


    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'train_utt2spk.txt'),is_train=True,is_eval=False)
    # d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'dev_utt2spk.txt'),is_train=True,is_eval=False)
    # d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'test_utt2spk.txt'),is_train=True,is_eval=False)

    print('no. of training trials',len(file_train))
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'train/'),algo=args.algo)
    # train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'dev/'),algo=args.algo)
    # train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'test/'),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    del train_set,d_label_trn
    

    # define validation dataloader   定义验证数据加载器
    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'test_utt2spk.txt'),is_train=False,is_eval=False)
    print('no. of validation trials',len(file_dev))

    #不应该用Dataset_ASVspoof2019_train,应该用Dataset_ASVspoof2021_eval
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'test/'),algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=1,num_workers=8, shuffle=False)
    del dev_set,d_label_dev


    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    # 自己
    logger = create_logger(model_save_path)
    for epoch in range(num_epochs):

        # running_loss = train_epoch(train_loader,model, args.lr,optimizer, device,epoch,model_save_path)
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device,epoch,logger)
        acc, confu_mat, cavg, eer, thd = evaluate_accuracy(dev_loader, model, device,logger)
        # writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        # print('\n{} - {} - {} '.format(epoch,running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
