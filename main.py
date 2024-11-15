import os
import re
import time
from os import listdir
import pandas as pd
import timm
from PIL import Image
import torch
from Data_generate import Data_generator
import torchvision.transforms as transforms
from torch import optim, nn
from model import Model_workflow, EarlyStop_func, M_Model
import os.path as op
import tqdm
import multiprocessing as mtp
from BTframe import BackTest


def preprocess(CRSP_PATH='N:/Dataset/CRSP/CRSP.csv',OPEN_PRICE='N:/Dataset/CRSP/Open price.csv'):
    df1 = pd.read_csv(CRSP_PATH)
    df2 = pd.read_csv(OPEN_PRICE)
    df3 = pd.concat([df1[["PERMNO", "date", "TICKER", "PRIMEXCH", "CUSIP", "BIDLO", "ASKHI", "PRC", "VOL", "CFACPR"]],
                     df2['OPENPRC']], axis=1)
    df3.rename(columns={"BIDLO": 'low', 'ASKHI': 'high', 'VOL': 'volume', 'PRC': 'close', 'OPENPRC': 'open'},
               inplace=True)
    df3 = df3.reindex(
        columns=["PERMNO", "date", "TICKER", "PRIMEXCH", "CUSIP", 'open', 'high', 'low', 'close', 'volume', 'CFACPR'])
    grouped = df3.groupby('TICKER')
    counter = 0
    for i in tqdm.tqdm(grouped):
        flag = False
        no, df = i[0], i[1]
        path = 'D:/test/PILtest/data_us'
        filename = op.join(path, str(no) + '.csv')
        df.drop(index=df[df['close'].isna()].index, inplace=True)
        df.drop(index=df[df['volume'] == 0].index, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date',inplace=True)
        df.sort_index(ascending=True,inplace=True)
        df.reset_index(inplace=True)
        for idx in range(len(df['CFACPR'])):
            val = df['CFACPR'][idx]
            if val == 0:
                flag = True
                with open('./log/preprocessing.txt', 'a') as file:
                    print(f'TICKER {no} has illegal CFACPR in line {idx}', file=file)
                    df.to_csv(f'./log/deprecated/{str(no)}' + '.csv')
                break
        if flag:
            continue
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(lambda x: x / df['CFACPR'],
                                                                                          axis=0)
        df.to_csv(filename, index=False)


def add_tech(idx):
    d_g = Data_generator('./data_us',market='us')
    d_g.addTechnicalIndicators(idx,save=True,MA=(5, 20, 60), BOLL=(10, 2), MACD=(12, 26, 9), RSI=(5, 10, 20))


if __name__ == '__main__':
    preprocess()
    data_path = './data_us/'
    d_g = Data_generator(data_path, market='us')
    idx_list = [i for i in range(len(d_g.csv_name_list))]
    with mtp.Pool(processes=10) as pool: # multiprocessing
        pool.map(add_tech,idx_list)

    # for i in range(len(d_g.csv_name_list)): # non-multiprocessing
    #     d_g.addTechnicalIndicators(i, save=True, MA=(5, 20, 60), BOLL=(10, 2), MACD=(12, 26, 9), RSI=(5, 10, 20))
    
    data_path = './data_us/tech'
    d_g = Data_generator(data_path, 'us')
    d_g.div_sample_test_sets('1992-1-1', '2023-1-1', 10)
    
    _model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2, in_chans=3)

    tech_list = (['BOLL','RSI'],['BOLL','MACD'],['BOLL','Vol'],['MA','RSI'],['MA','MACD'],['MA','Vol'])
    window_list = [5,20,60]
    seed = 1000
    for tech in tech_list:
        for window in window_list:
            lr = 1e-8
            criterion = nn.CrossEntropyLoss(reduction='mean')
            optimizer = optim.AdamW
            epoch = 100

            torch.manual_seed(seed)

            my_model = Model_workflow('./data_us/tech')
            my_model.register_save_path('./checkpoint')
            my_model.register_earlyStop(EarlyStop_func)
            my_model.train(_model, f'swin{window}{tech[0]}+{tech[1]}', epoch, lr, 10, 50, criterion, optimizer, torch.nn.init.xavier_normal_,
                           0, window, 5, tech, 2,seed)
            seed += 1
    with open('./log/validating accuracy.txt', 'r') as f:
        content = f.read()
        lines = content.split('\n')
    adict = {}
    for i, line in enumerate(lines):
        if len(line) == 0:
            continue
        words = line.split(' ')
        if i == 0:
            adict['model name'] = [words[2]]
            adict[words[3]] = [int(words[4])]
            adict[words[5] + ' ' + words[6][:-1]] = [float(words[7])]
        else:
            adict['model name'].append(words[2])
            adict[words[3]].append(int(words[4]))
            adict[words[5] + ' ' + words[6][:-1]].append(float(words[7]))
    df = pd.DataFrame(adict)
    df.to_csv('./statistics/model_accuracy.csv')
    group = df.groupby('model name')
    if op.exists('./log/outperforming model.txt'):
        os.remove('./log/outperforming model.txt')
    for i, g in enumerate(group):
        m = g[1]['validating accuracy'].mean()
        with open('./log/outperforming model.txt', 'a') as f:
            if m > 0.51:
                print(g[0], file=f)
    with open('./log/outperforming model.txt') as f:
        content = f.read()
        model_list = content.split('\n')
    for name in model_list:
        if len(name)>0:
            for w in window_list:
                if str(w) in name:
                    window = w
                    break
            for t in tech_list:
                if t[0] in name and t[1] in name:
                    tech = t
                    break
            _model.load_state_dict(torch.load(f'./checkpoint/{name}_params_complete.pt'))

            BT = BackTest('./data_us/tech/test', window, 5, _model, f'swin{window}{tech[0]}+{tech[1]}', tech
                          ,out_img_path='N:/Dataset/img')
