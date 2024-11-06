import shutil
import time

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
import torchvision.transforms as transforms
import tqdm
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import os.path as op
import os
from PIL import Image
from Data_generate import Data_generator
from DrawLib import DrawOHLCTechnical


class MiniDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        with Image.open(self.data[item], 'r') as img:
            img_t = transforms.Resize((224, 224))(img)
            img.close()
            img_tensor = transforms.ToTensor()(img_t)
            return img_tensor, item


class BackTest:
    def __init__(self, test_data_path: str, window: int, N_day: int, model, model_name: str, tech: list,
                 out_img_path='./result/img/'):
        if not op.exists(test_data_path):
            raise Exception('test data path doesn\'t exist!')

        hasCache = False
        if op.exists('./cache/BT_df.pt'):
            t1 = op.getmtime(test_data_path)
            t2 = op.getmtime('./cache/BT_df.pt')
            if t1 <= t2:
                hasCache = True
                df = torch.load('./cache/BT_df.pt')
        if not hasCache:
            file_list = os.listdir(test_data_path)
            bar = tqdm.tqdm(desc='reading data', total=len(file_list), initial=1)
            df_list = []
            for i, file in enumerate(file_list):
                if file[-4:] == '.csv':
                    df_list.append(pd.read_csv(op.join(test_data_path, file)))
                bar.update(1)
            df = pd.concat(df_list, axis=0)
            torch.save(df, './cache/BT_df.pt', pickle_protocol=4)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.eval()
        period_group = df.groupby('sample_period')
        p_bar = tqdm.tqdm(desc='processing', initial=1, total=len(period_group))
        existed_period_list = []
        if op.exists('./result/dataframe/'):
            existed_file = os.listdir('./result/dataframe/')
            for file in existed_file:
                token = file[:-3].split('_')
                name, _period = token[0], int(token[3])
                if name == model_name:
                    existed_period_list.append(_period)
        existed_period = max(existed_period_list) if len(existed_period_list) > 0 else -1
        for i, p in enumerate(period_group):
            period_i = p[0]
            if period_i <= existed_period:
                p_bar.update(1)
                continue
            if op.exists(op.join(out_img_path, f'{model_name}')):
                period_folder = os.listdir(op.join(out_img_path, f'{model_name}'))
                for folder in period_folder:
                    if int(folder) < period_i:
                        shutil.rmtree(op.join(out_img_path, f'{model_name}', folder))
            df_i = p[1]
            stock_group = df_i.groupby('TICKER')
            td_list = []
            hasTD = False
            if op.exists(f'./cache/td_list_{period_i}.pt'):
                t1 = op.getmtime(test_data_path)
                t2 = op.getmtime(f'./cache/td_list_{period_i}.pt')
                if t1 <= t2:
                    td_list = torch.load(f'./cache/td_list_{period_i}.pt')
                    hasTD = True
            if not hasTD:
                for j, s in enumerate(stock_group):
                    td_list.append(pd.to_datetime(s[1]['date']))
                td = pd.concat(td_list)
                td.drop_duplicates(keep='first', inplace=True)
                td_list = td.tolist()
                td_list.sort()
                td_list = [str(t.date()) for t in td_list]
                torch.save(td_list, f'./cache/td_list_{period_i}.pt')
            max_length = len(td_list) - window - N_day
            ret_dict = {'date': []}
            rank_dict = {'date': []}
            t_bar = tqdm.tqdm(desc=f'sample period: {period_i}|processing cross-sectional data', initial=1,
                              total=max_length, leave=False)
            cache_list = os.listdir('./cache/')
            exist_t_list = []
            for cache in cache_list:
                if 'retdict' in cache or 'rankdict' in cache:
                    f_name = cache[:-3].split('_')
                    m_n, p_i, _t = f_name[1], int(f_name[2]), int(f_name[3])
                    if m_n == model_name:
                        if p_i == period_i:
                            exist_t_list.append(int(_t))
            newest_t = max(exist_t_list) if len(exist_t_list) > 0 else -1

            for t in range(max_length):
                # time tag 0
                t1 = time.time()
                if t <= newest_t:
                    if t == newest_t:
                        ret_dict = torch.load(f'./cache/retdict_{model_name}_{period_i}_{newest_t}.pt')
                        rank_dict = torch.load(f'./cache/rankdict_{model_name}_{period_i}_{newest_t}.pt')
                    t_bar.update(1)
                    continue
                aligned_series = {'start_date': '', 'end_date': '', 'projection_date': '', 'df_series': [], 'stock': [],
                                  'img_list': [],
                                  'rank': [], 'ret': []}

                for j, s in tqdm.tqdm(enumerate(stock_group), desc=f'aligning data on date {td_list[t]}', leave=False,
                                      total=len(stock_group)):
                    stock_j = s[0]
                    stock_j: str
                    df_j = s[1]
                    df_j: pd.DataFrame
                    df_j.reset_index(inplace=True)
                    if t + window + N_day >= len(df_j):
                        continue
                    exist_T = False
                    bool_s = df_j['date'].isin([td_list[t]])
                    if True in bool_s.values:
                        exist_T = True
                    if not exist_T:
                        continue
                    T = df_j.index[bool_s][0]

                    if len(aligned_series['start_date']) == 0:
                        aligned_series['start_date'] = td_list[t]
                        aligned_series['end_date'] = td_list[t + window]
                        aligned_series['projection_date'] = td_list[t + window + N_day]
                        ret_dict['date'].append(aligned_series['projection_date'])
                        rank_dict['date'].append(aligned_series['end_date'])
                    if ((T + window + N_day < len(df_j['date']))
                            and (aligned_series['start_date'] == df_j['date'][T]
                                 and aligned_series['end_date'] == df_j['date'][T + window]
                                 and aligned_series['projection_date'] == df_j['date'][
                                     T + window + N_day])):
                        aligned_series['df_series'].append(df_j.loc[T:T + window + N_day, :].copy().reset_index())
                        aligned_series['stock'].append(stock_j)
                        if f'stock_{stock_j}' not in list(rank_dict.keys()):
                            ret_dict[f'stock_{stock_j}'] = []
                            rank_dict[f'stock_{stock_j}'] = []
                start_date = aligned_series['start_date']
                if len(aligned_series['stock']) > 0:
                    if not op.exists(op.join(out_img_path,f'{model_name}/{period_i}/{start_date}/')):
                        os.makedirs(op.join(out_img_path,f'{model_name}/{period_i}/{start_date}/'))
                    with mp.Pool(processes=12) as pool:
                        l = pool.starmap(fn, [(out_img_path, model_name, period_i, start_date,
                                               aligned_series['df_series'][s], window, N_day, tech) for s in
                                              range(len(aligned_series['df_series']))])
                    aligned_series['ret'] = [l[i][0] for i in range(len(l))]
                    aligned_series['img_list'] = [l[i][1] for i in range(len(l))]
                    # if (len(aligned_series['df_series']) != len(aligned_series['ret'])) and (
                    #         len(aligned_series['df_series']) != len(aligned_series['img_list'])) and (
                    #         len(aligned_series['df_series']) != len(aligned_series['stock'])):
                    #     raise Exception('data series is not aligned!')
                    mini_dataset = MiniDataset(aligned_series['img_list'])
                    loader = DataLoader(mini_dataset,
                                        batch_size=20 if len(aligned_series['stock']) > 8 else 1,
                                        num_workers=8 if len(aligned_series['stock']) > 8 else 1, shuffle=False,
                                        persistent_workers=True, pin_memory=True)
                    m_bar = tqdm.tqdm(initial=1, total=len(mini_dataset),
                                      desc=f'sample period: {period_i}|date: {start_date}|feeding model', leave=False)

                    for item in loader:
                        with torch.inference_mode():
                            img_tensor = item[0].to(device)
                            idx = item[1]
                            pred_label = model(img_tensor)
                            batch_size = pred_label.shape[0]
                            pred_label = pred_label.detach().cpu().numpy()
                            for batch in range(batch_size):
                                prob = pred_label[batch].max()
                                label = pred_label[batch].argmax()
                                score = prob * -1 if label == 0 else prob
                                aligned_series['rank'].append(score)
                            m_bar.update(batch_size)
                    # if len(aligned_series['df_series']) != len(aligned_series['rank']):
                    #     raise Exception('data series is not aligned!')

                    for j in range(len(aligned_series['stock'])):
                        stock_j = aligned_series['stock'][j]
                        ret = aligned_series['ret'][j]
                        rank = aligned_series['rank'][j]
                        ret_dict[f'stock_{stock_j}'].append((aligned_series['projection_date'], ret))
                        rank_dict[f'stock_{stock_j}'].append((aligned_series['end_date'], rank))

                def clear_cache():
                    clist = os.listdir('./cache')
                    for cache in clist:
                        if 'retdict' in cache or 'rankdict' in cache:
                            f_name = cache[:-3].split('_')
                            m_n, p_i, _t = f_name[1], int(f_name[2]), int(f_name[3])
                            if m_n == model_name:
                                if p_i < period_i or _t < t:
                                    os.remove(f'./cache/{cache}')

                if t % 10 == 0 or t == max_length - 1:
                    torch.save(ret_dict, f'./cache/retdict_{model_name}_{period_i}_{t}.pt', pickle_protocol=4)
                    torch.save(rank_dict, f'./cache/rankdict_{model_name}_{period_i}_{t}.pt', pickle_protocol=4)
                    clear_cache()
                t2 = time.time()
                print(f'\n total time consumption on t{t}:', t2 - t1)
                t_bar.update(1)
            ret_list = []
            rank_list = []
            for key in ret_dict.keys():
                if key != 'date':
                    ret_dict[key] = pd.DataFrame(ret_dict[key], columns=['date', f'{key}'])
                    ret_dict[key]['date'] = pd.to_datetime(ret_dict[key]['date'])
                    ret_dict[key].drop_duplicates(subset=['date'], keep='first', inplace=True)
                    ret_dict[key].set_index('date', inplace=True)
                    ret_list.append(ret_dict[key])
                    rank_dict[key] = pd.DataFrame(rank_dict[key], columns=['date', f'{key}'])
                    rank_dict[key]['date'] = pd.to_datetime(rank_dict[key]['date'])
                    rank_dict[key].drop_duplicates(subset=['date'], keep='first', inplace=True)
                    rank_dict[key].set_index('date', inplace=True)
                    rank_list.append(rank_dict[key])
            ret_df = pd.concat(ret_list, axis=1)
            rank_df = pd.concat(rank_list, axis=1).rank(axis=1, method='first')
            ret_df.to_csv(f'./result/dataframe/{model_name}_sample_period_{period_i}_ret.csv', index=True)
            rank_df.to_csv(f'./result/dataframe/{model_name}_sample_period_{period_i}_rank.csv', index=True)
            p_bar.update(1)
        if op.exists(f'{out_img_path}/{model_name}'):
            shutil.rmtree(f'{out_img_path}/{model_name}')


def fn(out_img_path, model_name, period_i, start_date, df: pd.DataFrame, window, N_day, tech):
    d_g = Data_generator(with_no_init=True, market='us')
    d_g.df = df.copy()
    data_zip = d_g.sep_OHLC_Techs_Label(start=0, window=window, N_days_after=N_day,
                                        require_actual_ret=True,
                                        require_true_date=True)
    stock_no = df['TICKER'][0]
    ret = data_zip['label_true']
    img_path = op.join(out_img_path, f'{model_name}/{period_i}/{start_date}/', f'{stock_no}_{ret}.bmp')
    if not op.exists(img_path):
        img, status = DrawOHLCTechnical(224, data_zip['OHLC'],
                                        f'{out_img_path}/{model_name}/{period_i}/{start_date}/',
                                        f'{stock_no}_{ret}.bmp',
                                        MA=data_zip['MA'] if 'MA' in tech else None,
                                        MACD=data_zip['MACD'] if 'MACD' in tech else None,
                                        Vol=data_zip['Vol'] if 'Vol' in tech else None,
                                        BOLL=data_zip['BOLL'] if 'BOLL' in tech else None,
                                        RSI=data_zip['RSI'] if 'RSI' in tech else None)
        img.close()
    return ret, img_path
