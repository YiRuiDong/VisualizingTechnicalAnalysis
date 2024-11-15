import copy

import numpy as np
import pandas as pd
import shutil
import time
import multiprocessing as mp
import torch
import os.path as op
import re
import os
import tqdm


class Benchmark:
    def __init__(self, data_path):
        ts = pd.date_range(start='1992-1-1', end='2023-1-1', periods=11)
        self.period_dict = {i: (ts[i], ts[i + 1]) for i in range(10)}
        self.df = None
        hasCache = False
        if op.exists('./cache/benchmark_df.pt'):
            hasCache = True
            self.df = torch.load('./cache/benchmark_df.pt')
        if not hasCache:
            self.df = []
            csv_list = []
            file_list = os.listdir(data_path)

            for f in copy.copy(file_list):
                if '.csv' in f:
                    csv_list.append(f)
            with mp.Pool(processes=12) as pool:
                self.df = pool.starmap(fn, [(data_path, self.period_dict, i) for i in csv_list])
            self.df = pd.concat(self.df, axis=0)
            self.df.reset_index(inplace=True)
            torch.save(self.df, './cache/benchmark_df.pt', pickle_protocol=4)

    def add_mom(self, *n):
        if len(n) == 0:
            return
        save_flag = False
        for i in n:
            if f'mom_{i}' in self.df.columns:
                self.df.drop(columns=[f'mom_{i}'], inplace=True)
            mom_n = self.df.groupby('TICKER')['close'].apply(lambda x: x.pct_change(i)).reset_index(drop=True)
            self.df[f'mom_{i}'] = mom_n
            save_flag = True
        if not save_flag:
            torch.save(self.df, './cache/benchmark_df.pt', pickle_protocol=4)
        return

    def add_revs(self, *n):
        if len(n) == 0:
            return
        save_flag = False
        for i in n:
            if f'revs_{i}' in self.df.columns:
                continue
            revs_n = self.df.groupby('TICKER')['close'].apply(
                lambda x: (x.rolling(i).mean() - x) / x.rolling(i).mean()).reset_index(drop=True)
            self.df[f'revs_{i}'] = revs_n
            save_flag = True
        if not save_flag:
            torch.save(self.df, './cache/benchmark_df.pt', pickle_protocol=4)
        return

    def gen_rank_ret(self, *sample_period_txt_path):
        if len(sample_period_txt_path) == 0:
            return
        else:
            self.df.set_index('date', inplace=True)
            indicators_list = []
            for k in self.df.columns:
                if ('mom' in k) or ('revs' in k):
                    indicators_list.append(k)
            for path in sample_period_txt_path:
                test_period = []
                with open(op.join(path, 'sample_period.txt'), 'r') as f:
                    lines = f.read().split('\n')
                    for line in lines:
                        if len(line) == 0:
                            continue
                        if line == 'test_period':
                            test = True
                            continue
                        if line == 'train_period':
                            test = False
                            continue
                        if test:
                            t1 = line.split('_')[0]
                            t2 = line.split('_')[1]
                            test_period.append((pd.to_datetime(t1), pd.to_datetime(t2)))
                period_i = 0
                for t in test_period:
                    for indicator in indicators_list:
                        # temp = re.sub('_','',indicator)
                        # if op.exists(f'{path}/result/dataframe/{temp}_sample_period_{period_i}_ret.csv'):
                        #     continue
                        ret_dict = {}
                        rank_dict = {}
                        ret_list = []
                        rank_list = []
                        g = self.df.groupby('TICKER')
                        for s in g:
                            stock = s[0]
                            if type(stock) is not str:
                                continue
                            stock_df = s[1]
                            stock_df = stock_df.loc[t[0]:t[1], :]
                            if len(stock_df) == 0:
                                continue
                            stock_df.reset_index(inplace=True)
                            try:
                                ret_dict[f'stock_{stock}'] = pd.DataFrame(
                                    {'date': stock_df['date'].shift(-5), f'stock_{stock}':
                                        stock_df['close'].pct_change(5, fill_method=None).shift(-5) * 100})
                                rank_dict[f'stock_{stock}'] = pd.DataFrame(
                                    {'date': stock_df['date'], f"stock_{stock}": stock_df[indicator]})
                            except Exception as e:
                                print(e)
                                raise e
                        for key in ret_dict.keys():
                            if key != 'date':
                                ret_dict[key].drop_duplicates(subset=['date'], keep='first', inplace=True)
                                ret_dict[key].set_index('date', inplace=True)
                                ret_list.append(ret_dict[key])
                                rank_dict[key].drop_duplicates(subset=['date'], keep='first', inplace=True)
                                rank_dict[key].set_index('date', inplace=True)
                                rank_list.append(rank_dict[key])
                        ret_df = pd.concat(ret_list, axis=1)
                        rank_df = pd.concat(rank_list, axis=1)
                        indicator = re.sub('_', '', indicator)
                        ret_df.to_csv(f'{path}/result/dataframe/{indicator}_sample_period_{period_i}_ret.csv',
                                      index=True)
                        rank_df.to_csv(f'{path}/result/dataframe/{indicator}_sample_period_{period_i}_rank.csv',
                                       index=True)
                    plain_ret_dict = {}
                    plain_rank_dict = {}
                    plain_ret_list = []
                    plain_rank_list = []
                    g = self.df.groupby('TICKER')
                    for s in g:
                        stock = s[0]
                        if type(stock) is not str:
                            continue
                        stock_df = s[1]
                        stock_df = stock_df.loc[t[0]:t[1], :]
                        stock_df.reset_index(inplace=True)
                        if len(stock_df) == 0:
                            continue
                        try:
                            plain_ret_dict[f'stock_{stock}'] = pd.DataFrame({'date': stock_df['date'],
                                                                             f'stock_{stock}': stock_df[
                                                                                                   'close'].pct_change(
                                                                                 1, fill_method=None) * 100})
                            plain_rank_dict[f'stock_{stock}'] = pd.DataFrame(
                                {'date': stock_df['date'], f"stock_{stock}": 1})
                        except Exception as e:
                            print(e)
                            raise e
                    for key in plain_ret_dict.keys():
                        if key != 'date':
                            plain_ret_dict[key].drop_duplicates(subset=['date'], keep='first', inplace=True)
                            plain_ret_dict[key].set_index('date', inplace=True)
                            plain_ret_list.append(plain_ret_dict[key])
                            plain_rank_dict[key].drop_duplicates(subset=['date'], keep='first', inplace=True)
                            plain_rank_dict[key].set_index('date', inplace=True)
                            plain_rank_list.append(plain_rank_dict[key])
                    ret_df = pd.concat(plain_ret_list, axis=1)
                    rank_df = pd.concat(plain_rank_list, axis=1)
                    ret_df.to_csv(
                        f'{path}/result/dataframe/buy&hold_sample_period_{period_i}_ret.csv',
                        index=True)
                    rank_df.to_csv(
                        f'{path}/result/dataframe/buy&hold_sample_period_{period_i}_rank.csv',
                        index=True)
                    period_i += 1


def fn(data_path, period_dict, f):
    df_i = pd.read_csv(f"{data_path}/{f}")
    if len(df_i) < 250:
        return None
    df_i['date'] = pd.to_datetime(df_i['date'])
    df_i.set_index('date', inplace=True)
    df_i['period'] = -1
    for k in period_dict.keys():
        t1 = period_dict[k][0]
        t2 = period_dict[k][1]
        df_i.loc[t1:t2, ['period']] = k
    return df_i


if __name__ == "__main__":
    # df = pd.DataFrame({'c1':[i for i in range(10)],'c2':['200'+str(i) for i in range(10)],'c3':[np.NaN if i%3==0 else 1 for i in range(10)]})
    # df['c2'] = pd.to_datetime(df['c2'])
    # f = pd.isna(df['c3'].iloc[3])
    # print(df['c1'])
    # print(df['c1'].pct_change(1))
    # print(f)
    # df.set_index('c2',inplace=True)
    # df1 = pd.DataFrame({'date':df['c2'],'c3':df['c3']})
    # print(df1)
    # print(df.loc[pd.to_datetime('1992-1-1'):pd.to_datetime('2003-1-1'),:])
    # print(list(df['c2']))
    # for i,item in enumerate([(i,2)for i in df['c2']]):
    #     print(i,item)
    # g = list(df.groupby('c2').groups.keys())
    # g.sort()
    # print(g)
    # df.sort_values(['c2', 'c1'], inplace=True)
    # df.reset_index(inplace=True,drop=True)
    # df['c4'] = df['c1']/df['c3']
    # print(df['c1'])
    # print(df['c1'].rolling(2).mean())
    b = Benchmark('./data_us')
    b.add_mom(5, 20, 60)
    b.gen_rank_ret('./archive/model set 1', './archive/model set 12')
    pass
