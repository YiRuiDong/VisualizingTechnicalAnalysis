from decimal import Decimal
import os
import copy
import random
import numpy
import pandas as pd
import tqdm
import tulipy as ti
import numpy as np
from os import path as op
from os import makedirs, listdir
import re
from torch import save as cache_save
from torch import load as cache_load
import multiprocessing as mp
import DrawLib
import shutil


class Data_generator:
    def __init__(self, raw_data_path: str = './', market='cn', with_no_init: bool = False):
        assert market in ['cn', 'us']
        self.market = market
        if not with_no_init:
            self.raw_data_path = raw_data_path
            self.csv_name_list = []
            self.MA_method = None
            self.processed_data_path = None
            has_cache = False
            if op.exists('./cache/csv_name_list.pt'):
                t1 = op.getmtime(self.raw_data_path)
                t2 = op.getmtime('./cache/csv_name_list.pt')
                if t1 <= t2:
                    has_cache = True
                    self.csv_name_list = cache_load('./cache/csv_name_list.pt')
            if not has_cache:
                File_list = listdir(self.raw_data_path)
                for F in File_list:
                    if F[-4:] == '.csv':
                        self.csv_name_list.append(F)
                cache_save(self.csv_name_list, './cache/csv_name_list.pt')
            self.df = None
        else:
            self.csv_name_list = None
            self.df = None

    def __len__(self):
        return len(self.csv_name_list)

    def div_sample_test_sets(self, start_date, end_date, pieces=3):
        assert type(start_date) is str
        assert type(end_date) is str
        assert type(pieces) is int
        ts = pd.date_range(start=start_date, end=end_date, periods=pieces + 1)
        train_period = []
        test_period = []
        for i, t in enumerate(ts):
            if i < len(ts) - 1:
                if i % 2 == 0:
                    train_period.append((ts[i], ts[i + 1]))
                else:
                    test_period.append((ts[i], ts[i + 1]))
        total = train_period + test_period
        random.shuffle(total)
        train_period = total[:len(train_period)]
        test_period = total[len(train_period):]
        with open("./log/sample_period.txt", 'a') as log:
            print('train_period', file=log)
            for i in train_period:
                print(f'{i[0].date()}_{i[1].date()}', file=log)
            print('test_period', file=log)
            for i in test_period:
                print(f'{i[0].date()}_{i[1].date()}', file=log)
        if not op.exists(op.join(self.raw_data_path, 'sample')):
            makedirs(op.join(self.raw_data_path, 'sample'))
        else:
            shutil.rmtree(op.join(self.raw_data_path, 'sample'))
            makedirs(op.join(self.raw_data_path, 'sample'))
        if not op.exists(op.join(self.raw_data_path, 'test')):
            makedirs(op.join(self.raw_data_path, 'test'))
        else:
            shutil.rmtree(op.join(self.raw_data_path, 'test'))
            makedirs(op.join(self.raw_data_path, 'test'))
        bar = tqdm.tqdm(total=len(self.csv_name_list), desc='separating_test_and_train', initial=0)
        for i, stock in enumerate(self.csv_name_list):
            self.df = pd.read_csv(op.join(self.raw_data_path, stock))
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('date', inplace=True)
            train_df = []
            test_df = []
            self.df['sample_period'] = 0
            for j, t in enumerate(train_period):
                self.df.loc[t[0]:t[1], ['sample_period']] = j
                if len(self.df.loc[t[0]:t[1], :]) > 0:
                    if len(train_df) == 0:
                        train_df = self.df.loc[t[0]:t[1], :].copy()
                    else:
                        train_df = pd.concat([train_df, self.df.loc[t[0]:t[1]]], axis=0)
            for j, t in enumerate(test_period):
                self.df.loc[t[0]:t[1], ['sample_period']] = j
                if len(self.df.loc[t[0]:t[1], :]) > 0:
                    if len(test_df) == 0:
                        test_df = self.df.loc[t[0]:t[1], :].copy()
                    else:
                        test_df = pd.concat([test_df, self.df.loc[t[0]:t[1]]], axis=0)
            if len(train_df) > 0:
                train_df.reset_index(inplace=True)
                train_df.rename(columns={'index': 'date'}, inplace=True)
                train_df['date'] = train_df['date'].apply(lambda x: x.date())
                train_df.to_csv(op.join(self.raw_data_path, 'sample', stock), index=False)
            if len(test_df) > 0:
                test_df.reset_index(inplace=True)
                test_df.rename(columns={'index': 'date'}, inplace=True)
                test_df['date'] = test_df['date'].apply(lambda x: x.date())
                test_df.to_csv(op.join(self.raw_data_path, 'test', stock), index=False)
            bar.update(1)

    def addTechnicalIndicators(self, stock_name_or_index, exclude_ST=True, save=False, **technical_params):
        assert type(stock_name_or_index) in [int, str]
        if type(stock_name_or_index) == str:
            if self.market == 'cn':
                check = re.match('(([0]|[3]|[6])[0-9]{5}.?[Ss][HhZz](\.csv)?)', stock_name_or_index)
                if not check:
                    raise Exception(
                        'can\'t find stock data, you can have access to data by index or stock code like 000001SZ, 000001sz, '
                        '000001.SZ or 000001.sz.csv etc.')
                else:
                    stock_code = stock_name_or_index[:6]
                    for idx in range(len(self.csv_name_list)):
                        if self.csv_name_list[idx][:6] == stock_code:
                            self.df = pd.read_csv(op.join(self.raw_data_path, self.csv_name_list[idx]))
                            file_name = self.csv_name_list[idx][:9]
                            break
            else:
                stock_code = stock_name_or_index[:-4]
                find = False
                for idx in range(len(self.csv_name_list)):
                    if self.csv_name_list[idx][:-4] == stock_code:
                        self.df = pd.read_csv(op.join(self.raw_data_path, self.csv_name_list[idx]))
                        file_name = self.csv_name_list[idx][:-4]
                        find = True
                        break
                if not find:
                    raise Exception('can\'t find stock data, you can try access data through item index.')
        else:
            self.df = pd.read_csv(op.join(self.raw_data_path, self.csv_name_list[stock_name_or_index]))
            if self.market == 'cn':
                file_name = self.csv_name_list[stock_name_or_index][:9]
            else:
                file_name = self.csv_name_list[stock_name_or_index][:-4]
        for idx in range(len(self.df)):
            if self.market == 'cn':
                if exclude_ST:
                    if self.df.loc[idx, 'isST'] == 1:
                        print('st stock excluded!')
                        return 0
                if self.df.loc[idx, 'volume'] == 0 or self.df.loc[idx, 'tradestatus'] == 0:
                    self.df.drop(index=idx, inplace=True)
            else:
                if (self.df.loc[idx, 'volume'] == 0) | (self.df.loc[idx, 'close'] <= 0) | (
                        self.df.loc[idx, 'open'] <= 0) | (self.df.loc[idx, 'open'] <= 0) | (
                        self.df.loc[idx, 'high'] <= 0) | (
                        self.df.loc[idx, 'low'] <= 0) | (pd.isna(self.df.loc[idx, 'open'])) | (
                        pd.isna(self.df.loc[idx, 'high'])) | (pd.isna(self.df.loc[idx, 'close'])) | (
                        pd.isna(self.df.loc[idx, 'low']) | (np.inf in self.df.loc[idx, :])):
                    self.df.drop(index=idx, inplace=True)
        self.df.reset_index(inplace=True)
        self.df.drop(columns='index', inplace=True)
        if len(self.df) < 250:
            print('\nStock trading date is too short, jump this one!')
            return 0
        tech_args = (1,)
        for indicator in technical_params.keys():
            if indicator in ['save', 'excludeST']:
                continue
            assert indicator in ['MA', 'ma', 'Ma', 'RSI', 'rsi', 'Rsi', 'MACD', 'Macd', 'macd', 'BOLL', 'boll', 'Boll']
            file_name += "_" + str(indicator)
            if indicator in ['MA', 'ma', 'Ma']:
                assert type(technical_params[indicator]) in [tuple]
                close_array = self.df['close'].to_numpy()
                if type(technical_params[indicator]) == int:
                    if not self.MA_method:
                        print(
                            'default method for calculating moving average is simple moving average, you may change it through \"setMA\"')
                        ma_array = ti.sma(close_array, technical_params[indicator])
                        ma_array = np.concatenate(
                            (np.array([0 for i in range(technical_params[indicator] - 1)]), ma_array))
                        ma_df = pd.DataFrame({'ma'f'{technical_params[indicator]}': ma_array})
                        self.df = pd.concat([self.df, ma_df], axis=1)
                    else:
                        if self.MA_method == 'vmma':
                            pass
                        elif self.MA_method == 'ema':
                            pass
                        elif self.MA_method == 'hma':
                            pass
                else:
                    for period in technical_params[indicator]:
                        assert type(period) == int
                        if not self.MA_method:
                            ma_array = ti.sma(close_array, period)
                            ma_array = np.concatenate((np.array([0 for i in range(period - 1)]), ma_array))
                            ma_df = pd.DataFrame({'ma'f'{period}': ma_array})
                            self.df = pd.concat([self.df, ma_df], axis=1)
                        else:
                            if self.MA_method == 'vmma':
                                pass
                            elif self.MA_method == 'ema':
                                pass
                            elif self.MA_method == 'hma':
                                pass

            elif indicator in ['RSI', 'rsi', 'Rsi']:
                assert type(technical_params[indicator]) in [tuple]
                close_array = self.df['close'].to_numpy()
                if type(technical_params[indicator]) == int:
                    rsi_array = ti.rsi(close_array, technical_params[indicator])
                    rsi_array = np.concatenate((np.array([0 for i in range(period)]), rsi_array))
                    rsi_df = pd.DataFrame({'rsi'f'{technical_params[indicator]}': rsi_array})
                    self.df = pd.concat([self.df, rsi_df], axis=1)
                else:
                    for period in technical_params[indicator]:
                        rsi_array = ti.rsi(close_array, period)
                        rsi_array = np.concatenate((np.array([0 for i in range(period)]), rsi_array))
                        rsi_df = pd.DataFrame({'rsi'f'{period}': rsi_array})
                        self.df = pd.concat([self.df, rsi_df], axis=1)

            elif indicator in ['MACD', 'Macd', 'macd']:
                assert type(technical_params[indicator]) in [tuple]
                assert len(technical_params[indicator]) == 3
                for param in technical_params[indicator]:
                    assert type(param) == int
                short, long, signal = technical_params[indicator][0], technical_params[indicator][1], \
                    technical_params[indicator][2]
                close_array = self.df['close'].to_numpy()
                macd_array, macd_signal_array, macd_histogram_array = ti.macd(close_array, short, long, signal)
                macd_array = np.concatenate((np.array([0 for i in range(long - 1)]), macd_array))
                macd_signal_array = np.concatenate((np.array([0 for i in range(long - 1)]), macd_signal_array))
                macd_histogram_array = np.concatenate((np.array([0 for i in range(long - 1)]), macd_histogram_array))
                macd_df = pd.DataFrame(
                    {'macd': macd_array, 'macd_sign': macd_signal_array, 'macd_hist': macd_histogram_array})
                self.df = pd.concat([self.df, macd_df], axis=1)

            elif indicator in ['BOLL', 'boll', 'Boll']:
                assert type(technical_params[indicator]) in [tuple]
                assert len(technical_params[indicator]) == 2
                for item in technical_params[indicator]:
                    assert type(item) == int
                period, std = technical_params[indicator][0], technical_params[indicator][1]
                close_array = self.df['close'].to_numpy()
                bbands_lower_array, bbands_middle_array, bbands_upper_array = ti.bbands(close_array, period, std)
                bbands_lower_array = np.concatenate((np.array([0 for i in range(period - 1)]), bbands_lower_array))
                bbands_middle_array = np.concatenate((np.array([0 for i in range(period - 1)]), bbands_middle_array))
                bbands_upper_array = np.concatenate((np.array([0 for i in range(period - 1)]), bbands_upper_array))
                boll_df = pd.DataFrame({'bands_lower': bbands_lower_array, 'bands_middle': bbands_middle_array,
                                        'bands_upper': bbands_upper_array})
                self.df = pd.concat([self.df, boll_df], axis=1)
            tech_args += technical_params[indicator]
        if save:
            file_name += '.csv'
            self.processed_data_path = 'tech'
            if not op.exists(op.join(self.raw_data_path, self.processed_data_path)):
                makedirs(op.join(self.raw_data_path, self.processed_data_path))
            self.df = self.df.iloc[max(tech_args):, :]
            if op.exists(op.join(self.raw_data_path, self.processed_data_path, file_name)):
                pass
            else:
                self.df.to_csv(op.join(self.raw_data_path, self.processed_data_path, file_name), index=False)
        return 1

    def sep_OHLC_Techs_Label(self, start, window, N_days_after, require_actual_ret=False, require_true_date=False):
        if not len(self.df.columns) > 4:
            raise Exception('It\'s seems that you don\'t have technical indicator in your data!')
        if len(self.df) < start + window + N_days_after:
            print(len(self.df),start + window + N_days_after)
            print('window exceeds the data length!!')
            return 0
        OHLC = copy.deepcopy(self.df.loc[start:start + window - 1, ['open', 'high', 'low', 'close']])
        OHLC.reset_index(inplace=True)
        OHLC.drop(columns='index', inplace=True)
        if self.market == 'cn':
            Tech = copy.deepcopy(self.df.iloc[start:start + window, 13:])
        else:
            if 'sample_period' in self.df.columns:
                Tech = copy.deepcopy(self.df.iloc[start:start + window, 11:-1])
            else:
                Tech = copy.deepcopy(self.df.iloc[start:start + window, 11:])
        Tech.reset_index(inplace=True)
        Tech.drop(columns='index', inplace=True)
        Volume = copy.deepcopy(pd.DataFrame(self.df.loc[start:start + window - 1, 'volume']))
        # if self.df.loc[start + window - 1, 'close'] == 0 or type(
        #         self.df.loc[start + window - 1, 'close']) is not numpy.float64 or (
        #         not np.isfinite(self.df.loc[start + window - 1, 'close'])):
        #     with open('./log/invalid_value.txt', 'a') as file:
        #         print('value in', self.df.loc[start + window - 1, 'PERMNO'], f'line {start + window - 1}',
        #               'is invalid!', file=file)
        #     raise Exception('invalid value detected! Check integrity of your data!')
        N_days_ret = self.df.loc[start + window + N_days_after - 1, 'close'] / self.df.loc[start + window - 1, 'close']
        label = 1 if N_days_ret > 1 else 0

        if require_actual_ret:
            label_true = Decimal(str((N_days_ret-1)*100)).quantize(Decimal('.01'))
        # annual_ret = np.power(N_days_ret, 360 / N_days_after) - 1
        # ret_rank = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
        # label = None
        # for i in range(len(ret_rank) - 1):
        #     if ret_rank[i] <= annual_ret < ret_rank[i + 1]:
        #         label = i + 1
        #     elif ret_rank[0] > annual_ret:
        #         label = 0
        #     elif ret_rank[len(ret_rank) - 1] <= annual_ret:
        #         label = 9
        RSI = []
        MACD = []
        BOLL = []
        MA = []
        RSI_df = None
        MACD_df = None
        MA_df = None
        BOLL_df = None
        for indicator in Tech.columns:
            if 'rsi' in indicator:
                RSI.append(Tech.loc[:, indicator])
            elif 'macd' in indicator:
                MACD.append(Tech.loc[:, indicator])
            elif 'bands' in indicator:
                BOLL.append(Tech.loc[:, indicator])
            elif 'ma' in indicator[:2]:
                MA.append(Tech.loc[:, indicator])
        if len(RSI) > 0:
            RSI_df = pd.concat(RSI, axis=1)
        if len(MACD) > 0:
            MACD_df = pd.concat(MACD, axis=1)
        if len(MA) > 0:
            MA_df = pd.concat(MA, axis=1)
        if len(BOLL) > 0:
            BOLL_df = pd.concat(BOLL, axis=1)
        if len(RSI_df) == 0 | len(BOLL_df) == 0 | len(MA_df) == 0 | len(MACD_df) == 0:
            raise Exception('empty dataframe!')
        if require_true_date:
            DATE = pd.DataFrame(copy.deepcopy(self.df.loc[start:start + window - 1, 'date']))
            DATE.reset_index(inplace=True)
            DATE.drop(columns='index', inplace=True)
            if require_actual_ret:
                return {'OHLC': OHLC, 'Vol': Volume, 'RSI': RSI_df, 'MACD': MACD_df, 'BOLL': BOLL_df, 'MA': MA_df,
                        'label': label,'label_true':label_true, 'date': DATE}
            else:
                return {'OHLC': OHLC, 'Vol': Volume, 'RSI': RSI_df, 'MACD': MACD_df, 'BOLL': BOLL_df, 'MA': MA_df,
                        'label': label, 'date': DATE}
        else:
            if require_actual_ret:
                return {'OHLC': OHLC, 'Vol': Volume, 'RSI': RSI_df, 'MACD': MACD_df, 'BOLL': BOLL_df, 'MA': MA_df,
                        'label': label, 'label_true': label_true}
            else:
                return {'OHLC': OHLC, 'Vol': Volume, 'RSI': RSI_df, 'MACD': MACD_df, 'BOLL': BOLL_df, 'MA': MA_df,
                        'label': label}

    def read_data(self, stock_name_or_index, flag='sample'):
        assert type(stock_name_or_index) in [int, str]
        if type(stock_name_or_index) == str:
            if self.market == 'cn':
                check = re.match('(([0]|[3]|[6])[0-9]{5}.?[Ss][HhZz](\.csv)?)', stock_name_or_index)
                if not check:
                    raise Exception(
                        'can\'t find stock data, you can have access to data by index or stock code like 000001SZ, 000001sz, '
                        '000001.SZ or 000001.sz.csv etc.')
                else:
                    stock_code = stock_name_or_index[:6]
                    for idx in range(len(self.csv_name_list)):
                        if self.csv_name_list[idx][:6] == stock_code:
                            if op.exists(op.join(self.raw_data_path, flag, self.csv_name_list[idx])):
                                self.df = pd.read_csv(op.join(self.raw_data_path, flag, self.csv_name_list[idx]))
                                return self.df
                            else:
                                return None
            else:
                stock_code = stock_name_or_index[:-4]
                find = False
                for idx in range(len(self.csv_name_list)):
                    if self.csv_name_list[idx][:-4] == stock_code:
                        self.df = pd.read_csv(op.join(self.raw_data_path, flag, self.csv_name_list[idx]))
                        find = True
                        return self.df
                if not find:
                    return None
        else:
            if op.exists(op.join(self.raw_data_path, flag, self.csv_name_list[stock_name_or_index])):
                self.df = pd.read_csv(op.join(self.raw_data_path, flag, self.csv_name_list[stock_name_or_index]))
                return self.df
            else:
                return None

    def register_sample_csv_list(self, minimum_length=250, flag='sample', existed_list=None):
        if existed_list is None:
            has_cache = False
            if op.exists(f'./cache/csv_name_list_{flag}.pt'):
                t1 = op.getmtime(op.join(self.raw_data_path, flag))
                t2 = op.getmtime(f'./cache/csv_name_list_{flag}.pt')
                if t1 <= t2:
                    has_cache = True
                    self.csv_name_list = cache_load(f'./cache/csv_name_list_{flag}.pt')
            if not has_cache:
                self.csv_name_list = os.listdir(op.join(self.raw_data_path, flag))
                temp = []
                for i in range(len(self.csv_name_list)):
                    self.read_data(i)
                    if len(self.df) > minimum_length:
                        temp.append(self.csv_name_list[i])
                self.csv_name_list = temp
                cache_save(self.csv_name_list, f'./cache/csv_name_list_{flag}.pt')
        else:
            self.csv_name_list = existed_list
        self.df = None

if __name__ == '__main__':
    pass
