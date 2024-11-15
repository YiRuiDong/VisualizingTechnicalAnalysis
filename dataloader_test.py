import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from os import path as op
from os import listdir
from PIL import Image
import re
import tqdm
from Data_generate import Data_generator
import DrawLib
import copy
import pandas as pd


class GetData(Dataset):
    def __init__(self, root_dir: str, start: int, data_window: int, N_day_after: int, flag: str, market: str):
        assert market in ['cn', 'us']
        assert flag in ['sample']
        self.data_generator = Data_generator(root_dir, market=market)
        self.data_generator.register_sample_csv_list(flag=flag)
        self.len_dict = {}
        self.flag = flag
        self.start = start
        self.window = data_window
        self.N_day = N_day_after
        self.tech_indicator = None
        self.item_dict = {}
        self.validating = False
        temp_list = copy.copy(self.data_generator.csv_name_list)
        bar = tqdm.tqdm(desc='initializing',total=len(temp_list),initial=0)
        for stock in temp_list:
            df = self.data_generator.read_data(stock, self.flag)
            if df is not None:
                self.len_dict[stock] = {}
                df: pd.DataFrame
                sample_group = df.groupby('sample_period')
                for g in sample_group:
                    sample_period = g[0]
                    data = g[1]
                    sample_len = len(data) - self.window - self.start - self.N_day
                    if sample_len > 0:
                        self.len_dict[stock][f"sample_period{sample_period}"] = (sample_period, sample_len)
                    else:
                        self.len_dict[stock][f"sample_period{sample_period}"] = (sample_period, -1)
            else:
                self.data_generator.csv_name_list.remove(stock)
            bar.update(1)
        if self.flag == 'sample':
            self.val_item_dict = {}
            for stock in self.len_dict.keys():
                for sample_period in self.len_dict[stock].keys():
                    if self.len_dict[stock][sample_period][1] != -1:
                        if sample_period == 'sample_period0':
                            if len(self.val_item_dict) == 0:
                                self.val_item_dict[self.len_dict[stock][sample_period][1]] = {'stock': stock,
                                                                                              'sample_period':self.len_dict[stock][sample_period][0]}
                            else:
                                self.val_item_dict[self.len_dict[stock][sample_period][1] + pre_val_length] = {
                                    'stock': stock,
                                    'sample_period': self.len_dict[stock][sample_period][0]}
                            pre_val_length = list(self.val_item_dict.keys())[-1]
                        else:
                            if len(self.item_dict) == 0:
                                self.item_dict[self.len_dict[stock][sample_period][1]] = {'stock': stock,
                                                                                          'sample_period': self.len_dict[stock][sample_period][0]}
                            else:
                                self.item_dict[self.len_dict[stock][sample_period][1] + pre_length] = {'stock': stock,
                                                                                                       'sample_period': self.len_dict[stock][sample_period][0]}
                            pre_length = list(self.item_dict.keys())[-1]
        else:
            self.val_item_dict = None

    def __len__(self):
        if self.flag == 'sample':
            if not self.validating:
                total_length = list(self.item_dict.keys())[-1]
            else:
                total_length = list(self.val_item_dict.keys())[-1]
            return total_length

        elif self.flag == 'test':
            return

    def register_tech(self, tech):
        self.tech_indicator = tech

    def set_validating_flag(self, flag: bool):
        self.validating = flag

    def __getitem__(self, item):
        if self.flag == 'sample':
            if not self.validating:
                for i, key in enumerate(self.item_dict.keys()):
                    if item < key:
                        if i != 0:
                            item -= list(self.item_dict.keys())[i - 1]
                        dfGet = self.data_generator.read_data(self.item_dict[key]['stock'], self.flag)
                        if dfGet is not None:
                            group = dfGet.groupby('sample_period')
                            df = group.get_group(self.item_dict[key]['sample_period']).copy()
                            df.reset_index(inplace=True)
                            df.drop(columns='index', inplace=True)
                            self.data_generator.df = df.copy()
                            data_zip = self.data_generator.sep_OHLC_Techs_Label(self.start + item, self.window,
                                                                                self.N_day,require_actual_ret=True)
                            img, status = DrawLib.DrawOHLCTechnical(224, data_zip['OHLC'], show=False,log_index=self.item_dict[key]['stock'],
                                                                    # image_out_path='temp',filename=f'{item%50}.bmp',
                                                                    MA=data_zip[
                                                                        'MA'] if 'MA' in self.tech_indicator else None,
                                                                    MACD=data_zip[
                                                                        'MACD'] if 'MACD' in self.tech_indicator else None,
                                                                    Vol=data_zip[
                                                                        'Vol'] if 'Vol' in self.tech_indicator else None,
                                                                    BOLL=data_zip[
                                                                        'BOLL'] if 'BOLL' in self.tech_indicator else None,
                                                                    RSI=data_zip[
                                                                        'RSI'] if 'RSI' in self.tech_indicator else None
                                                                    )
                            label = data_zip['label']
                            ret = float(data_zip['label_true'])
                            img_t = transforms.Resize((224, 224))(img)
                            img_tensor = transforms.ToTensor()(img_t)
                            img.close()
                            # label_list = [1. for i in range(2)]
                            # label_tensor = torch.tensor(np.diag(label_list)[label])
                            return img_tensor, label, status, ret
            else:
                for i, key in enumerate(self.val_item_dict.keys()):
                    if item < key:
                        if i != 0:
                            item -= list(self.val_item_dict.keys())[i - 1]
                        dfGet = self.data_generator.read_data(self.val_item_dict[key]['stock'])
                        if dfGet is not None:
                            group = dfGet.groupby('sample_period')
                            df = group.get_group(self.val_item_dict[key]['sample_period']).copy()
                            df.reset_index(inplace=True)
                            df.drop(columns='index', inplace=True)
                            self.data_generator.df = df.copy()
                            data_zip = self.data_generator.sep_OHLC_Techs_Label(self.start + item, self.window,
                                                                                self.N_day,require_actual_ret=True)
                            img, status = DrawLib.DrawOHLCTechnical(224, data_zip['OHLC'], show=False,log_index=self.val_item_dict[key]['stock'],
                                                                    # image_out_path='temp',filename=f'{item%50}.bmp',
                                                                    MA=data_zip[
                                                                        'MA'] if 'MA' in self.tech_indicator else None,
                                                                    MACD=data_zip[
                                                                        'MACD'] if 'MACD' in self.tech_indicator else None,
                                                                    Vol=data_zip[
                                                                        'Vol'] if 'Vol' in self.tech_indicator else None,
                                                                    BOLL=data_zip[
                                                                        'BOLL'] if 'BOLL' in self.tech_indicator else None,
                                                                    RSI=data_zip[
                                                                        'RSI'] if 'RSI' in self.tech_indicator else None
                                                                    )
                            label = data_zip['label']
                            ret = float(data_zip['label_true'])
                            img = transforms.Resize((224, 224))(img)
                            img_tensor = transforms.ToTensor()(img)
                            # label_list = [1. for i in range(2)]
                            # label_tensor = torch.tensor(np.diag(label_list)[label])
                            return img_tensor, label, status, ret



if __name__ == '__main__':
    pass
