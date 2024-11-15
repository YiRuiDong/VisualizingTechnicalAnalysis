import pandas as pd
import multiprocessing as mp


def ma_sig(src_data: pd.DataFrame, n=5):
    try:
        TICKER = src_data['TICKER'].iloc[-1]
        data = src_data[['open', 'high', 'low', 'close', f'ma{n}']].copy(deep=True)
        data['sig'] = 0
        data.loc[data['close'] > data[f'ma{n}'], 'sig'] = 1
        data.loc[data['close'] < data[f'ma{n}'], 'sig'] = -1
        sig = data['sig'].shift(1).copy()
        sig.name = TICKER
        ret = data['close'].pct_change().copy()
        ret.name = TICKER
        print(sig)
        return sig, ret
    except Exception as e:
        print(e)
        return None, None


def boll_sig(src_data: pd.DataFrame):
    try:
        TICKER = src_data['TICKER'].iloc[-1]
        data = src_data[['open', 'high', 'low', 'close', 'bands_lower', 'bands_middle', 'bands_upper']].copy(deep=True)
        data['sig'] = 0
        data.loc[(data['close'] > data['bands_middle']) & (data['close'] < data['bands_upper']), 'sig'] = 1
        data.loc[(data['close'] < data['bands_middle']) & (data['close'] > data['bands_lower']), 'sig'] = -1
        sig = data['sig'].shift(1).copy()
        sig.name = TICKER
        ret = data['close'].pct_change().copy()
        ret.name = TICKER
        print(sig)
        return sig, ret
    except Exception as e:
        print(e)
        return None, None


def macd_sig(src_data: pd.DataFrame):
    try:
        TICKER = src_data['TICKER'].iloc[-1]
        data = src_data[['open', 'high', 'low', 'close', 'macd', 'macd_sign']].copy(deep=True)
        data['sig'] = 0

        data.loc[data['macd'] > data['macd_sign'], 'sig'] = 1
        data.loc[data['macd'] < data['macd_sign'], 'sig'] = -1
        sig = data['sig'].shift(1).copy()
        sig.name = TICKER
        ret = data['close'].pct_change().copy()
        ret.name = TICKER
        print(sig)
        return sig, ret
    except Exception as e:
        print(e)
        return None, None


def rsi_sig(src_data: pd.DataFrame, n=5, m=10):
    try:
        TICKER = src_data['TICKER'].iloc[-1]
        data = src_data[['open', 'high', 'low', 'close', f'rsi{n}', f'rsi{m}']].copy(deep=True)
        data['sig'] = 0
        data.loc[data[f'rsi{n}'] > data[f'rsi{m}'], 'sig'] = 1
        data.loc[data[f'rsi{n}'] < data[f'rsi{m}'], 'sig'] = -1
        sig = data['sig'].shift(1).copy()
        sig.name = TICKER
        ret = data['close'].pct_change().copy()
        ret.name = TICKER
        print(sig)
        return sig, ret
    except Exception as e:
        print(e)
        return None, None


def read_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    return df


def ma_fn(data_path):
    return ma_sig(read_data(data_path))


def macd_fn(data_path):
    return macd_sig(read_data(data_path))


def boll_fn(data_path):
    return boll_sig(read_data(data_path))


def rsi_fn(data_path):
    return rsi_sig(read_data(data_path))


def compute_ret(src_sig_table: pd.DataFrame, src_ret_table: pd.DataFrame):
    sig_table = src_sig_table.copy(deep=True).fillna(0)
    ret_table = src_ret_table.copy(deep=True).fillna(0)
    sig_table_norm = sig_table.apply(lambda x: x / x.abs().sum() if x.abs().sum() > 0 else x, axis=1)
    ret_table[ret_table.abs() > 10] = 0
    ret_series = (sig_table_norm * ret_table).sum(axis=1) - sig_table_norm.diff().abs().sum(axis=1) * 0.0001
    print((ret_series+1).cumprod())
    return ret_series


if __name__ == "__main__":
    import os
    import os.path as op
    import numpy as np

    file_list = [op.join('./data_us/tech', file) for file in os.listdir('./data_us/tech') if '.csv' in file]

    with mp.Pool(processes=4) as pool:
        ma_list = pool.map(ma_fn, file_list)
    ma_sig_table = pd.concat([i[0] for i in ma_list], axis=1)
    ma_ret_table = pd.concat([i[1] for i in ma_list], axis=1)
    ma_ret_series = compute_ret(ma_sig_table, ma_ret_table)
    ma_ret_series.name = 'ma_ret'
    del ma_list, ma_sig_table, ma_ret_table
    with mp.Pool(processes=4) as pool:
        boll_list = pool.map(boll_fn, file_list)
    boll_sig_table = pd.concat([i[0] for i in boll_list], axis=1)
    boll_ret_table = pd.concat([i[1] for i in boll_list], axis=1)
    boll_ret_series = compute_ret(boll_sig_table, boll_ret_table)
    boll_ret_series.name = 'boll_ret'
    del boll_list,boll_sig_table,boll_ret_table
    with mp.Pool(processes=4) as pool:
        macd_list = pool.map(macd_fn, file_list)
    macd_sig_table = pd.concat([i[0] for i in macd_list], axis=1)
    macd_ret_table = pd.concat([i[1] for i in macd_list], axis=1)
    macd_ret_series = compute_ret(macd_sig_table, macd_ret_table)
    macd_ret_series.name = 'macd_ret'
    del macd_list, macd_sig_table, macd_ret_table
    with mp.Pool(processes=4) as pool:
        rsi_list = pool.map(rsi_fn, file_list)
    rsi_sig_table = pd.concat([i[0] for i in rsi_list], axis=1)
    rsi_ret_table = pd.concat([i[1] for i in rsi_list], axis=1)
    rsi_ret_series = compute_ret(rsi_sig_table, rsi_ret_table)
    rsi_ret_series.name = 'rsi_ret'
    del rsi_list, rsi_ret_table, rsi_sig_table
    df = pd.concat([ma_ret_series, boll_ret_series, macd_ret_series, rsi_ret_series], axis=1)
    df.to_csv('./extra_benchmark/trad_tech.csv', index_label='date')
    # macd_ret_series.to_csv('./extra_benchmark/trad_tech.csv', index_label='date')
