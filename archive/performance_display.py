import copy
import os
import decimal
import pandas as pd
import os.path as op
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from scipy import stats
import re
import matplotlib.patches as mpatches
from scipy.stats import skew, kurtosis, norm
import statsmodels.api as sm


def show_ret(model_name: str, p: int, q: float, dq: float, HmL: bool, exclude_extreme: bool, set_num, buy_hold=False,
             trans_fee=0.0001):
    df_ret = pd.read_csv(f'./model set {set_num}/result/dataframe/{model_name}_sample_period_{p}_ret.csv')
    df_rank = pd.read_csv(f'./model set {set_num}/result/dataframe/{model_name}_sample_period_{p}_rank.csv')
    date = pd.to_datetime(df_ret['date'])
    df_ret.drop(columns=['date'], inplace=True)
    df_rank.drop(columns=['date'], inplace=True)
    if exclude_extreme:
        df_ret[df_ret.abs() > 1000] = pd.NA
    if not buy_hold:
        # df_rank = df_rank.rank(axis=1, method='first')
        df_rank = df_rank.apply(lambda x: (x - x.median()) / x.abs().sum(), axis=1)
        qt1 = df_rank.quantile(float(Decimal(str(q))), axis=1)
        qt2 = df_rank.quantile(float(Decimal(str(q)) + Decimal(str(dq))), axis=1)
        if not HmL:
            sign = -1 if q < 0.5 else 1
            for i in range(len(qt1)):
                df_rank.loc[i, df_rank.iloc[i, :] <= qt1[i]] = 0
                df_rank.loc[i, df_rank.iloc[i, :] > qt2[i]] = 0
        else:
            sign = 1
            mirror_qt1 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(q))), axis=1)
            mirror_qt2 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(q)) - Decimal(str(dq))), axis=1)
            for i in range(len(qt1)):
                df_rank.loc[i, df_rank.iloc[i, :] < mirror_qt2[i]] = 0
                df_rank.loc[i, (df_rank.iloc[i, :] <= qt1[i]) & (mirror_qt1[i] <= df_rank.iloc[i, :])] = 0
                df_rank.loc[i, df_rank.iloc[i, :] > qt2[i]] = 0

        scale = df_rank.abs().sum(axis=1)

        for i in range(len(df_rank.columns)):
            df_rank.iloc[:, i] /= scale * 5
        ret = (df_rank * df_ret).sum(axis=1) * 0.01 * sign
        std = ret.std() * np.sqrt(250)

        cum_ret = ((df_rank * df_ret * 0.01 * sign).sum(axis=1) - trans_fee * df_rank.diff(5).abs().sum(
            axis=1) + 1).cumprod()
        df = pd.concat([cum_ret, date], axis=1)
        df.set_index('date', inplace=True)
        cum_ret = df.iloc[:, 0]

        cum = cum_ret.iloc[-1]
        annual = np.power(cum, 250 / len(cum_ret)) - 1
        print('annual std:', std)
        print('cumulative return:', cum)
        print('annualized return:', annual)

    else:
        df_rank = df_rank.apply(lambda x: x / x.sum(), axis=1)
        row_sum = 0
        i = 0
        while row_sum == 0:
            row_sum = df_rank.iloc[i, :].sum()
            i += 1
        remove_c = []
        for j in range(len(df_rank.iloc[i, :])):
            if pd.isna(df_rank.iloc[i, j]):
                remove_c.append(df_rank.columns[j])
        df_rank.drop(columns=remove_c, inplace=True)
        df_ret.drop(columns=remove_c, inplace=True)
        ret = (df_rank * df_ret).sum(axis=1) * 0.01
        std = ret.std() * np.sqrt(250)
        cum_ret = ((df_rank * df_ret * 0.01).sum(axis=1) + 1).cumprod()
        df = pd.concat([cum_ret, date], axis=1)
        df.set_index('date', inplace=True)
        cum_ret = df.iloc[:, 0]

        cum = cum_ret.iloc[-1]
        annual = np.power(cum, 250 / len(cum_ret)) - 1

        print('annual std:', std)
        print('cumulative return:', cum)
        print('annualized return:', annual)
    # plt.grid()
    # plt.xlabel('date')
    # plt.ylabel('ret')
    # plt.title(f'{model_name} test period {p}')
    # plt.plot(cum_ret)
    # plt.show()
    # plt.close()

    return std, annual, len(ret)


def show_general_ret():
    files = os.listdir('./')
    model_sets = []
    bond_data = pd.read_csv('../US TREASURY.csv')
    us_bill = bond_data[['Time Period', '3M']]
    us_bill.dropna(inplace=True)
    us_bill.reset_index(inplace=True, drop=True)
    for i in range(len(us_bill)):
        if us_bill.loc[i, '3M'] == 'ND':
            us_bill.drop(index=i, inplace=True)
    us_bill.reset_index(inplace=True, drop=True)
    us_bill['Time Period'] = pd.to_datetime(us_bill['Time Period'])
    us_bill.set_index('Time Period', inplace=True)
    for f in files:
        if 'model set' in f and '.' not in f:
            model_sets.append(f)
    stat_dict = {'model set': [], 'model name': [], 'ret': [], 'stddev': [], 'SR': []}
    para_list = []
    num_trials = 22
    for s in model_sets:
        test_period = []
        with open(f'./{s}/sample_period.txt', 'r') as f:
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
                    t1 = pd.to_datetime(line.split('_')[0])
                    t2 = pd.to_datetime(line.split('_')[1])
                    test_period.append((t1, t2))
        names = os.listdir(f'./{s}/result/dataframe')
        model_list = []
        counter = 0
        rf_dict = {}
        for t in test_period:
            rf_dict[counter] = us_bill.loc[t[0]:t[1], '3M'].astype(float).mean()
            counter += 1

        for n in names:
            model_name = n.split('_')[0]
            if 'swin' in model_name and model_name not in model_list:
                model_list.append(model_name)

        emc = 0.5772156649
        for m_n in model_list:
            print(s, m_n)
            df_list = []
            for p in range(5):
                df_ret = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_ret.csv')
                df_rank = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_rank.csv')
                df_ret.drop(columns=['date'], inplace=True)
                df_rank.drop(columns=['date'], inplace=True)
                df_ret[df_ret.abs() > 1000] = pd.NA
                df_rank = df_rank.apply(lambda x: (x - x.median()) / x.abs().sum(), axis=1)

                scale = df_rank.abs().sum(axis=1)

                for i in range(len(df_rank.columns)):
                    df_rank.iloc[:, i] /= scale * 5

                ret = (df_rank * df_ret).sum(axis=1) * 0.01 - 0.0001 * df_rank.diff(5).abs().sum(axis=1)
                df_list.append(ret)

            df = pd.concat(df_list)
            std = df.std() * np.sqrt(250)
            mean = np.power(df.mean() + 1, 250) - 1

            n = df.count()

            sk = skew(df.to_numpy())
            kurt = kurtosis(df.to_numpy())

            low99 = stats.t.interval(0.99, n - 1, mean, std)[0]
            low95 = stats.t.interval(0.95, n - 1, mean, std)[0]
            low90 = stats.t.interval(0.90, n - 1, mean, std)[0]
            stat_dict['model name'].append(m_n)
            stat_dict['ret'].append(str(Decimal(mean * 100).quantize(Decimal('0.01'))) + "%")
            stat_dict['stddev'].append(str(Decimal(std * 100).quantize(Decimal('0.01'))) + "%")
            stat_dict['model set'].append(int(s.split(' ')[-1]))
            rf_mean = 0
            for k in rf_dict.keys():
                rf_mean += rf_dict[k]
            rf_mean /= 500
            sr = (mean - rf_mean) / std

            stat_dict['SR'].append(str(Decimal(sr).quantize(Decimal('0.01'))))

            para_dict = {}

            para_dict['sk'] = sk
            para_dict['kurt'] = kurt
            para_dict['T'] = n
            para_dict['SR'] = sr

            print(para_dict)

            para_list.append(para_dict)

            if low99 > 0:
                stat_dict['ret'][-1] = '***' + stat_dict['ret'][-1]
            elif low95 > 0:
                stat_dict['ret'][-1] = '**' + stat_dict['ret'][-1]
            elif low90 > 0:
                stat_dict['ret'][-1] = '*' + stat_dict['ret'][-1]
            if low99 - rf_mean > 0:
                stat_dict['SR'][-1] = '***' + stat_dict['SR'][-1]
            elif low95 - rf_mean > 0:
                stat_dict['SR'][-1] = '**' + stat_dict['SR'][-1]
            elif low90 - rf_mean > 0:
                stat_dict['SR'][-1] += '*' + stat_dict['SR'][-1]

    def norm_inv(x):
        return -norm.isf(x)

    def calculate_DSR(data_list):
        dsr_array = []
        sr_array = [data['SR'] if data['SR'] > 0 else 0 for data in data_list]
        for data_dict in data_list:
            SR_nonann = data_dict['SR'] / np.sqrt(250)
            if SR_nonann < 0:
                dsr_array.append('-')
            else:
                sr_array = np.array(sr_array)
                var_SR_nonann = sr_array.var() / 250
                SR_nonann_0 = np.sqrt(var_SR_nonann) * (
                        (1 - emc) * norm_inv(1 - 1 / num_trials) + emc * norm_inv(1 - 1 / (np.exp(1) * num_trials)))
                T = data_dict['T']
                SK = data_dict['sk']
                KT = data_dict['kurt']
                x = (SR_nonann - SR_nonann_0) * np.sqrt(T - 1) / np.sqrt(
                    1 - SK * SR_nonann + (KT - 1) / 4 * np.square(SR_nonann))
                DSR = norm.cdf(x)
                dsr_array.append(str(Decimal(DSR).quantize(Decimal('0.0001'))))
        return dsr_array

    stat_df = pd.DataFrame(stat_dict)

    stat_df['DSR'] = calculate_DSR(para_list)

    print(stat_df['DSR'])

    stat_df.sort_values('model set', inplace=True)

    stat_df.to_csv(f'../statistics/overall_new1.csv', index=False)


def show_details(model_dict: dict):
    model_sets = list(model_dict.keys())
    bond_data = pd.read_csv('../US TREASURY.csv')
    us_bill = bond_data[['Time Period', '3M']]
    us_bill.dropna(inplace=True)
    us_bill.reset_index(inplace=True, drop=True)
    for i in range(len(us_bill)):
        if us_bill.loc[i, '3M'] == 'ND':
            us_bill.drop(index=i, inplace=True)
    us_bill.reset_index(inplace=True, drop=True)
    us_bill['Time Period'] = pd.to_datetime(us_bill['Time Period'])
    us_bill.set_index('Time Period', inplace=True)
    for s in model_sets:
        test_period = []
        with open(f'./{s}/sample_period.txt', 'r') as f:
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
                    t1 = pd.to_datetime(line.split('_')[0])
                    t2 = pd.to_datetime(line.split('_')[1])
                    test_period.append((t1, t2))
        rf_dict = {}
        counter = 0
        for t in test_period:
            rf_dict[counter] = us_bill.loc[t[0]:t[1], '3M'].astype(float).mean()
            counter += 1
        rank_array = np.linspace(0.0, 0.9, 10)
        df_list = []
        model_list = model_dict[s]
        for p in range(5):
            data_p = []
            lv1 = []
            lv2 = []
            rf_mean = rf_dict[p]
            for model_name in model_list:
                data_m = []
                m_cols = ['ret', 'SR', 'stddev']
                for c in m_cols: lv1.append(c)
                for c in m_cols: lv2.append(model_name)
                for q in rank_array:
                    if model_name == 'buy&hold':
                        if q == 0:
                            std, r, n = show_ret(f'{model_name}', p, q, 0.1, False, True, set_num=s.split(' ')[-1],
                                                 buy_hold=True)
                            copy_tuple = copy.deepcopy((std, r, n))
                        else:
                            std, r, n = copy_tuple[0], copy_tuple[1], copy_tuple[2]
                    else:
                        std, r, n = show_ret(f'{model_name}', p, q, 0.1, False, True, set_num=s.split(' ')[-1])
                    low99 = stats.t.interval(0.99, n - 1, r, std)[0]
                    low95 = stats.t.interval(0.95, n - 1, r, std)[0]
                    low90 = stats.t.interval(0.90, n - 1, r, std)[0]
                    rf_mean /= 100
                    sharpe = (r - rf_mean) / std
                    std = str(Decimal(std * 100).quantize(Decimal('0.01'))) + '%'
                    if low99 > 0:
                        r = '***' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'

                    elif low95 > 0:
                        r = '**' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'

                    elif low90 > 0:
                        r = '*' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                    else:
                        r = str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                    if low99 - rf_mean > 0:
                        sharpe = '***' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                    elif low95 - rf_mean > 0:
                        sharpe = '**' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                    elif low90 - rf_mean > 0:
                        sharpe = '*' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                    else:
                        sharpe = str(Decimal(sharpe).quantize(Decimal('0.01')))
                    data_m.append([r, sharpe, std])
                if model_name == 'buy&hold':
                    std, r, n = show_ret(f'{model_name}', p, 0.5, 0.1, False, True, set_num=s.split(' ')[-1],
                                         buy_hold=True)
                else:
                    std, r, n = show_ret(f'{model_name}', p, 0.9, 0.1, True, True, set_num=s.split(' ')[-1])
                rf_mean /= 100
                low99 = stats.t.interval(0.99, n - 1, r, std)[0]
                low95 = stats.t.interval(0.95, n - 1, r, std)[0]
                low90 = stats.t.interval(0.90, n - 1, r, std)[0]
                sharpe = (r - rf_mean) / std
                std = str(Decimal(std * 100).quantize(Decimal('0.01'))) + '%'
                if low99 > 0:
                    r = '***' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'

                elif low95 > 0:
                    r = '**' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'

                elif low90 > 0:
                    r = '*' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'

                else:
                    r = str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                if low99 - rf_mean > 0:
                    sharpe = '***' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                elif low95 - rf_mean > 0:
                    sharpe = '**' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                elif low90 - rf_mean > 0:
                    sharpe = '*' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                else:
                    sharpe = str(Decimal(sharpe).quantize(Decimal('0.01')))
                data_m.append([r, sharpe, std])
                data_p.append(np.array(data_m))
            data_p = np.concatenate(data_p, axis=1)
            df = pd.DataFrame(data_p, index=[i if i != 11 else 'H-L' for i in range(1, 12)], columns=[lv2, lv1])
            df_list.append(df)
        with pd.ExcelWriter(f'./{s}.xlsx') as f:
            for p in range(len(df_list)):
                df_list[p].to_excel(f,
                                    sheet_name=f'{str(test_period[p][0].date()) + "_" + str(test_period[p][1].date())}')


def show_scatter(*model_performance_table_path: str):
    data_dict = {}
    for path in model_performance_table_path:
        df_dict = pd.read_excel(path, sheet_name=None, index_col=0)
        for sheet_name in df_dict.keys():
            df = df_dict[sheet_name]
            for c in df.columns:
                if 'Unnamed' not in c:
                    model_name = c
                if 'SR' in df.loc[:, c].unique():
                    if model_name not in data_dict.keys():
                        data_dict[model_name] = []
                    sharp = float(re.sub('\*', '', df.loc['H-L', c]))
                    if sharp not in data_dict[model_name]:
                        data_dict[model_name].append([sharp, sheet_name])
    x = []
    y = []
    label = []
    year = []
    data_dict = dict(sorted(data_dict.items()))
    print(data_dict)
    if 'buy&hold' in data_dict.keys(): data_dict.pop('buy&hold')
    for i, k in enumerate(data_dict):
        for j in data_dict[k]:
            x.append(i)
            y.append(j[0])
            j[1] = re.sub('_', '-', re.sub('-', '/', j[1]))
            year.append(j[1])
        label.append(k)
    year = sorted(year)
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', '+', 'D', 'd', 'x', '|', '_']
    colorlist = matplotlib.colormaps['tab10'].resampled(10)
    color_dict = {}
    counter = 0
    for time in year:
        if time not in color_dict.keys():
            color_dict[time] = counter
            counter += 1
    plt.figure(figsize=(8, 8), layout='tight')
    plt.grid()
    for i, k in enumerate(data_dict):
        plt.scatter([i for j in data_dict[k]], [j[0] for j in data_dict[k]],
                    c=colorlist([color_dict[j[1]] for j in data_dict[k]]), s=80, marker=markers[i])
    # plt.title('Sharp ratio distributions for different strategies')
    color_patch = []
    for key in color_dict.keys():
        color_patch.append(mpatches.Patch(color=colorlist(color_dict[key]), label=key))
    plt.legend(handles=color_patch, loc='lower right')
    plt.xticks([i for i in range(len(data_dict.keys()))], label, rotation=90)
    plt.ylabel('Sharp ratio')
    plt.show()


def compare_with_trad_tech(model_dict: dict):
    model_sets = list(model_dict.keys())
    bond_data = pd.read_csv('../US TREASURY.csv')
    us_bill = bond_data[['Time Period', '3M']]
    us_bill.dropna(inplace=True)
    us_bill.reset_index(inplace=True, drop=True)
    for i in range(len(us_bill)):
        if us_bill.loc[i, '3M'] == 'ND':
            us_bill.drop(index=i, inplace=True)
    us_bill.reset_index(inplace=True, drop=True)
    us_bill['Time Period'] = pd.to_datetime(us_bill['Time Period'])
    us_bill.set_index('Time Period', inplace=True)
    trad_tech_df = pd.read_csv('../extra_benchmark/trad_tech.csv', index_col=0)
    trad_tech_df.index = pd.to_datetime(trad_tech_df.index)
    df_list = []
    for s in model_sets:
        test_period = []
        with open(f'./{s}/sample_period.txt', 'r') as f:
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
                    t1 = pd.to_datetime(line.split('_')[0])
                    t2 = pd.to_datetime(line.split('_')[1])
                    test_period.append((t1, t2))
        rf_dict = {}
        counter = 0
        for t in test_period:
            rf_dict[counter] = us_bill.loc[t[0]:t[1], '3M'].astype(float).mean()
            counter += 1
        model_list = model_dict[s]
        data_p = []
        lv1 = []
        lv2 = []
        for p in range(5):
            rf_mean = rf_dict[p]
            rf_mean /= 100
            for model_name in model_list:
                print(model_name)
                m_cols = ['ret', 'SR']
                if p == 0:
                    for c in m_cols: lv1.append(c)
                    for _ in m_cols: lv2.append(model_name)
                std, r, n = show_ret(f'{model_name}', p, 0.9, 0.1, True, True, set_num=s.split(' ')[-1])

                low99 = stats.t.interval(0.99, n - 1, r, std)[0]
                low95 = stats.t.interval(0.95, n - 1, r, std)[0]
                low90 = stats.t.interval(0.90, n - 1, r, std)[0]
                sharpe = (r - rf_mean) / std
                # std = str(Decimal(std * 100).quantize(Decimal('0.01'))) + '%'
                if low99 > 0:
                    r = '***' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                elif low95 > 0:
                    r = '**' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                elif low90 > 0:
                    r = '*' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                else:
                    r = str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                if low99 - rf_mean > 0:
                    sharpe = '***' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                elif low95 - rf_mean > 0:
                    sharpe = '**' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                elif low90 - rf_mean > 0:
                    sharpe = '*' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                else:
                    sharpe = str(Decimal(sharpe).quantize(Decimal('0.01')))
                print(r, sharpe)
                data_m = [r, sharpe]
                data_p.append(np.array(data_m))
            test_start, test_end = test_period[p][0], test_period[p][1]
            trad_tech_df_p = trad_tech_df.loc[test_start:test_end, :].dropna()
            for tech in trad_tech_df_p.columns:
                stgy_name = tech.split('_')[0].upper()
                print(stgy_name)
                m_cols = ['ret', 'SR']
                if p == 0:
                    for c in m_cols: lv1.append(c)
                    for _ in m_cols: lv2.append(stgy_name)
                std = trad_tech_df_p[tech].std() * np.sqrt(250)
                cum_ret = (trad_tech_df_p[tech] + 1).cumprod()
                r = np.power(cum_ret.iloc[-1], 250 / len(cum_ret)) - 1
                n = len(trad_tech_df_p[tech])
                print(r, n, std)
                low99 = stats.t.interval(0.99, n - 1, r, std)[0]
                low95 = stats.t.interval(0.95, n - 1, r, std)[0]
                low90 = stats.t.interval(0.90, n - 1, r, std)[0]
                sharpe = (r - rf_mean) / std
                if low99 > 0:
                    r = '***' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                elif low95 > 0:
                    r = '**' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                elif low90 > 0:
                    r = '*' + str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                else:
                    r = str(Decimal(r * 100).quantize(Decimal('0.01'))) + '%'
                if low99 - rf_mean > 0:
                    sharpe = '***' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                elif low95 - rf_mean > 0:
                    sharpe = '**' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                elif low90 - rf_mean > 0:
                    sharpe = '*' + str(Decimal(sharpe).quantize(Decimal('0.01')))
                else:
                    sharpe = str(Decimal(sharpe).quantize(Decimal('0.01')))
                print(r, sharpe)
                data_m = [r, sharpe]
                data_p.append(np.array(data_m))
        data_p = np.reshape(data_p, (5, -1))
        print(data_p)

        df = pd.DataFrame(data_p,
                          index=[t[0].strftime('%Y/%m/%d') + '-' + t[1].strftime('%Y/%m/%d') for t in test_period],
                          columns=[lv2, lv1])
        print(df)
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df.fillna('-', inplace=True)
    print(df)

    with pd.ExcelWriter(f'../extra_benchmark/compare_with_trad_tech.xlsx') as f:
        df.to_excel(f)


def compare_with_sector_idx(model_dict: dict):
    model_sets = list(model_dict.keys())
    df_list = []
    regression_dict = {}
    for s in model_sets:
        for m_n in model_dict[s]:
            dfn_list = []
            for p in range(5):
                df_ret = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_ret.csv')
                df_rank = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_rank.csv')
                date = df_ret['date']
                df_ret.drop(columns='date', inplace=True)
                df_rank.drop(columns='date', inplace=True)
                df_ret[df_ret.abs() > 1000] = pd.NA
                df_rank = df_rank.apply(lambda x: (x - x.median()) / x.abs().sum(), axis=1)
                qt1 = df_rank.quantile(float(Decimal(str(0.9))), axis=1)
                qt2 = df_rank.quantile(float(Decimal(str(0.9)) + Decimal(str(0.1))), axis=1)
                mirror_qt1 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(0.9))), axis=1)
                mirror_qt2 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(0.9)) - Decimal(str(0.1))), axis=1)
                for i in range(len(qt1)):
                    df_rank.loc[i, df_rank.iloc[i, :] < mirror_qt2.iloc[i]] = 0
                    df_rank.loc[i, (df_rank.iloc[i, :] <= qt1.iloc[i]) & (mirror_qt1.iloc[i] <= df_rank.iloc[i, :])] = 0
                    df_rank.loc[i, df_rank.iloc[i, :] > qt2.iloc[i]] = 0
                scale = df_rank.abs().sum(axis=1)
                for i in range(len(df_rank.columns)):
                    df_rank.iloc[:, i] /= scale * 5
                ret = (df_rank * df_ret).sum(axis=1) * 0.01 - df_rank.diff(5).abs().sum(axis=1) * 0.0001
                ret.index = date
                ret.name = m_n
                dfn_list.append(ret)
            df_n = pd.concat(dfn_list)
            df_n = df_n.sort_index()
            df_list.append(df_n)
    df_model = pd.concat(df_list, axis=1)
    df_model.index = pd.to_datetime(df_model.index)
    df_sector = pd.read_csv('../extra_benchmark/10_Industry_Portfolios_Daily.csv', index_col=0)
    df_sector.index = pd.to_datetime(df_sector.index.astype(str))
    df_sector *= 0.01
    df = pd.concat([df_model, df_sector], axis=1)
    corr_mat = df.corr()
    corr_mat.drop(index=df_sector.columns, columns=df_model.columns, inplace=True)
    print(corr_mat)
    corr_mat.to_csv('../extra_benchmark/corr_mat_sector.csv', index_label='model')
    for m_n in df_model.columns:
        ols = sm.OLS(df[m_n], sm.add_constant(df[df_sector.columns]), missing='drop').fit()
        params = ols.params
        p_values = ols.pvalues
        for variable in p_values.index:
            if p_values[variable] <= 0.01:
                params[variable] = '***' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            elif p_values[variable] <= 0.05:
                params[variable] = '**' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            elif p_values[variable] <= 0.1:
                params[variable] = '*' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            else:
                params[variable] = str(Decimal(params[variable]).quantize(Decimal('0.0001')))
        params['R2'] = str(Decimal(ols.rsquared).quantize(Decimal('0.001')))
        regression_dict[m_n] = params
    reg_df = pd.DataFrame(regression_dict).transpose()
    print(reg_df)
    reg_df.to_csv('../extra_benchmark/reg_result_sector.csv', index_label='model')


def FF5_analysis(model_dict: dict):
    model_sets = list(model_dict.keys())
    df_list = []
    regression_dict = {}
    for s in model_sets:
        for m_n in model_dict[s]:
            dfn_list = []
            for p in range(5):
                df_ret = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_ret.csv')
                df_rank = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_rank.csv')
                date = df_ret['date']
                df_ret.drop(columns='date', inplace=True)
                df_rank.drop(columns='date', inplace=True)
                df_ret[df_ret.abs() > 1000] = pd.NA
                df_rank = df_rank.apply(lambda x: (x - x.median()) / x.abs().sum(), axis=1)
                qt1 = df_rank.quantile(float(Decimal(str(0.9))), axis=1)
                qt2 = df_rank.quantile(float(Decimal(str(0.9)) + Decimal(str(0.1))), axis=1)
                mirror_qt1 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(0.9))), axis=1)
                mirror_qt2 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(0.9)) - Decimal(str(0.1))), axis=1)
                for i in range(len(qt1)):
                    df_rank.loc[i, df_rank.iloc[i, :] < mirror_qt2.iloc[i]] = 0
                    df_rank.loc[i, (df_rank.iloc[i, :] <= qt1.iloc[i]) & (mirror_qt1.iloc[i] <= df_rank.iloc[i, :])] = 0
                    df_rank.loc[i, df_rank.iloc[i, :] > qt2.iloc[i]] = 0
                scale = df_rank.abs().sum(axis=1)
                for i in range(len(df_rank.columns)):
                    df_rank.iloc[:, i] /= scale * 5
                ret = (df_rank * df_ret).sum(axis=1) * 0.01 - df_rank.diff(5).abs().sum(axis=1) * 0.0001
                ret.index = date
                ret.name = m_n
                dfn_list.append(ret)
            df_n = pd.concat(dfn_list)
            df_n = df_n.sort_index()
            df_list.append(df_n)
    df_model = pd.concat(df_list, axis=1)
    df_model.index = pd.to_datetime(df_model.index)
    df_ff5 = pd.read_csv('../extra_benchmark/F-F_Research_Data_5_Factors_2x3_daily.csv', index_col=0)
    df_ff5.index = pd.to_datetime(df_ff5.index.astype(str))
    df_ff5 *= 0.01
    df = pd.concat([df_model, df_ff5], axis=1)
    for m_n in df_model.columns:
        ols = sm.OLS(df[m_n] - df_ff5['RF'], df[df_ff5.columns[:-1]], hasconst=True, missing='drop').fit()
        print(ols.summary())
        params = ols.params
        p_values = ols.pvalues
        for variable in p_values.index:
            if p_values[variable] <= 0.01:
                params[variable] = '***' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            elif p_values[variable] <= 0.05:
                params[variable] = '**' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            elif p_values[variable] <= 0.1:
                params[variable] = '*' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            else:
                params[variable] = str(Decimal(params[variable]).quantize(Decimal('0.0001')))
        params['R2'] = str(Decimal(ols.rsquared).quantize(Decimal('0.001')))
        regression_dict[m_n] = params
    reg_df = pd.DataFrame(regression_dict).transpose()
    print(reg_df)
    reg_df.to_csv('../extra_benchmark/reg_result_ff5.csv', index_label='model')

    for m_n in df_model.columns:
        ols = sm.OLS(df[m_n] - df_ff5['RF'], sm.add_constant(df[df_ff5.columns[:-1]]), missing='drop').fit()
        print(ols.summary())
        params = ols.params
        p_values = ols.pvalues
        for variable in p_values.index:
            if p_values[variable] <= 0.01:
                params[variable] = '***' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            elif p_values[variable] <= 0.05:
                params[variable] = '**' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            elif p_values[variable] <= 0.1:
                params[variable] = '*' + str(Decimal(params[variable]).quantize(Decimal('0.0001')))
            else:
                params[variable] = str(Decimal(params[variable]).quantize(Decimal('0.0001')))
        params['R2'] = str(Decimal(ols.rsquared).quantize(Decimal('0.001')))
        regression_dict[m_n] = params
    reg_df = pd.DataFrame(regression_dict).transpose()
    print(reg_df)
    reg_df.to_csv('../extra_benchmark/reg_result_ff5_with_const.csv', index_label='model')


def volatility_analysis(model_dict):
    vix1 = pd.read_csv('../extra_benchmark/vix93-03.csv')[['Date', 'VIX Close']]
    vix1 = vix1.rename(columns={'Date': 'date', 'VIX Close': 'close'})

    vix2 = pd.read_csv('../extra_benchmark/vix04-24.csv')[['DATE', 'CLOSE']]
    vix2 = vix2.rename(columns={'DATE': 'date', 'CLOSE': 'close'})
    print(vix1)
    vix1['date'] = pd.to_datetime(vix1['date'], format='%m/%d/%y')

    vix2['date'] = pd.to_datetime(vix2['date'])

    vix = pd.concat([vix1, vix2], axis=0)
    vix.set_index('date', drop=True, inplace=True)
    vix.sort_index(inplace=True)

    vix_year = vix.resample('Y').mean()

    volatility_dict = {'extreme': [], 'high': [], 'moderate': [], 'low': []}

    for index in vix_year.index:
        if vix_year.loc[index, 'close'] > 25:
            volatility_dict['extreme'].append(str(index.year))

        elif 20 < vix_year.loc[index, "close"] <= 25:

            volatility_dict['high'].append(str(index.year))

        elif 15 < vix_year.loc[index, "close"] <= 20:

            volatility_dict['moderate'].append(str(index.year))

        elif vix_year.loc[index, "close"] <= 15:

            volatility_dict['low'].append(str(index.year))

    model_sets = list(model_dict.keys())
    df_dict = {}

    for s in model_sets:
        for m_n in model_dict[s]:
            dfret_list = []
            dfpc_list = []
            for p in range(5):
                df_ret = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_ret.csv')
                df_rank = pd.read_csv(f'./{s}/result/dataframe/{m_n}_sample_period_{p}_rank.csv')
                date = df_ret['date']
                df_ret.drop(columns='date', inplace=True)
                df_rank.drop(columns='date', inplace=True)
                df_ret[df_ret.abs() > 1000] = pd.NA
                df_rank = df_rank.apply(lambda x: (x - x.median()) / x.abs().sum(), axis=1)
                qt1 = df_rank.quantile(float(Decimal(str(0.9))), axis=1)
                qt2 = df_rank.quantile(float(Decimal(str(0.9)) + Decimal(str(0.1))), axis=1)
                mirror_qt1 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(0.9))), axis=1)
                mirror_qt2 = df_rank.quantile(float(Decimal(str(1)) - Decimal(str(0.9)) - Decimal(str(0.1))), axis=1)
                for i in range(len(qt1)):
                    df_rank.loc[i, df_rank.iloc[i, :] < mirror_qt2.iloc[i]] = 0
                    df_rank.loc[i, (df_rank.iloc[i, :] <= qt1.iloc[i]) & (mirror_qt1.iloc[i] <= df_rank.iloc[i, :])] = 0
                    df_rank.loc[i, df_rank.iloc[i, :] > qt2.iloc[i]] = 0
                scale = df_rank.abs().sum(axis=1)
                for i in range(len(df_rank.columns)):
                    df_rank.iloc[:, i] /= scale * 5
                ret = (df_rank * df_ret).sum(axis=1) * 0.01
                position_change = df_rank.diff(5).abs().sum(axis=1)
                ret.index = date
                ret.name = m_n
                ret.index = pd.to_datetime(ret.index)
                position_change.index = date
                position_change.name = m_n
                position_change.index = pd.to_datetime(position_change.index)
                dfret_list.append(ret)
                dfpc_list.append(position_change)
            dfret = pd.concat(dfret_list)
            dfret = dfret.sort_index()
            dfpc = pd.concat(dfpc_list)
            dfpc = dfpc.sort_index()
            df_dict[m_n] = {'ret': dfret, 'position_change': dfpc}

    def cal_ret(commission_rate, r, pos_chg):
        r_c = r - pos_chg * commission_rate
        cum_r_c = (r_c + 1).cumprod()
        total_ret = (cum_r_c.iloc[-1]) - 1
        return total_ret

    from scipy.optimize import root

    breakeven_data = {m: [] for m in df_dict.keys()}

    for m in df_dict.keys():
        ret = df_dict[m]['ret']
        position_change = df_dict[m]['position_change']

        for state in volatility_dict.keys():
            ret_list = []
            position_change_list = []
            for year in volatility_dict[state]:
                try:
                    ret_list.append(ret.loc[year])
                    position_change_list.append(position_change.loc[year])
                except KeyError as e:
                    continue
            ret_state = pd.concat(ret_list,axis=0)
            position_change_state = pd.concat(position_change_list,axis=0)
            break_even_fee = root(cal_ret, np.array([0.00005]), (ret_state, position_change_state))['x'][0]
            breakeven_data[m].append(str(Decimal(break_even_fee).quantize(Decimal('0.00001'))))

    breakeven_df = pd.DataFrame(breakeven_data, index=list(volatility_dict.keys()))
    print(breakeven_df)
    breakeven_df.to_csv('../extra_benchmark/fee_sensitivity.csv')

def volatility_fig():
    import matplotlib.pyplot as plt
    vix1 = pd.read_csv('../extra_benchmark/vix93-03.csv')[['Date', 'VIX Close']]
    vix1 = vix1.rename(columns={'Date': 'date', 'VIX Close': 'close'})

    vix2 = pd.read_csv('../extra_benchmark/vix04-24.csv')[['DATE', 'CLOSE']]
    vix2 = vix2.rename(columns={'DATE': 'date', 'CLOSE': 'close'})
    print(vix1)
    vix1['date'] = pd.to_datetime(vix1['date'], format='%m/%d/%y')

    vix2['date'] = pd.to_datetime(vix2['date'])

    vix = pd.concat([vix1, vix2], axis=0)
    vix.set_index('date', drop=True, inplace=True)
    vix.sort_index(inplace=True)

    vix_year = vix.resample('Y').mean()

    print(vix_year)

    vix_year.index = vix_year.index.year

    fig = plt.figure(layout='constrained')

    plt.fill_between(x=vix_year.index[:-2], y1=[25 for _ in vix_year.index[:-2]], y2=[vix_year.max().item()+0.5 for _ in vix_year.index[:-2]],color='#FFC1C1')

    plt.fill_between(x=vix_year.index[:-2], y1=[20 for _ in vix_year.index[:-2]],
                     y2=[25 for _ in vix_year.index[:-2]], color='#FFDEAD')

    plt.fill_between(x=vix_year.index[:-2], y1=[15 for _ in vix_year.index[:-2]],
                     y2=[20 for _ in vix_year.index[:-2]], color='#E0EEE0')

    plt.fill_between(x=vix_year.index[:-2], y1=[vix_year.min().item()-0.5 for _ in vix_year.index[:-2]],
                     y2=[15 for _ in vix_year.index[:-2]], color='#E0EEEE')

    plt.plot(vix_year.iloc[:-2],label='mean VIX')

    plt.text(vix_year.index[3], (25+vix_year.max().item()+0.5)/2, r'EXTREME', fontsize=12,ha="center", va="center")

    plt.text(vix_year.index[3], 22.5, r'HIGH', fontsize=12, ha="center", va="center")

    plt.text(vix_year.index[10], 17.5, r'MODERATE', fontsize=12, ha="center", va="center")

    plt.text(vix_year.index[19], (15 + vix_year.min().item()-0.5) / 2, r'LOW', fontsize=12, ha="center", va="center")

    plt.legend()

    plt.grid()

    plt.margins(x=0,y=0)

    plt.show()





if __name__ == '__main__':
    # show_general_ret()
    # show_details({'model set 12': ['swin20BOLL+MACD',
    #                                'swin5BOLL+RSI',
    #                                'swin5MA+Vol',
    #                                'mom5',
    #                                'mom20',
    #                                'mom60',
    #                                'revs5',
    #                                'revs20',
    #                                'revs60']})
    # show_scatter('model set 12.xlsx')
    # compare_with_trad_tech({'model set 12': ['swin20BOLL+MACD', 'swin5BOLL+RSI', 'swin5MA+Vol']})
    # compare_with_sector_idx(
    #     {'model set 12': ['swin20BOLL+MACD', 'swin5BOLL+RSI', 'swin5MA+Vol']})
    # FF5_analysis({'model set 12': ['swin20BOLL+MACD', 'swin5BOLL+RSI', 'swin5MA+Vol']})
    volatility_fig()
    # volatility_analysis({'model set 12': ['swin20BOLL+MACD', 'swin5BOLL+RSI', 'swin5MA+Vol']})

    # df = pd.DataFrame({'c1':[i for i in range(200)],'c2':[i*2 for i in range(200)]},index=[str(200000 + i%12+1 + i//12 * 100) for i in range(200)])
    # df.index = pd.to_datetime(df.index,format='%Y%m')
    # print(df)
    # print(df.loc['2025',:])
