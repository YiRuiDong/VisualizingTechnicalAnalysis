import copy

from PIL import Image
from PIL import ImageDraw
from os import path as op
from os import makedirs
import pandas as pd


def DrawOHLCTechnical(height, OHLC, image_out_path=None, filename=None, show=False,log_index='NO_LOG_INDEX', **kwargs):
    """
    This function generate an image of OHLC information, and can include technical indicators if provided
    :param height: int; Height of the image.
    :param OHLC: List or Dataframe; The data should be arranged in open, high, low and close respectively.
    :param image_out_path: str; Where the image be saved to.
    :param filename: str; filename of the image.
    :param show: bool; whether to show the image.
    :param technical_kwargs: dict; left blank if no technical indicator is needed. Keys in the dict is the name of corresponding technical indicator, with
    value of Dataframe type. For example, MACD = pd.Dataframe({'MACD':...})
    """
    assert type(OHLC) in [list, pd.DataFrame]
    if type(OHLC) == list:
        width = len(OHLC[0]) * 3
    else:
        width = len(OHLC) * 3
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    MA_color = ['yellow', 'blue', 'orange']
    Short_long_color = ['gold', 'white']
    Histogram_color = ['green', 'red']
    Boll_bands_color = ["purple", "cyan", "yellow"]
    RSI_color = ['white', 'blue', 'gold']
    indicators_pos_flag = {"MA": 1, "MACD": 2, "Vol": 2, "RSI": 2, 'BOLL': 1}  # value 1 for indicators drawn on OHLC chart, value 2 for indicators drawn below OHLC chart.
    upper_charts_count = 1
    lower_chart_count = 0
    technical_kwargs = {}
    for key in kwargs.keys():
        if (key not in ['image_out_path', 'height', 'OHLC', 'filename', 'show','log_index']) and kwargs[key] is not None:
            technical_kwargs[key] = kwargs[key]
    if technical_kwargs:
        for indicator in technical_kwargs.keys():
            if indicator not in list(indicators_pos_flag.keys()):
                raise Exception('Illegal indicator!')
            if type(technical_kwargs[indicator]) is not pd.DataFrame:
                raise Exception('pandas.DataFrame type data expected!')
            if width / 3 != len(technical_kwargs[indicator]):
                with open('log/exception_log.txt','a') as file:
                    print(f'Exception::\"technical indicator length doesn\'t match OHLC length!\" encountered while processing {log_index}',file=file)
                raise Exception('technical indicator length doesn\'t match OHLC length!')
            if indicators_pos_flag[indicator] == 1:
                upper_charts_count += 1
            else:
                lower_chart_count += 1
            assert upper_charts_count <= 2
            assert lower_chart_count <= 1
    upper_chart_sup = []
    upper_chart_inf = []
    if type(OHLC) == list:
        upper_chart_sup.append(max(OHLC[1]))
        upper_chart_inf.append(min(OHLC(2)))
    else:
        upper_chart_sup.append(OHLC.iloc[:, 1].max())
        upper_chart_inf.append(OHLC.iloc[:, 2].min())
    if technical_kwargs:
        if lower_chart_count > 0:
            lower_chart_sup = []
            lower_chart_inf = []
            for indicator in technical_kwargs.keys():
                if indicators_pos_flag[indicator] == 1:
                    for columns_name in technical_kwargs[indicator].columns:
                        upper_chart_sup.append(technical_kwargs[indicator].loc[:, columns_name].max())
                        upper_chart_inf.append(technical_kwargs[indicator].loc[:, columns_name].min())
                else:
                    for columns_name in technical_kwargs[indicator].columns:
                        lower_chart_sup.append(technical_kwargs[indicator].loc[:, columns_name].max())
                        lower_chart_inf.append(technical_kwargs[indicator].loc[:, columns_name].min())
            lower_chart_sup = max(lower_chart_sup)
            lower_chart_inf = min(lower_chart_inf)
        else:
            for indicator in technical_kwargs.keys():
                for columns_name in technical_kwargs[indicator].columns:
                    upper_chart_sup.append(technical_kwargs[indicator].loc[:, columns_name].max())
                    upper_chart_inf.append(technical_kwargs[indicator].loc[:, columns_name].min())
    upper_chart_sup = max(upper_chart_sup)
    upper_chart_inf = min(upper_chart_inf)
    if lower_chart_count == 0:  # when there is no lower chart
        if type(OHLC) == list:
            O, H, L, C = OHLC[0], OHLC[1], OHLC[2], OHLC[3]
            # normalize list-type OHLC data, and rescale them to fit the image size.
            for i in range(len(O)):
                for j in range(len(OHLC[i])):
                    if type(upper_chart_sup) is not float or type(upper_chart_inf) is not float:
                        return img, False
                    OHLC[i][j] -= upper_chart_inf
                    OHLC[i][j] /= upper_chart_sup - upper_chart_inf
                    OHLC[i][j] *= height
        else:
            # normalize DataFrame-type OHLC data, and rescale them to fit the image size.
            if type(upper_chart_sup) is not float or type(upper_chart_inf) is not float:
                return img, False
            OHLC -= upper_chart_inf
            OHLC /= upper_chart_sup - upper_chart_inf
            OHLC *= height
        if technical_kwargs:
            # to normalize technical indicator data which are included in the upper chart, and rescale them to fit the chart.
            for indicator in technical_kwargs.keys():
                if type(upper_chart_sup) is not float or type(upper_chart_inf) is not float:
                    return img, False
                technical_kwargs[indicator] -= upper_chart_inf
                technical_kwargs[indicator] /= upper_chart_sup - upper_chart_inf
                technical_kwargs[indicator] *= height
    else:  # when there is a lower chart.
        if type(OHLC) == list:
            O, H, L, C = OHLC[0], OHLC[1], OHLC[2], OHLC[3]
            # the upper chart data needs to be move up after normalization and rescaling when there is a lower chart.
            for i in range(len(O)):
                for j in range(len(OHLC[i])):
                    OHLC[i][j] -= upper_chart_inf
                    if type(upper_chart_sup) is not float or type(upper_chart_inf) is not float:
                        return img, False
                    else:
                        OHLC[i][j] /= upper_chart_sup - upper_chart_inf
                        OHLC[i][j] *= 3 / 4 * height - 2
                        OHLC[i][j] += 1 / 4 * height + 2
        else:
            OHLC -= upper_chart_inf
            OHLC /= upper_chart_sup - upper_chart_inf
            OHLC *= 3 / 4 * height - 2
            OHLC += 1 / 4 * height + 2
        for indicator in technical_kwargs.keys():
            if indicators_pos_flag[indicator] == 1:
                technical_kwargs[indicator] -= upper_chart_inf
                if type(upper_chart_sup) is not float or type(upper_chart_inf) is not float:
                    return img, False
                else:
                    technical_kwargs[indicator] /= upper_chart_sup - upper_chart_inf
                    technical_kwargs[indicator] *= 3 / 4 * height - 2
                    technical_kwargs[indicator] += 1 / 4 * height + 2
            else:
                if type(upper_chart_sup) is not float or type(upper_chart_inf) is not float:
                    return img, False
                else:
                    technical_kwargs[indicator] -= lower_chart_inf
                    technical_kwargs[indicator] /= lower_chart_sup - lower_chart_inf
                    technical_kwargs[indicator] *= 1 / 4 * height - 2
    if type(OHLC) == list:
        O, H, L, C = OHLC[0], OHLC[1], OHLC[2], OHLC[3]
        bar_center = 1
        for i in range(len(O)):
            draw.line((bar_center, H[i], bar_center, max(O[i], C[i])), fill=Histogram_color[1 if O[i] <= C[i] else 0])
            draw.line((bar_center, L[i], bar_center, min(O[i], C[i])), fill=Histogram_color[1 if O[i] <= C[i] else 0])
            if O[i] > C[i]:
                draw.rectangle([bar_center - 1, C[i], bar_center + 1, O[i]], fill=Histogram_color[0])
            elif O[i] <= C[i]:
                draw.rectangle([bar_center - 1, O[i], bar_center + 1, C[i]], fill=Histogram_color[1])
            bar_center += 3
    else:
        O, H, L, C = OHLC.iloc[:, 0], OHLC.iloc[:, 1], OHLC.iloc[:, 2], OHLC.iloc[:, 3]
        bar_center = 1
        for i in range(len(O)):
            draw.line((bar_center, H.iloc[i], bar_center, max(O.iloc[i], C.iloc[i])), fill=Histogram_color[1 if C[i] >= O[i] else 0])
            draw.line((bar_center, L.iloc[i], bar_center, min(O.iloc[i], C.iloc[i])), fill=Histogram_color[1 if C[i] >= O[i] else 0])
            if O[i] > C[i]:
                draw.rectangle((bar_center - 1, C.iloc[i], bar_center + 1, O.iloc[i]), fill=Histogram_color[0])
            elif C[i] >= O[i]:
                draw.rectangle((bar_center - 1, O.iloc[i], bar_center + 1, C.iloc[i]), fill=Histogram_color[1])
            bar_center += 3
    if technical_kwargs:
        lower_chart_center = ((height / 4) - 2) / 2
        for indicator in technical_kwargs.keys():
            if indicator == 'MA':
                df = technical_kwargs[indicator]
                # make sure the number of columns does not exceed three.
                MA_counter = 0
                for MA in df.columns:
                    assert MA_counter <= 3
                    bar_center = 1
                    for i in range(len(df.loc[:, MA]) - 1):
                        draw.line((bar_center, df.loc[i, MA], bar_center + 3, df.loc[i + 1, MA]), fill=MA_color[MA_counter])
                        bar_center += 3
                    MA_counter += 1
            elif indicator == 'MACD':
                df = technical_kwargs[indicator]
                # make sure that the columns take the order of macd, macd signal and macd histogram
                bar_center = 1
                for i in range(len(df.iloc[:, 0]) - 1):
                    draw.line((bar_center, df.iloc[i, 0], bar_center + 3, df.iloc[i + 1, 0]), fill=Short_long_color[0])
                    draw.line((bar_center, df.iloc[i, 1], bar_center + 3, df.iloc[i + 1, 1]), fill=Short_long_color[1])
                    draw.line((bar_center, lower_chart_center + df.iloc[i, 0] - df.iloc[i, 1], bar_center, lower_chart_center),
                              fill=Histogram_color[1 if df.iloc[i, 0] - df.iloc[i, 1] + lower_chart_center >= lower_chart_center else 0])
                    if i == len(df.iloc[:, 0]) - 2:
                        i += 1
                        draw.line((bar_center + 3, lower_chart_center + df.iloc[i, 0] - df.iloc[i, 1], bar_center + 3, lower_chart_center),
                                  fill=Histogram_color[1 if df.iloc[i, 0] - df.iloc[i, 1] + lower_chart_center >= lower_chart_center else 0])
                    bar_center += 3
            elif indicator == 'Vol':
                df = technical_kwargs[indicator]
                # make sure there is only one column
                bar_center = 1
                for i in range(len(df.iloc[:, 0])):
                    draw.line((bar_center, df.iloc[i, 0], bar_center, 0), fill=Histogram_color[1 if C[i] >= O[i] else 0])
                    bar_center += 3
            elif indicator == 'RSI':
                df = technical_kwargs[indicator]
                # make sure the number of columns does not exceed three.
                RSI_counter = 0
                for RSI in df.columns:
                    assert RSI_counter <= 3
                    bar_center = 1
                    for i in range(len(df.loc[:, RSI]) - 1):
                        draw.line((bar_center, df.loc[i, RSI], bar_center + 3, df.loc[i + 1, RSI]), fill=RSI_color[RSI_counter])
                        bar_center += 3
                    RSI_counter += 1
            elif indicator == 'BOLL':
                df = technical_kwargs[indicator]
                # make sure that the columns take the order of bands lower, bands middle and bands upper.
                bar_center = 1
                for i in range(len(df.iloc[:, 0]) - 1):
                    draw.line((bar_center, df.iloc[i, 0], bar_center + 3, df.iloc[i + 1, 0]), fill=Boll_bands_color[0])
                    draw.line((bar_center, df.iloc[i, 1], bar_center + 3, df.iloc[i + 1, 1]), fill=Boll_bands_color[1])
                    draw.line((bar_center, df.iloc[i, 2], bar_center + 3, df.iloc[i + 1, 2]), fill=Boll_bands_color[2])
                    bar_center += 3
    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if show:
        img.show()
    if image_out_path and filename:
        if not op.exists(image_out_path):
            makedirs(image_out_path)
        img.save(op.join(image_out_path, filename))
    return img, True


if __name__ == '__main__':
    pass
