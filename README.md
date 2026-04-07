# VisualizingTechnicalAnalysis
Created: 2026-03-31

> NOTE: this repository only contains code, for full-project files please refer to [Yi, R., & Wang, J. (2026). VisualizingTechnicalAnalysis. Zenodo.](https://doi.org/10.5281/zenodo.19416222)
## Install

Python version: `python >= 3.11`

Package Dependencies:
- `matplotlib==3.8.0`
- `numpy==1.24.3`
- `pandas~=2.1.4`
- `Pillow~=9.4.0`
- `scipy==1.11.1`
- `timm==0.9.7`
- `torch==2.1.1+cu121`
- `torchvision==0.16.1+cu121`
- `tqdm==4.65.0`
- `tulipy==0.4.0`
- `statsmodels~=0.14.0`
- `opencv-python~=4.9.0.80`
- `grad-cam==1.5.0`
```bash
pip install -r requirements.txt
```

## Content and Structure

- `main.py`: script to train and test SwinTransformer Based models, output trained models and predictions to `./result`.
- `./result/*`: backtest results of `main.py`
- `./cache`: cached files during running of `main.py`
- `./checkpoint`: model checkpoint files during training.
- `./data_us/*.csv`: adjusted kline data files by tickers.
- `./archive/model set *`: archive of sets of model parameters (`./archive/model set */checkpoint`) and predictions (`./archive/model set */result/dataframe`).
- `./archive/performance_display.py`: script generates tables and figures present in the essay, including the following:
  - Figure 6: Scatters of sharp ratios for different strategies.
  - Table 2: Overall performance of long-short portfolios of models on test periods. `./statistics/overall.csv`
  - Table 3: Break-even transaction costs of models on different market volatility periods. `./extra_benchmark/fee_sensitivity.csv`
  - Figure 7: Yearly mean VIX index from 1990 to 2022.
  - Table 4: Portfolios performance comparison in Model Set 12. `./archive/model set 12/model set 12.xlsx`
  - Table 5: Portfolios performance comparison with traditional technical strategies. `./extra_benchmark/compare_with_trad_tech.xlsx`
  - Table 6: Regression results of three machine-learning portfolios against 10 industry portfolios. `./extra_benchmark/reg_result_sector.csv`
  - Table 7: Regression results of three machine-learning portfolios against FF 5-factor model. `./extra_benchmark/reg_result_ff5_with_const.csv`
- `./extra_benchmark`: orignal data, and result files for extra benchmark analysis. The orignal data files including:
  - `10_Industry_Portfolios_Daily.csv`: Sector Indices data 
  - `F-F_Research_Data_5_Factors_2x3_daily.csv`: FF5 factors data
  - `vix04-24.csv` and `vix93-03.csv`: VIX index data
- `visualize_model.py`: script visualizes the attention map of model, output images to `./temp`
- `trad_tech.py`: script generates traditional technical indicators strategy signals to `./extra_benchmark/trad_tech.csv`

## Hardware Requirements
1. Up to 5TB of disk space to store intermediary artifacts.
2. GPU of at least 12GB of VRAM (non-parallel)

> NOTE: The `main.py` may take 120 hours running under non-parallel mode. (i7-13700K+RTX4070TI), 
> 
> However, this processes can be easily paralleled by using process pool to substitute the loop in `line 42` ~ `line 56` (the VRAM and RAM it consumes will be multiplied)

## Contact
For technical support, please send email to `yrdyrd123@sina.com`