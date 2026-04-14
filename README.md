# Visualizing Technical Analysis

**Created:** 2026-03-31

**The Author of this Reproducibility Package:** Yi, Ruidong (yrdyrd123@sina.com)

**Note:** This repository contains the source code for the paper. The full reproducibility package (including large data files and trained models) is available at [Zenodo](https://doi.org/10.5281/zenodo.19416222).

---

## Overview

This replication package contains the code to reproduce the results presented in the paper "Visualizing Technical Analysis". The project implements Swin Transformer-based deep learning models to identify technical chart patterns in stock market data and evaluates their predictive performance through backtesting.

### What this code produces

The code in this repository generates:
- **18 trained models** with different technical indicator combinations and window sizes
- **Backtest results** including predictions and performance metrics
- **6 Figures** and **6 Tables** presented in the paper (generated via `archive/performance_display.py`)

---

## Repository Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python package dependencies
├── main.py                             # Main training and backtesting script
├── Data_generate.py                    # Data preprocessing and technical indicator calculation
├── BTframe.py                          # Backtesting framework
├── DrawLib.py                          # Visualization utilities
├── visualize_model.py                  # Model attention visualization (Grad-CAM)
├── trad_tech.py                        # Traditional technical strategy benchmark
├── benchmark.py                        # Benchmark analysis tools
├── dataloader_test.py                  # Data loader testing utilities
├── model.py                            # Neural network model definitions
├── US TREASURY.csv                     # US Treasury yield data (sample)
│
├── data_us/                            # Raw stock price data (user-provided)
│   └── [TICKER].csv                    # Individual stock OHLCV data files
│
├── data_us/tech/                       # Processed data with technical indicators
│   
├── cache/                              # Cached intermediate files
├── checkpoint/                         # Model checkpoint files during training
├── result/                             # Backtest prediction outputs
├── log/                                # Training and validation logs
│
├── archive/                            # Analysis and visualization scripts
│   ├── performance_display.py          # Generates all figures and tables for the paper
│   └── model set [N]/                  # Archived model sets (parameters + predictions)
│       ├── checkpoint/                 # Saved model weights
│       └── result/dataframe/           # Prediction results
│
├── extra_benchmark/                    # Additional benchmark analysis data
│   ├── 10_Industry_Portfolios_Daily.csv    # Fama-French industry portfolios
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv  # Fama-French 5 factors
│   ├── vix04-24.csv & vix93-03.csv     # VIX index data
│   └── [generated output files]        # Regression results and comparison tables
│
├── statistics/                         # Statistical analysis outputs
├── temp/                               # Temporary files for visualization
└── fig/                                # ingnore
```

---

## Computing Environment

### System Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| **Operating System** | Windows 10/11, Linux, or macOS |
| **CPU** | Intel i7 or equivalent (multiprocessing recommended) |
| **RAM** | 32 GB recommended |
| **GPU** | NVIDIA GPU with ≥ 12 GB VRAM (required for training) |
| **Storage** | ≥ 5 TB free space (for intermediate image artifacts) |

### Software Dependencies

- **Python:** >= 3.11
- **Deep Learning Framework:** PyTorch 2.1.1 with CUDA 12.1 support
- **Key Packages:**
  - `torch==2.1.1+cu121`
  - `torchvision==0.16.1+cu121`
  - `timm==0.9.7` (Swin Transformer implementation)
  - `matplotlib==3.8.0`
  - `numpy==1.24.3`
  - `pandas~=2.1.4`
  - `scipy==1.11.1`
  - `statsmodels~=0.14.0`
  - `opencv-python~=4.9.0.80`
  - `grad-cam==1.5.0`
  - `tulipy==0.4.0` (Technical indicators)
  - `tqdm==4.65.0`
  - `Pillow~=9.4.0`

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd VisualizingTechnicalAnalysis

# Install dependencies
pip install -r requirements.txt

# For CUDA 12.1 support, you may need to install PyTorch separately:
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## Data Availability and Provenance

### Raw Data

**US Stock Price Data** (`./data_us/*.csv`)
- **Description:** Daily OHLCV (Open, High, Low, Close, Volume) price data for individual US stocks
- **Format:** CSV files, one per ticker (e.g., `AAPL.csv`, `MSFT.csv`)
- **Columns:** Typically include Date, Open, High, Low, Close, Volume, Adjusted Close
- **Source:** User-provided (e.g., from CRSP, Yahoo Finance, or other financial data providers)
- **Coverage:** The paper uses data from 1990 to 2023 for training and testing
- **Access:** Due to licensing restrictions, raw price data is not included in this repository. Users must obtain their own data.

### Intermediate Data

**Technical Indicator Data** (`./data_us/tech/`)
- **Generated by:** `main.py` (calls `Data_generate.addTechnicalIndicators()`)
- **Description:** Raw price data enhanced with technical indicators (MA, BOLL, MACD, RSI, Volume)
- **Format:** CSV files with calculated indicator values
- **Note:** This is computationally expensive to regenerate. Pre-processed data can be obtained from the Zenodo archive.

**Image Samples** (`<USER_DEFINED_STORAGE_PATH>`)
- **Generated by:** `main.py` during data preparation
- **Description:** Candlestick chart images generated from price data for model input
- **Size:** ~5 TB total (main storage requirement)

### Output Data

**Model Predictions** (`./result/` and `./archive/model set */result/`)
- **Generated by:** `main.py` during backtesting phase
- **Description:** Model predictions (rankings) for test periods
- **Format:** CSV files with date and predicted rankings per stock

---

## Code and Output Mapping

The following table maps each figure and table in the paper to the script that produces it:

| Output | Script | Description | Notes |
|--------|--------|-------------|-------|
| **Figure 6** | `archive/performance_display.py` | Scatters of Sharpe ratios for different strategies | Requires model predictions from `./archive/model set */result/dataframe/` |
| **Figure 7** | `archive/performance_display.py` | Yearly mean VIX index from 1990 to 2022 | Uses VIX data from `./extra_benchmark/` |
| **Table 2** | `archive/performance_display.py` | Overall performance of long-short portfolios | Output: `./statistics/overall.csv` |
| **Table 3** | `archive/performance_display.py` | Break-even transaction costs | Output: `./extra_benchmark/fee_sensitivity.csv` |
| **Table 4** | `archive/performance_display.py` | Portfolios comparison in Model Set 12 | Output: `./archive/model set 12/model set 12.xlsx` |
| **Table 5** | `archive/performance_display.py` | Comparison with traditional technical strategies | Output: `./extra_benchmark/compare_with_trad_tech.xlsx` |
| **Table 6** | `archive/performance_display.py` | Regression against 10 industry portfolios | Output: `./extra_benchmark/reg_result_sector.csv` |
| **Table 7** | `archive/performance_display.py` | Regression against FF 5-factor model | Output: `./extra_benchmark/reg_result_ff5_with_const.csv` |

### Model Training

The main training pipeline is executed via `main.py`:

1. **Data Preprocessing:** Calculates technical indicators for all stocks
2. **Data Splitting:** Randomly divides data into 10 periods for training (4 periods), validating (1 periods) and testing (5 periods).
3. **Model Training:** Trains 18 Swin Transformer models (6 technical indicator combinations × 3 window sizes)
4. **Model Selection:** Selects models with validation accuracy > 51%
5. **Backtesting:** Runs backtests on selected models

---

## Instructions

### Quick Start

1. **Run Model Pipeline:**
   ```bash
   python main.py
   ```
2. **Archive Results:**
   ```bash
   cp result ./archive/model set [n]/result
   cp checkpoint ./archive/model set [n]/checkpoint
   ``` 
3. **Generate Figures and Tables:**
   ```bash
   cd ./archive
   python performance_display.py
   ```

4. **Visualize Model Attention (Optional):**
   ```bash
   python visualize_model.py
   ```

### Configuration Notes

- **Image Output Path:** In `main.py` line 99, update the path to your local drive with sufficient space:
  ```python
  out_img_path='N:/Dataset/img'  # Change to your path
  ```

- **Multiprocessing:** The code uses 10 processes by default for technical indicator calculation. Adjust in `main.py` line 27 if needed.

- **GPU Memory:** If you encounter OOM errors, reduce batch size in `Model_workflow.train()` call (line 54 of `main.py`)

---

## Expected Runtime

| Phase | Hardware | Approximate Time        |
|-------|----------|-------------------------|
| Technical indicator calculation | i7-13700K  | 5 minutes               |
| Data splitting | i7-13700K | 10 minutes              |
| Model training (18 models) | RTX 4070 Ti (12GB) | ~10 hours total         |
| Backtesting | RTX 4070 Ti | ~100 hours              |
| **Total Pipeline** | **i7-13700K + RTX 4070 Ti** | **~120 hours (5 days)** |

**Note:** Runtime can be significantly reduced with:
- More powerful GPUs (RTX 4090, A100, etc.)
- Parallel training across multiple GPUs
- Using pre-computed intermediate datasets from Zenodo

---

## Special Setup Requirements

### GPU Requirements
- **Mandatory:** NVIDIA GPU with CUDA support
- **VRAM:** Minimum 12 GB for non-parallel training
- **CUDA Version:** 12.1 recommended (matching PyTorch cu121 build)

### Storage Requirements
- **Intermediate images:** ~5 TB during training
- **Model checkpoints:** ~2 GB per model (36 GB total for 18 models)
- **Results and logs:** ~1 GB

### External Data for Benchmarks
The following files are required for full replication of benchmark analyses:
- **Fama-French 5 Factors:** `F-F_Research_Data_5_Factors_2x3_daily.csv` from [Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- **Industry Portfolios:** `10_Industry_Portfolios_Daily.csv` from the same source
- **VIX Index:** Historical VIX data from [CBOE](https://www.cboe.com/tradable_products/vix/)

---

## License

This code is released under the MIT License (see `LICENSE.txt`).

Note that the trained models and datasets may have separate licensing restrictions. Please refer to the Zenodo archive for specific terms.

---

## Contact

For technical support or questions about the code, please contact:

**Email:** yrdyrd123@sina.com
---