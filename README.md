# 🔥 Adaptive and Aggregated Intelligence (AAI) for Subseasonal-to-Seasonal (S2S) Climate Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)]()

This repository implements an **Adaptive and Aggregated Intelligence (AAI)** framework designed to improve **Subseasonal-to-Seasonal (S2S)** climate forecast skill, especially for **extreme weather events** like heatwaves.

> The **main script can run with random synthetic values**, so you can test it immediately.  
> If you want **real S2S forecast data**, please download it using the scripts in the `appendix/` folder.

---

## 📌 Repository Structure

```

AAI-S2S-Forecasting/
├── main/
│   └── Adaptive_and_Aggregated_Intelligence_Code_for_S2S.py   # Main script (runs independently)
├── appendix/
│   ├── Downloading_S2S_Files.py           # Download NCEP S2S forecast data
│   ├── Generation_of_Parameter_Sets.py    # Generate ML parameter sets
│   └── Validation_of_Heat_Wave_Warnings.py # Heatwave development & categorization
├── requirements.txt
├── LICENSE
└── README.md

````

✅ Main script runs independently  
✅ Appendix scripts are optional, required only for real S2S data

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/VenkateshBudamala/AAI-S2S-Forecasting.git
cd AAI-S2S-Forecasting
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Main Script (with random test values)

```bash
python main/Adaptive_and_Aggregated_Intelligence_Code_for_S2S.py
```

> ✅ Output: bias-corrected S2S forecasts (synthetic), performance metrics

### 4️⃣ (Optional) Use Real S2S Data

1. Run appendix scripts in order:

```bash
python appendix/Downloading_S2S_Files.py
python appendix/Generation_of_Parameter_Sets.py
python appendix/Validation_of_Heat_Wave_Warnings.py
```

2. Then run the main script with downloaded real data.

---

## 📦 Dependencies

* Python 3.8+
* numpy
* pandas
* sqlite3-binary
* torch
* scikit-learn
* xgboost
* tqdm

> All dependencies are listed in `requirements.txt`
> GPU optional — model runs on CPU if GPU unavailable

---

## 🧠 Model Features

* Adaptive learning for S2S forecast correction
* Ensemble ML: XGBoost, Random Forest, SVR
* Hyperparameter tuning & model validation
* Forecast skill improvement over raw S2S output
* Can run with **synthetic data** for quick testing

---

## 🏗️ Architecture Overview

```
S2S Data  --->  Preprocessing  --->  AAI Model Training  --->  Forecast Correction
                     |                       |
                     |                       └── Adaptive Learning
                     |
         Historical Observations
```

---

## 📬 Support / Questions

Please open an issue in this repository if you have questions or find bugs:
[https://github.com/VenkateshBudamala/AAI-S2S-Forecasting/issues](https://github.com/VenkateshBudamala/AAI-S2S-Forecasting/issues)

---

## 📄 License

This project is licensed under the **MIT License**.
You can find the full license text in the [LICENSE](./LICENSE) file.

© 2025 Venkatesh B.

