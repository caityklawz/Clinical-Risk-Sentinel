# Clinical Risk Sentinel

A two-module clinical risk platform:
- **Sepsis Early-Warning**: This predicts sepsis onset from ICU vital-sign/lab time series (PhysioNet/CinC 2019 Challenge data)
- **Readmission Risk Scorer**: This predicts 30-day hospital readmission risk at discharge (UCI Diabetes 130-US Hospitals dataset)

Both datasets are public and de-identified. No models are deployed on real patients.

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py
```

This downloads and prepares both datasets into `data/` (not committed to git — see `.gitignore`).

## Running in Google Colab

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com)
2. In the first cell:
   ```python
   !git clone https://github.com/<your-username>/<your-repo>.git
   %cd <your-repo>
   !pip install -r requirements.txt
   !python src/download_data.py
   ```
3. Continue running the pipeline notebooks/scripts in subsequent cells.

## Project Structure

```
├── data/               # Downloaded datasets (gitignored)
├── notebooks/          # Exploration and prototyping
├── src/
│   ├── download_data.py    # Fetches both datasets
│   ├── preprocessing/      # Feature engineering pipelines
│   ├── models/              # Training and evaluation
│   └── explain/              # SHAP explainability
├── dashboard/           # Streamlit app
├── requirements.txt
└── README.md
```

## Datasets

- Clore, Cios, DeShazo, Strack (2014). *Diabetes 130-US Hospitals for Years 1999-2008*. UCI ML Repository. https://doi.org/10.24432/C5230J
- Reyna et al. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*. https://physionet.org/content/challenge-2019/1.0.0/
