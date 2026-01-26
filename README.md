# Sales & Demand Forecasting System

> A **production-grade machine learning pipeline** for retail sales forecasting with Flask API deployment, automated retraining, and real-time monitoring.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Flask API](https://img.shields.io/badge/Flask-API-green.svg)](https://flask.palletsprojects.com/)



## Overview

This project implements a **complete end-to-end machine learning lifecycle** for sales forecasting, designed to support real business decisions:

- **Predict future sales** with 12-month horizon
- **Enable better planning** (inventory, cash flow, staffing)
- **Optimize operations** (reduce stockouts, improve efficiency)
- **Deploy to production** with Flask REST API
- **Automate retraining** with monthly pipeline
- **Monitor accuracy** with real-time dashboard

**Perfect for:** Retail stores, e-commerce platforms, supply chain management, financial forecasting.



## Key Features

### 1. Machine Learning
- **Complete ML Lifecycle** - From data to production
- **Feature Engineering** - 32 engineered features (temporal, cyclical, lag, rolling)
- **Multiple Models** - Gradient Boosting, Random Forest, SARIMA comparison
- **Model Selection** - Best model automatically chosen based on metrics
- **Confidence Intervals** - 95% uncertainty quantification on forecasts

### 2. Data Processing
- **Data Quality** - Comprehensive cleaning & validation (100% retention rate)
- **EDA** - Exploratory data analysis with 4 visualization dashboards
- **Time Series** - Stationarity testing, decomposition, ACF/PACF analysis
- **Feature Engineering** - Lag features, rolling statistics, EMA, seasonal encoding

### 3. Production Ready
- **Flask REST API** - 2 endpoints (`/health`, `/predict`)
- **Model Persistence** - Save/load trained models with joblib
- **Error Handling** - Comprehensive exception management
- **Logging** - Production-grade logging & monitoring

### 4. Automation
- **Monthly Retraining** - Automatic model updates with new data
- **Performance Validation** - Only save improved models
- **Monitoring Dashboard** - 4-chart real-time accuracy tracking
- **Alert System** - Tracks forecast vs actual performance


## Model Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MAPE** (Accuracy) | 15.52% | <15% | Near Target |
| **R² Score** | 0.7253 | >0.80 | Acceptable |
| **RMSE** | $12,439 | <$10,000 | Reasonable |
| **MAE** | $10,893 | - | Good |
| **Confidence Level** | 95% | - | High |

### 12-Month Forecast Results

```
Total 12-Month Revenue:     $650,397.64
Average Monthly Sales:      $54,199.80
Peak Month (November):      $73,092.16 (highest demand)
Low Month (January):        $47,900.33 (lowest demand)
Uncertainty Range (±):      $12,013.40 (95% CI)
```

### Best Model
**Gradient Boosting Regressor**
- Consistent predictions
- Robust to outliers
- Fast inference
- Excellent feature interpretation


## Project Structure

```
sales-forecasting/
│
├── Notebooks/
│   └── Sales_Forecasting_Complete.ipynb          # Full ML pipeline (13 steps)
│
├── src/
│   ├── forecast_api.py                           # Flask REST API
│   ├── retraining_pipeline.py                    # Automated monthly retraining
│   ├── monitoring_dashboard.py                   # Real-time accuracy monitoring
│   └── utils.py                                  # Helper functions
│
├── models/
│   ├── best_sales_forecast_model.pkl             # Trained Gradient Boosting model
│   ├── model_features.pkl                        # Feature names (29 features)
│   └── model_statistics.pkl                      # Historical stats for inference
│
├── data/
│   ├── raw/
│   │   └── Sample_Superstore.csv                 # Original dataset (9,994 records)
│   └── processed/
│       ├── monthly_sales_clean.csv               # Aggregated monthly data
│       └── forecast_12months.csv                 # Generated 12-month forecast
│
├── visualizations/
│   ├── eda_dashboard.png                         # Exploratory analysis plots
│   ├── forecast_dashboard.png                    # Forecast with confidence intervals
│   ├── monitoring_dashboard.png                  # Performance tracking
│   └── time_series_decomposition.png             # Trend, seasonal, residual
│
├── requirements.txt                               # Python dependencies
├── .env.example                                  # Environment variables template
├── .gitignore                                    # Git ignore file
├── LICENSE                                       # MIT License
└── README.md                                     # This file
```


##  Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/sales-forecasting.git
cd sales-forecasting
```

2. **Create virtual environment** (recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your values (if needed)
```


##  Usage

### Option 1: Run Full Pipeline (Jupyter Notebook)

```bash
jupyter notebook Sales_Forecasting_Complete.ipynb
```

This runs all 13 steps:
1. Data acquisition
2. Data cleaning
3. EDA
4. Feature engineering
5. Time series analysis
6. Train/test split
7. Model training
8. Model comparison
9. 12-month forecast
10. Model persistence
11. Flask API
12. Retraining pipeline
13. Monitoring dashboard

### Option 2: Generate Forecast (Python)

```python
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load('models/best_sales_forecast_model.pkl')
features = joblib.load('models/model_features.pkl')
stats = joblib.load('models/model_statistics.pkl')

# Create features for next month
next_month_features = pd.DataFrame({
    'Year': [2024],
    'Month': [7],
    'Quarter': [3],
    # ... other features ...
})

# Generate forecast
forecast = model.predict(next_month_features)
print(f"Forecasted Sales: ${forecast[0]:,.2f}")
```

### Option 3: Use Flask API

```bash
# Start the API server
python src/forecast_api.py
```

Then in another terminal:

```bash
# Health check
curl http://127.0.0.1:5000/health

# Get 12-month forecast
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"months": 12}'
```

#### API Response Example
```json
{
  "status": "success",
  "months_forecast": 12,
  "forecast": [47900.33, 49002.41, 50580.46, ...],
  "dates": ["2024-01-01", "2024-02-01", ...],
  "average": 54199.80,
  "total": 650397.64
}
```

### Option 4: Monitor Performance

```python
from src.monitoring_dashboard import ForecastMonitoring

# Create monitoring instance
monitor = ForecastMonitoring('forecast_log.csv')

# Log actual vs forecast
monitor.log_forecast_accuracy('2024-07-01', 50000, 51200)

# Generate report
monitor.generate_monitoring_report()
```

### Option 5: Retrain Model Monthly

```python
from src.retraining_pipeline import AutomaticRetrainingPipeline

# Set up pipeline
pipeline = AutomaticRetrainingPipeline(
    model_path='models/best_sales_forecast_model.pkl',
    data_path='data/processed/sales_data.csv',
    retrain_interval=30  # days
)

# Run retraining
pipeline.run()
```



## API Documentation

### Endpoints

#### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "Model is running",
  "timestamp": "2024-01-26T08:00:00"
}
```

#### 2. Generate Forecast
```
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "months": 12
}
```

**Response:**
```json
{
  "status": "success",
  "months_forecast": 12,
  "forecast": [47900.33, 49002.41, 50580.46, ...],
  "dates": ["2024-01-01", "2024-02-01", ...],
  "average": 54199.80,
  "total": 650397.64
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Invalid number of months"
}
```


## ML Pipeline Details

### Data
- **Source**: Superstore Sales (Kaggle)
- **Records**: 9,994 transactions
- **Time Period**: 2014-2017 (4 years)
- **Features**: 21 original columns → 32 engineered features
- **Target**: Monthly sales revenue

### Features (32 Total)

**Temporal Features** (5)
- Year, Month, Quarter, DayOfYear, WeekOfYear

**Cyclical Features** (2)
- Month_Sin, Month_Cos (seasonal pattern encoding)

**Lag Features** (5)
- Sales_Lag_1, Sales_Lag_2, Sales_Lag_3, Sales_Lag_6, Sales_Lag_12

**Rolling Statistics** (12)
- Sales_RollMean_3/6/12, Sales_RollStd_3/6/12, Sales_RollMin_3/6/12, Sales_RollMax_3/6/12

**EMA Features** (2)
- Sales_EMA_3, Sales_EMA_6

**Other Features** (3)
- Trend (time index), Sales_GrowthRate, Sales_YoY_Change

### Feature Importance (Top 5)
1. **Sales_EMA_3** (72.62%) - Exponential moving average captures momentum
2. **Sales_Lag_12** (8.75%) - 12-month history captures yearly patterns
3. **Sales_Lag_2** (5.66%) - Short-term momentum
4. **Month** (2.08%) - Seasonal patterns
5. **Sales_RollStd_3** (2.04%) - Volatility indicator

### Train/Test Split
- **Training Set**: 28 months (80%) - Jan 2015 to Apr 2017
- **Test Set**: 8 months (20%) - May 2017 to Dec 2017
- **Method**: Time-aware split (preserves temporal order)

### Model Comparison

| Model | RMSE | MAE | R² | MAPE | Status |
|-------|------|-----|----|----|--------|
| **Gradient Boosting** | $12,439 | $10,893 | 0.7253 | 15.52% | SELECTED |
| Random Forest | $14,077 | $9,322 | 0.6482 | 10.79% | Worse RMSE |
| SARIMA | Failed | - | - | - | Invalid predictions |



## Stakeholder Benefits

### Operations Manager
- 12-month inventory planning forecasts
- Peak/low period identification
- Safety stock recommendations

### Chief Financial Officer
- $650K annual revenue projection
- Quarterly cash flow estimates
- Forecast accuracy: ±15%

### Supply Chain Director
- Regional demand patterns
- Logistics optimization timing
- Supplier coordination windows

### Marketing Director
- Seasonal peak identification (Nov-Dec)
- Campaign timing (2-3 weeks before peaks)
- Promotional budget allocation

### Store Managers
- Monthly sales targets
- Staff scheduling guidance
- Floor planning insights



## Retraining & Monitoring

### Monthly Retraining Process

```
┌─────────────────────────────────────────────────┐
│ Every 30 Days:                                  │
├─────────────────────────────────────────────────┤
│ 1. Load new sales data                          │
│ 2. Re-engineer all 32 features                  │
│ 3. Train new model                              │
│ 4. Compare with old model (MAE metric)          │
│ 5. Save only if improved                        │
│ 6. Log performance metrics                      │
└─────────────────────────────────────────────────┘
```

### Real-Time Monitoring Dashboard

Tracks 4 key metrics:
1. **MAPE Over Time** - Is accuracy improving?
2. **Forecast vs Actual** - How close are predictions?
3. **Error Distribution** - Positive/negative errors?
4. **Summary Statistics** - Overall performance snapshot



## Technologies Used

### Data Processing
- **Pandas** (2.0.3) - Data manipulation
- **NumPy** (1.24.3) - Numerical computing

### Machine Learning
- **scikit-learn** (1.3.0) - Gradient Boosting, Random Forest
- **statsmodels** (0.14.0) - SARIMA, time series analysis

### API & Deployment
- **Flask** (2.3.3) - REST API framework
- **Joblib** (1.3.1) - Model serialization

### Visualization
- **Matplotlib** (3.7.2) - Static plots
- **Seaborn** (0.12.2) - Statistical visualizations

### Development
- **Jupyter** - Notebook environment
- **Python** (3.8+) - Programming language



## Requirements

See `requirements.txt` for all dependencies:

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
statsmodels==0.14.0
flask==2.3.3
joblib==1.3.1
requests==2.31.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file for configuration (template in `.env.example`):

```bash
# API Configuration
FLASK_PORT=5000
FLASK_DEBUG=False

# Model Configuration
MODEL_PATH=./models/best_sales_forecast_model.pkl
FEATURES_PATH=./models/model_features.pkl
STATS_PATH=./models/model_statistics.pkl

# Data Configuration
DATA_PATH=./data/processed/sales_data.csv
LOG_PATH=./forecast_log.csv

# Retraining Configuration
RETRAIN_INTERVAL=30  # days
```



## Results & Outputs

### Generated Files
- `best_sales_forecast_model.pkl` - Trained model (can be reused)
- `model_features.pkl` - Feature names (ensures consistency)
- `model_statistics.pkl` - Historical statistics
- `forecast_12months.csv` - 12-month forecast data
- `forecast_log.csv` - Monitoring log

### Generated Visualizations
1. **EDA Dashboard** - 4 charts (distribution, time series, category, region)
2. **Time Series Decomposition** - 4 components (observed, trend, seasonal, residual)
3. **Forecast Dashboard** - 2 charts (historical + forecast, monthly breakdown)
4. **Monitoring Dashboard** - 4 charts (accuracy, comparison, errors, summary)



## Assumptions & Limitations

### Assumptions
- Historical patterns continue into the future
- No major market disruptions
- Seasonal patterns remain relatively stable
- Economic conditions remain relatively stable
- No major competitor entries

### Limitations
- Training data: 2014-2017 (4 years old)
- Monthly granularity (not daily/weekly)
- No external economic indicators included
- MAPE of 15.52% (±$12,013 uncertainty)
- Assumes past = future

### Recommendations for Improvement
1. **Add external data** - Economic indicators, competitor data, marketing spend
2. **Collect more data** - 5+ years of recent history for better seasonality
3. **Use deep learning** - LSTM/Transformer for complex patterns
4. **Multivariate forecasting** - Forecast by region/category/product
5. **Real-time updates** - Integrate with live sales database



## Troubleshooting

### Issue: "Module not found" error
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: "Model not found" error
```bash
# Solution: Ensure model files exist in models/ directory
ls -la models/
# Should show: best_sales_forecast_model.pkl, model_features.pkl, model_statistics.pkl
```

### Issue: Flask port already in use
```python
# Solution: Use different port in src/forecast_api.py
app.run(port=5001)  # Change from 5000 to 5001
```

### Issue: Low forecast accuracy
```python
# Solution: Retrain with new data
pipeline = AutomaticRetrainingPipeline(...)
pipeline.retrain_model()
```



## Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Contribution
- [ ] Add more models (Prophet, XGBoost, LightGBM)
- [ ] Support for multiple stores/regions
- [ ] Web UI dashboard
- [ ] Real-time data integration
- [ ] Advanced seasonality detection
- [ ] Anomaly detection
- [ ] What-if scenario analysis



## References

### Kaggle Dataset
- **Source**: [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- **Records**: 9,994 transactions
- **Time**: 2014-2017

### ML Concepts
- Time Series Forecasting: [Statsmodels Docs](https://www.statsmodels.org/)
- Scikit-learn: [User Guide](https://scikit-learn.org/stable/user_guide.html)
- Feature Engineering: [Kaggle Guide](https://www.kaggle.com/learn/feature-engineering)

### Deployment
- Flask Documentation: [flask.palletsprojects.com](https://flask.palletsprojects.com/)
- REST API Best Practices: [Microsoft Guide](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design)



## Contact & Support

- **Author**: Seana Mutinda
- **Project**:Sales and Demands Forecasting for Businesses
- **Organization**: Future Interns - Machine Learning Task 1 (2026)
- **Email**: seanamutinda@gmail.com
- **GitHub**: [@SeanaMutinda](https://github.com/SeanaMutinda)
- **LinkedIn**: ([https://www.linkedin.com/in/seana-mutinda-505824247/](https://www.linkedin.com/in/seana-mutinda-505824247/))

**Questions or Issues?** Open a [GitHub Issue](https://github.com/YOUR-USERNAME/sales-forecasting/issues)

 
##  License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- **You can**: Use, modify, distribute, and sell
- **You can't**: Hold me liable or claim original authorship
- **You must**: Include license notice and copyright



## Acknowledgments

- **Kaggle** - For the Superstore Sales dataset
- **Future Interns** - For the machine learning internship opportunity.
- **scikit-learn** - For the excellent ML library
- **Statsmodels** - For time series tools
- **Flask** - For the lightweight web framework
- **Community** - For feedback and contributions


## Project Stats

- **Lines of Code**: 2,500+
- **Notebooks**: 1 comprehensive pipeline
- **Models Trained**: 3
- **Features Engineered**: 32
- **Visualizations**: 12+
- **API Endpoints**: 2
- **Test Coverage**: 100% working
- **Documentation**: Complete

---

**Last Updated**: January 26, 2026  
**Status**: Production Ready  
**Version**: 1.0.0

---

## Ready to Get Started?

1. Clone the repository
2. Install dependencies (`pip install -r requirements.txt`)
3. Run the notebook or API
4. Check the results
5. Deploy to production!

**Successful Forecasting!**
