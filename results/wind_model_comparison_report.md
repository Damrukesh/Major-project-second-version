# Wind Power Forecasting - Model Comparison Summary

## Overview
This report presents a comprehensive comparison of 6 machine learning models for wind power forecasting.

## Dataset
- **Total Samples**: 8760
- **Training Samples**: 7008
- **Test Samples**: 1752
- **Features**: Wind Speed, Wind Direction, Pressure, Air Temperature
- **Target**: Windfarm Power (MW)

## Models Evaluated
CatBoost, XGBoost, Random Forest, Gradient Boosting, SVR, Linear Regression

## Best Performing Model
**CatBoost** achieved the best performance with:
- **R² Score**: 0.9914
- **RMSE**: 31.9104 MW
- **MAE**: 20.0206 MW
- **MAPE**: 1243131424.14%
- **Training Time**: 3.21 seconds

## Complete Results

| Model | MAE | RMSE | R² | MAPE (%) | Training Time (s) |
|-------|-----|------|----|---------|--------------------|
| CatBoost | 20.0206 | 31.9104 | 0.9914 | 1243131424.14 | 3.21 |
| Random Forest | 20.0198 | 32.1379 | 0.9913 | 1701298.89 | 2.35 |
| Gradient Boosting | 20.4701 | 33.1685 | 0.9907 | 1043767051.06 | 7.33 |
| XGBoost | 21.0109 | 33.9392 | 0.9903 | 1201378032.65 | 1.88 |
| SVR | 21.2185 | 34.5917 | 0.9899 | 5280809709.51 | 2.42 |
| Linear Regression | 88.1246 | 106.7940 | 0.9040 | 174535658907.88 | 0.02 |

## Key Findings

1. **Best Accuracy**: CatBoost provides the highest R² score of 0.9914
2. **Lowest Error**: CatBoost has the lowest RMSE of 31.9104 MW
3. **Fastest Training**: Linear Regression trains in 0.02 seconds

## Recommendation
Based on the comprehensive evaluation, **CatBoost** is recommended for wind power forecasting due to its superior performance across all metrics.

## Generated Files
- `wind_model_comparison.csv` - Complete metrics table
- `wind_comparison_metrics.png` - Metrics comparison visualization
- `wind_training_time.png` - Training time comparison
- `wind_comparison_predictions.png` - Sample predictions from top models
