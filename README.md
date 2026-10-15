# House Price Prediction

A hands-on project to learn Data Science and Machine Learning with Python, building a house price prediction model step by step.

The project follows an iterative approach: exploring data, training models, evaluating results and progressively implementing more advanced machine learning techniques.

## Roadmap

### Phase 1 — Data Exploration and Preprocessing
**Status: Completed**

- Loaded and explored the housing dataset using Pandas.
- Analyzed descriptive statistics and visualized relationships between variables.
- Identified missing values and applied basic data cleaning techniques.
- Saved the processed dataset in Parquet format.

### Phase 2 — Simple Linear Regression
**Status: Completed**

- Trained a linear regression model using house surface area as the only input feature.
- Generated predictions for unseen observations.
- Learned how to interpret model coefficients and predictions.

### Phase 3 — Model Evaluation
**Status: Completed**

- Split the dataset into training and test sets.
- Evaluated model performance using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Coefficient of Determination (R²)
- Visualized actual versus predicted house prices.
- Established a baseline model for future comparisons.

**Baseline results:**

| Metric | Simple Linear Regression |
|---|---:|
| MAE | €28,348 |
| RMSE | €35,022 |
| R² | 0.71 |

### Phase 4 — Multiple Linear Regression
**Status: Completed**

- Extended the model to use seven housing features:
  - Surface area
  - Number of rooms
  - Number of bathrooms
  - Year of construction
  - Distance to the city center
  - Garage availability
  - Elevator availability
- Compared the new model against the simple linear regression baseline.

**Results:**

| Metric | Simple Regression | Multiple Regression |
|---|---:|---:|
| MAE | €28,348 | €19,532 |
| RMSE | €35,022 | €24,422 |
| R² | 0.71 | 0.86 |

The multiple linear regression model achieved better results on the current test set, showing the potential value of using additional property features.

### Phase 5 — Non-Linear Machine Learning Models
**Status: Planned**

- Implement a Random Forest Regressor.
- Explore Gradient Boosting models.
- Compare their performance with linear regression.
- Investigate whether non-linear relationships improve price predictions.

### Phase 6 — Model Validation and Optimization
**Status: Planned**

- Apply cross-validation.
- Tune model hyperparameters.
- Analyze prediction errors and identify where the models perform poorly.
- Compare models using consistent evaluation procedures.

### Phase 7 — Model Interpretation and Feature Analysis
**Status: Planned**

- Analyze feature importance.
- Study the relationship between housing characteristics and predicted prices.
- Understand the strengths and limitations of each model.

### Phase 8 — Interactive Prediction Application
**Status: Planned**

- Build a simple web application using Streamlit.
- Allow users to enter property characteristics.
- Generate predictions using the selected model.
- Document how to run the application locally.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib
- SciPy
- Scikit-learn
- PyArrow

## Project Structure

```text
.
├── data/
│   └── houses.csv
├── src/
│   ├── exploration.py
│   └── train_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The project uses a synthetic housing dataset created for learning and experimentation. It includes property characteristics and house prices.

The results should not be interpreted as real-world property valuation estimates.

## Project Goal

The goal is to progressively develop practical Data Science and Machine Learning skills, from data preparation and basic regression to model comparison, optimization and deployment.

Each phase builds on the previous one, with an emphasis on understanding the methods and evaluating their results.