# Assessment - 2 : Predicting Airbnb Listing Prices with MLflow and AWS S3
## Submitted By - Sandhya Sivakumar, c0956298

---

## Overview
Listing prices on StayWise vary significantly, even among similar properties. The business team wants a machine learning model that predicts the optimal nightly price for new listings based on factors such as location, amenities, and reviews. The dataset from S3 is noisy, containing missing values, outliers, and categorical fields that require extensive preprocessing

This assessment builds a complete machine learning workflow to predict nightly Airbnb listing prices using regression models, MLflow experiment tracking, and AWS S3 for dataset storage.

The repository follows a clean, reproducible workflow including data preprocessing, model training, experiment tracking, model comparison, and registering the best-performing model in the MLflow Model Registry.

---

## Objectives

1. Retrieve the Airbnb dataset stored in AWS S3.
2. Clean and preprocess the dataset, including feature engineering and outlier handling.
3. Train and compare multiple regression models.
4. Track parameters, metrics, and artifacts using MLflow.
5. Register the best-performing model in the MLflow Model Registry.
6. Document the full workflow in a structured GitHub repository.

---

## Dataset Source (AWS S3)

The raw Airbnb listings dataset was stored in AWS S3 under the following path:

`` s3://airbnb-pricing-ml-project-sandhya/airbnb/raw_data/AB_NYC_2019.csv``

The file was accessed using boto3.
![s3_data.png](results%2Fs3_data.png)
---

## Step 1: Data Cleaning and Preprocessing

The raw Airbnb dataset contained several quality issues such as missing values, invalid date formats, extreme outliers, and categorical fields requiring transformation. A systematic preprocessing workflow was applied to ensure the dataset was clean, consistent, and ready for modelling.

### Key Preprocessing Steps

- **Handling Missing Values**
  - Filled missing values in `reviews_per_month` with **0**, assuming no reviews.
  - Replaced missing `last_review` dates with a default timestamp (`2000-01-01`).
  - Ensured no nulls remained in important numerical or date fields.

- **Fixing Data Types**
  - Converted the `last_review` column to proper datetime format using `errors='coerce'`.
  - Extracted a new feature:  
    **`last_review_year` = year component of last_review**.

- **Outlier Treatment**
  - Clipped extreme values in `price` (range: **10 to 1000**) to reduce skew and stabilize model training.
  - Capped `minimum_nights` at **30** to prevent inflated values from affecting model behaviour.
  
  **Before & After Outlier Visualization:**  
![outlier_handling.png](results%2Foutlier_handling.png)

---

## Step 2: Feature Engineering and Train–Test Split

After cleaning the dataset, additional steps were performed to prepare the data for modelling. These steps ensure consistent feature formatting, correct handling of categorical variables, and a reproducible workflow for training and evaluating models.

### Feature Engineering

- Removed the raw `last_review` column since its useful information was already captured in the engineered `last_review_year` feature.
- Separated the target variable (`price`) from the feature matrix.
- Identified and grouped variables into:
  - **Categorical features:**  
    - `neighbourhood_group`, `neighbourhood`, `room_type`
  - **Numerical features:**  
    - `latitude`, `longitude`, `minimum_nights`,  
      `number_of_reviews`, `reviews_per_month`,  
      `calculated_host_listings_count`, `availability_365`,  
      `last_review_year`
- Defined a preprocessing pipeline using `ColumnTransformer`:
  - **StandardScaler** for numerical features  
  - **OneHotEncoder** for categorical features  
- This pipeline ensures consistent transformations across all models and MLflow runs.

**Preprocessing Pipeline**  
![preprocessor.png](results%2Fpreprocessor.png)

### Train–Test Split

The dataset was split into training and testing sets to ensure unbiased evaluation of model performance.

- **80% training data**  
- **20% test data**  
- `random_state=42` was used for reproducibility

This split resulted in:
- **39,116 rows** in the training set  
- **9,779 rows** in the test set  

This step completes the dataset preparation by delivering a clean, transformed, and properly partitioned structure required for developing robust regression models.

---

## Step 3: Model Development

To evaluate different approaches for predicting Airbnb listing prices, five regression models were developed and trained using the fully preprocessed dataset:

- **Linear Regression** – baseline model capturing simple linear relationships  
- **Ridge Regression** – linear model with L2 regularization to reduce overfitting  
- **Lasso Regression** – linear model with L1 regularization for feature sparsity  
- **Random Forest Regressor** – ensemble of decision trees that captures strong non-linear patterns  
- **Gradient Boosting Regressor** – sequential tree-based model that improves residual errors iteratively  

Each model was integrated with the preprocessing pipeline and trained on the training split.  
Performance was evaluated using industry-standard regression metrics:

- **MAE (Mean Absolute Error)** – average absolute difference between predicted and actual prices  
- **RMSE (Root Mean Squared Error)** – penalizes larger errors more strongly  
- **R² (Coefficient of Determination)** – measures how much variance in price is explained by the model  

All metrics, parameters, preprocessing details, and model artifacts were logged automatically using **MLflow autologging**, enabling seamless comparison across all models.

**Model Training & Results**  
![initial_model_perf.png](results%2Finitial_model_perf.png)

---

## Step 4: Experiment Tracking with MLflow

MLflow was used to track every stage of the model development process, ensuring full reproducibility and easy comparison across different regression models.  
With **MLflow Autologging**, all essential components of each run were recorded automatically, including:

- **Model parameters** such as regularization strength, estimator type, and hyperparameters  
- **Evaluation metrics** (MAE, RMSE, R²) generated during testing  
- **Artifacts**, including preprocessing pipelines, serialized models, and feature metadata  
- **Input and output schema**, enabling consistent inference during model deployment  
- **Source code snapshot**, capturing the exact notebook state for each experiment  
- **Unique Run IDs**, making every experiment repeatable and traceable  

MLflow’s experiment UI also allowed visual comparison of metrics across models, helping identify the best-performing algorithm quickly and objectively.

### MLflow Home Page
![mlflow_homepage.png](results%2Fmlflow_homepage.png)

### Experiment Runs Table
![mlflow_runs_table.png](results%2Fmlflow_runs_table.png)

### Metrics Comparison Page
![metrics_comparison.png](results%2Fmetrics_comparison.png)

---

## Step 5: Best Model Selection

After comparing all five regression models using MAE, RMSE, and R², the **Random Forest Regressor** emerged as the clear top performer.

### Best Performing Model: Random Forest Regressor

- **MAE:** 51.66  
- **RMSE:** 97.28  
- **R²:** 0.420  

The Random Forest model achieved the **lowest error values** and the **highest explanatory power** among all tested models. Its ensemble-based structure allowed it to capture complex, non-linear relationships in the pricing data far better than the linear models and Gradient Boosting.  

This consistent performance across all metrics made Random Forest the most reliable and stable choice for predicting nightly Airbnb prices. As a result, it was selected as the final model and later **registered in the MLflow Model Registry** for versioning and potential deployment.

---

## Step 6: MLflow Model Registry

After identifying the Random Forest Regressor as the best-performing model, it was registered in the **MLflow Model Registry** to ensure proper versioning, lifecycle management, and reproducibility.  

MLflow automatically created a **versioned model entry**, allowing the model to be tracked, reviewed, and promoted through stages such as *Staging* or *Production* if needed. The registry also stored the preprocessing pipeline, model artifacts, input/output schema, and metadata associated with the training run.

This version-controlled registration ensures that the selected model can be reliably reused, compared against future models, and integrated into downstream systems if deployed.

### Model Registry Page
![model_registry.png](results%2Fmodel_registry.png)

---

## Setup and Execution Instructions

### 1. Clone the repository

- cd Assessment 2

### 2. Create virtual environment
- python -m venv .venv

### 3. Activate environment  
- Windows: .venv\Scripts\activate
- Mac/Linux: source .venv/bin/activate


### 4. Install dependencies

- pip install -r requirements.txt

### 5. Launch MLflow UI
- mlflow ui


### 6. Run the Jupyter notebook

- notebook/airbnb_price_prediction.ipynb

---

## Key Insights and Observations

- **Data Quality and Preprocessing Had a Major Impact on Performance**  
  The raw Airbnb dataset contained missing values, inconsistent review dates, and extreme outliers in features such as `minimum_nights`, `availability_365`, and `number_of_reviews`.  
  Applying careful cleaning, including date parsing, imputation, outlier clipping, and structured feature encoding, significantly stabilized model performance.  
  Without this preprocessing pipeline, the models produced noticeably higher errors, highlighting the importance of a robust data-preparation workflow.

- **Linear-Family Models Had Very Similar Predictive Behavior**  
  Linear Regression, Ridge, and Lasso all produced nearly identical metrics (MAE ≈ 58, RMSE ≈ 103, R² ≈ 0.34–0.35).  
  This suggests:
  - The relationship between most features and price is *not strongly linear*.  
  - Regularization does not meaningfully change predictive power for this dataset.  
  - These models struggle to capture location-based patterns, neighborhood interactions, and non-linear influences like review count, listing saturation, or seasonal availability.

- **Gradient Boosting Provided Moderate Improvements but Was Not the Best Model**  
  Gradient Boosting models are typically strong for tabular data, and here it did show improvement compared to linear models.  
  However:
  - Its errors were still higher than Random Forest.
  - Boosting may require deeper tuning (learning rate, number of trees, max depth) to reach optimal performance.
  - With default settings, it captured some non-linear behaviors but not as effectively as the Random Forest ensemble.

- **Random Forest Regressor Clearly Outperformed All Other Models**  
  Random Forest achieved:
  - **Lowest MAE (~51.66)**
  - **Lowest RMSE (~97.28)**
  - **Highest R² (~0.42)**  
  This implies:
  - Non-linear relationships dominate pricing behavior.
  - Feature interactions (e.g., neighborhood × room type) are important for prediction.
  - Ensemble averaging helps reduce overfitting and handles noisy data better than other methods.
  Random Forest’s robustness made it the most reliable model given the current feature set.

- **MLflow Greatly Enhanced Experiment Tracking and Model Management**  
  MLflow allowed automatic logging of:
  - Parameters (model type, hyperparameters)
  - Metrics (MAE, RMSE, R²)
  - Artifacts (preprocessors, feature lists, pipelines)
  - Trained models  
  Using the MLflow UI made it easy to compare models side by side, observe trends in errors, and visualize performance differences.  
  Registering the best-performing Random Forest model in the **MLflow Model Registry** ensured a production-ready workflow aligned with modern MLOps best practices.

- **Feature Engineering Decisions Strongly Influenced Model Outcomes**  
  Converting review dates to `last_review_year`, encoding categorical variables, and applying consistent numerical scaling helped the models generalize better.  
  The engineered feature `last_review_year` added temporal relevance — listings with more recent engagement tended to have more predictable price behavior.

- **Outlier Handling Improved Model Stability**  
  Extreme values in features like `minimum_nights` and `availability_365` introduced instability in the linear models and inflated prediction errors.  
  Clipping or constraining these outliers resulted in:
  - Smoother residuals
  - More stable model convergence
  - Reduced error variance  
  This reinforces that outlier treatment is essential for price-prediction tasks in marketplace datasets.

- **Model Performance Suggests Additional Features Could Improve Accuracy**  
  R² ≈ 0.42 for the best model indicates room for improvement.  
  Potential enhancements include:
  - Adding amenities data (WiFi, AC, TV, kitchen, etc.)
  - Location-based geospatial features (distance to downtown, transport hubs)
  - Host-related features (superhost status, response time)
  - Seasonal demand variables  
  These features could help models capture deeper pricing patterns beyond the current dataset.

- **The Workflow is Fully Reproducible and Production-Friendly**  
  With all experiments logged, the repository structured cleanly, and the best model registered, the entire process satisfies the assignment’s requirement for reproducible ML development:
  - Data → Preprocessing → Modeling → MLflow Tracking → Best Model Selection → Registry.

Overall, the assignment demonstrates that **ensemble tree-based models are currently the best fit for predicting Airbnb listing prices**, and that **data preprocessing plus MLflow-driven experiment tracking** greatly enhance both performance and reproducibility.

---

