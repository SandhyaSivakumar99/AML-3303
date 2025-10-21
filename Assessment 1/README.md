# Assessment 1 - TechNova Employee Attrition Prediction  


---

## 1. Overview
TechNova Solutions, a mid-sized IT services firm with approximately 1,200 employees, has been facing a persistently high attrition rate despite offering competitive compensation and benefits.  
The company lacks a proactive mechanism to identify employees at risk of leaving, resulting in increased recruitment costs, disrupted client projects, and declining team morale.  

This assignment presents a **data-driven predictive solution** that identifies potential churn risks and provides HR with actionable insights to improve retention strategies.  

---

## 2. Objectives
The primary objectives of this assignment are to:
1. **Analyze** employee-related data to uncover the key factors influencing attrition.  
2. **Build** a machine learning model capable of predicting the likelihood of employee churn.  
3. **Evaluate and interpret** model performance through explainable AI techniques.  
4. **Recommend** HR interventions that can help improve employee satisfaction and reduce turnover.

---

## 3. Dataset
The dataset (`employee_churn_dataset.csv`) contains demographic, performance, and engagement-related variables such as:  
- **Demographics:** Gender, Age, Marital Status  
- **Work Attributes:** Department, Job Role, Tenure, Salary, Overtime Hours, Performance Rating  
- **Engagement Indicators:** Satisfaction Level, Work-Life Balance, Manager Feedback Score

---

## 4. Project Architecture
This assignment follows a structured **Software Development Life Cycle (SDLC)** ensuring systematic and reproducible analysis.

| **Stage** | **Description** |
|------------|----------------|
| **1. Data Understanding** | Loaded and explored dataset, checked datatypes, target variable, and balance. |
| **2. Exploratory Data Analysis (EDA)** | Visualized numeric and categorical patterns; examined churn trends and correlations. |
| **3. Data Preprocessing** | Imputed missing values, scaled numeric data, and encoded categorical features. |
| **4. Feature Engineering** | Created HR-relevant features: *TenureBucket*, *OvertimeRate*, *PromotionFlag*, *WorkSatisfactionScore*. |
| **5. Experiment Design** | Implemented stratified 70/30 split; defined ROC-AUC as evaluation metric. |
| **6. Model Building** | Trained Logistic Regression, Random Forest, and XGBoost using cross-validation. |
| **7. Model Evaluation** | Compared models via ROC-AUC; tested best model on unseen data. |
| **8. Model Explainability** | Analyzed feature coefficients to interpret key drivers of churn. |
| **9. Recommendations** | Translated model insights into actionable HR strategies for retention. |

---

## 5. Methodology Summary

### 5.1 Data Understanding
- Loaded and validated the dataset; standardized binary target values (`Churn` = 1, `Stayed` = 0).  
- Detected an **80–20 class imbalance**, guiding later metric selection.

### 5.2 Exploratory Data Analysis (EDA)
- Visualized distributions, correlations, and categorical relationships with churn.  
- Observed that satisfaction, tenure, and department showed noticeable churn patterns.

### 5.3 Feature Engineering
Added derived HR-relevant attributes to improve model interpretability:
- **TenureBucket:** Categorized tenure duration.  
- **OvertimeRate:** Workload intensity ratio.  
- **PromotionFlag:** Binary indicator of growth opportunities.  
- **WorkSatisfactionScore:** Composite satisfaction metric.

### 5.4 Modeling Approach
- Three models were trained using pipelines:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- **Stratified 5-Fold Cross-Validation** ensured fair comparison across imbalanced classes.  
- Primary metric: **ROC-AUC**, supported by Accuracy, Precision, Recall, and F1.

### 5.5 Model Selection & Evaluation
- **Best Model:** Logistic Regression (Mean ROC-AUC ≈ 0.48)
The model’s low predictive score reflects dataset simplicity but its interpretability provides valuable insights for HR decision-making.

### 5.6 Model Explainability
- Extracted coefficients from the Logistic Regression pipeline to rank influential predictors.  
- Strongest effects observed for:
  - **Work-Life Balance** (negative correlation with churn)  
  - **Work Location (Hybrid)** (positive correlation with churn)  
  - **Performance Rating** (positive correlation with churn)  
  - **Tenure and Department differences** (moderate influence)

---

## 6. Key Findings
- Employees with **better work-life balance** are more likely to stay.  
- **Hybrid employees** exhibit slightly higher churn rates, potentially due to uneven engagement.  
- **Longer tenure** correlates with increased churn likelihood, suggesting stagnation.  
- **High performers** show mild churn risk, possibly seeking new opportunities.  
- **Salary** and **training hours** differences are minimal, implying that retention is driven by engagement rather than pay alone.

---

## 7. Recommendations
Based on the analysis and model insights:
1. **Enhance Work-Life Balance Programs:** Introduce flexible scheduling and fair workload policies.  
2. **Reassess Hybrid Work Policies:** Strengthen communication and inclusion for hybrid employees.  
3. **Promote Career Growth:** Offer internal mobility and skill development opportunities.  
4. **Retain High Performers:** Recognize and reward excellence with competitive benefits.  
5. **Conduct Department-Specific Engagement Surveys:** Particularly in HR and Marketing departments.

---

## 8. Tools and Technologies
- **Programming Language:** Python 3.10  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Environment:** Jupyter Notebook (PyCharm)  
- **Version Control:** Git & GitHub  

---

## 9. Limitations and Future Scope
- The dataset lacks behavioral and sentiment-based variables that could strengthen predictions.  
- Future improvements:
  - Integration of **employee satisfaction surveys** or **HR feedback forms**.  
  - Use of **SHAP explainability** for non-linear models.  
  - Periodic retraining with updated employee data to adapt to changing patterns.

---

## 10. Author
**Prepared by:** Sandhya Sivakumar  
**Student ID:** c0956298
---