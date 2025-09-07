# Diabetes Prediction using Machine Learning

This project applies supervised machine learning techniques to predict whether a patient is diabetic based on diagnostic health features. It uses the popular **PIMA Indians Diabetes Dataset** from the **National Institute of Diabetes and Digestive and Kidney Diseases**.

---
![image](https://github.com/user-attachments/assets/d12b27ed-0430-4e51-bd58-c61f7f63b15d)



## Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Instances**: 768 female patients of Pima Indian heritage (ages ≥ 21)
- **Target**: `Outcome` (1 = Diabetic, 0 = Non-diabetic)

### Features:
| Feature | Description |
|--------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Genetic risk indicator |
| Age | Age in years |

---

## Exploratory Data Analysis (EDA)

- Detected missing or zero values in features like `Insulin`, `SkinThickness`, and `BloodPressure`.
- Right-skewed distributions in `Insulin` and `DiabetesPedigreeFunction`.
- Strong correlation between `Glucose`, `BMI`, and diabetic outcome.

---

## Models Used
We apply and compare the performance of the following machine learning models:

1. **Support Vector Machine (SVM)**  
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **Gradient Boosting Classifier**
6. **K-Neighbors Classifier**

Each model was tuned using **GridSearchCV** for hyperparameter optimization.

---

## Handling Class Imbalance

- The dataset is imbalanced (65% non-diabetic, 35% diabetic).
- To reduce **false negatives** (Type II error), we performed **threshold tuning** on predicted probabilities.
- Evaluation metrics like **recall**, **F1-score**, and **ROC AUC** were prioritized over accuracy.

---

## Evaluation Metrics

- Accuracy: ~88-90%
- ROC AUC Score: Tracked for model comparison
- Confusion Matrix: Analyzed for Type I and II errors
- Threshold adjusted from 0.5 → 0.3 to reduce missed diabetic cases.

---

## Final Conclusion

This project demonstrated how machine learning can assist in early diabetes detection. By carefully analyzing features, adjusting decision thresholds, and using appropriate evaluation metrics, we built models that prioritize sensitivity and medical relevance.

> **Note**: In real-world healthcare applications, recall and interpretability are often more important than raw accuracy.

---

## Future Scope

- Try **SMOTE** or **class_weight='balanced'** for imbalance handling.
- Apply **XGBoost** or **LightGBM** for better recall.
- Use model **calibration** to improve probability-based decisions.
- Deploy the model as a **web app** using Streamlit or Flask.

---

## Author

- **Shivam Rawat** – M.Sc. Mathematics and Scientific Computing, NIT Warangal
- Project built for academic exploration and practical ML experience.
