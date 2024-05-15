#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# Streamlit app
def main():
    st.title("Non-Linear Regression Analysis")

    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Allow user to select the outcome and independent variables
        outcome_var = st.selectbox("Select the outcome variable:", df.columns)
        independent_vars = st.multiselect("Select independent variables:", df.columns)
        
        if outcome_var and independent_vars:
            df2 = df[independent_vars]
            y = df[outcome_var]

            # Perform statistical tests and plots
            st.header("Statistical Tests and Plots")

            # Bartlett’s Test of Sphericity
            chi2, p = calculate_bartlett_sphericity(df2)
            st.write("Bartlett’s Test of Sphericity:")
            st.write(f"Chi-squared value: {chi2}, p-value: {p}")

            # Kaiser-Meyer-Olkin (KMO) Test
            kmo_values, kmo_model = calculate_kmo(df2)
            st.write("Kaiser-Meyer-Olkin (KMO) Test:")
            st.write(f"KMO Test Statistic: {kmo_model}")

            # Scree Plot
            fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=df2.shape[1])
            fa.fit(df2)
            ev, _ = fa.get_eigenvalues()
            plt.figure(figsize=(10, 6))
            plt.scatter(range(1, df2.shape[1] + 1), ev)
            plt.plot(range(1, df2.shape[1] + 1), ev)
            plt.title('Scree Plot')
            plt.xlabel('Factors')
            plt.ylabel('Eigen Value')
            st.pyplot(plt)

            # Heatmap of correlation matrix
            plt.figure(figsize=(20, 10))
            sns.heatmap(df2.corr(), cmap="Reds", annot=True)
            st.pyplot(plt)

            # Variance Inflation Factor (VIF)
            df2_with_const = add_constant(df2)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = df2_with_const.columns
            vif_data["VIF"] = [variance_inflation_factor(df2_with_const.values, i) for i in range(df2_with_const.shape[1])]
            st.write("Variance Inflation Factor (VIF):")
            st.write(vif_data)

            # Factor Analysis
            n_factors = st.number_input("Enter the number of factors:", min_value=1, max_value=df2.shape[1], value=6)
            fa = FactorAnalyzer(n_factors=n_factors, method='principal', rotation="Quartimax")
            fa.fit(df2)
            fa_df = pd.DataFrame(fa.loadings_.round(2), index=df2.columns)
            st.write("Factor Loadings:")
            st.write(fa_df)

            final_factor = fa_df.copy()
            st.write("Final Factors Table:")
            st.write(final_factor)

            st.write("Factor Variance:")
            st.write(pd.DataFrame(fa.get_factor_variance(), index=['Variance', 'Proportional Var', 'Cumulative Var']))

            # User-defined cluster names
            cluster_titles = st.text_input("Enter cluster names (comma-separated):", value="Efficacy,Supply and Samples,Patient Benefits,Cost and Coverage,Approval,MACE")
            cluster_titles = [x.strip() for x in cluster_titles.split(",")]
            factor_scores = fa.transform(df2)
            factor_scores = pd.DataFrame(factor_scores, columns=cluster_titles)
            st.write("Factor Scores:")
            st.write(factor_scores)

            # Random Forest Classifier
            X = factor_scores
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
            rf_classifier = RandomForestClassifier(random_state=42, max_depth=3, max_features=3, n_estimators=500)
            rf_classifier.fit(X_train, y_train)
            y_train_pred = rf_classifier.predict(X_train)
            y_test_pred = rf_classifier.predict(X_test)

            # Metrics
            cf_train = confusion_matrix(y_train, y_train_pred)
            cf_test = confusion_matrix(y_test, y_test_pred)
            TN_train, FN_train, FP_train, TP_train = cf_train.ravel()
            TN_test, FN_test, FP_test, TP_test = cf_test.ravel()

            st.write("Train Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
            st.write(f"Sensitivity: {TP_train / (TP_train + FN_train)}")
            st.write(f"Specificity: {TN_train / (TN_train + FP_train)}")

            st.write("Test Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
            st.write(f"Sensitivity: {TP_test / (TP_test + FN_test)}")
            st.write(f"Specificity: {TN_test / (TN_test + FP_test)}")

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_test_pred))

            # Feature Importance
            imp_df = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})
            imp_df.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("Feature Importance:")
            st.write(imp_df)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            st.pyplot(plt)

if __name__ == "__main__":
    main()

