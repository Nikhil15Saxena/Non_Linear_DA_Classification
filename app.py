#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from xgboost import XGBClassifier, plot_tree as xgb_plot_tree
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO

# CSS to hide Streamlit menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 16px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title("Non-Linear Classification Analysis Model")

    # About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
            ### About this App
            This app was created by Nikhil Saxena for LMRI team use. It allows for comprehensive data analysis, including filtering, factor analysis, and classification using Random Forest, Gradient Boosting, and XGBoost.
                
            **Contact:** 
            - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
                
            **Features:**
            - Upload and filter datasets
            - Perform factor analysis with customizable settings
            - Train and evaluate classifiers with optional hyperparameter tuning
            - Visualize decision trees and feature importance
                
            ---
            """, unsafe_allow_html=True)
    
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Show original data shape
        st.write("Original Data Shape:")
        st.write(df.shape)

        # Multiple filtering options
        st.header("Filter Data")
        filter_columns = st.multiselect("Select columns to filter:", df.columns)
        filters = {}
        for col in filter_columns:
            filters[col] = st.text_input(f"Enter value to filter in '{col}' column:")

        filtered_df = df.copy()
        for col, val in filters.items():
            if val:
                filtered_df = filtered_df[filtered_df[col] == val]

        st.write("Filtered Data:")
        st.write(filtered_df)

        # Show filtered data shape
        st.write("Filtered Data Shape:")
        st.write(filtered_df.shape)

        # Allow user to select the outcome and independent variables
        outcome_var = st.selectbox("Select the outcome variable:", filtered_df.columns)
        independent_vars = st.multiselect("Select independent variables:", filtered_df.columns)
        
        if outcome_var and independent_vars:
            df2 = filtered_df[independent_vars]
            y = filtered_df[outcome_var]

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
            plt.grid()
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
            st.subheader("Factor Analysis")

            if st.checkbox("Click to select method and rotation"):
                rotation_options = ["None", "Varimax", "Promax", "Quartimax", "Oblimin"]
                rotation = st.selectbox("Select rotation:", rotation_options)
                method_options = ["Principal", "Minres", "ML", "GLS", "OLS"]
                method = st.selectbox("Select method:", method_options)
                if rotation == "None":
                    rotation = None
                if method == "Principal":
                    method = "principal"
            else:
                rotation = "varimax"
                method = "principal"

            st.write(f"Method: {method}, Rotation: {rotation}")

            n_factors = st.number_input("Enter the number of factors:", min_value=1, max_value=df2.shape[1], value=6)
            fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
            fa.fit(df2)
            fa_df = pd.DataFrame(fa.loadings_.round(2), index=df2.columns)
            st.write("Factor Loadings:")
            st.write(fa_df)

            st.write("Factor Variance:")
            variance_df = pd.DataFrame(fa.get_factor_variance(), index=['Variance', 'Proportional Var', 'Cumulative Var']).T
            st.write(variance_df)

            # Communality
            st.write("Communality:")
            st.write(pd.DataFrame(fa.get_communalities(), index=df2.columns, columns=["Communality"]))

            # User-defined cluster names
            cluster_titles = st.text_input("Enter cluster names (comma-separated):", value="Efficacy,Supply and Samples,Patient Benefits,Cost and Coverage,Approval,MACE")
            cluster_titles = [x.strip() for x in cluster_titles.split(",")]
            factor_scores = fa.transform(df2)
            factor_scores = pd.DataFrame(factor_scores, columns=cluster_titles)
            st.write("Factor Scores:")
            st.write(factor_scores)

            st.header("Model Training and Evaluation")

            # Random Forest
            st.subheader("Random Forest")
            rf_classifier = RandomForestClassifier(random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(factor_scores, y, train_size=0.7, random_state=42)
            rf_classifier.fit(X_train, y_train)
            y_train_pred_rf = rf_classifier.predict(X_train)
            y_test_pred_rf = rf_classifier.predict(X_test)

            st.write("Random Forest - Classification Report (Test Data):")
            st.text(classification_report(y_test, y_test_pred_rf))

            # Feature Importance
            imp_df_rf = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})
            imp_df_rf.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("Random Forest - Feature Importance:")
            st.write(imp_df_rf)

            # Gradient Boosting Machine
            st.subheader("Gradient Boosting Machine")
            gbm_classifier = GradientBoostingClassifier(random_state=42)
            gbm_classifier.fit(X_train, y_train)
            y_train_pred_gbm = gbm_classifier.predict(X_train)
            y_test_pred_gbm = gbm_classifier.predict(X_test)

            st.write("Gradient Boosting Machine - Classification Report (Test Data):")
            st.text(classification_report(y_test, y_test_pred_gbm))

            # Feature Importance
            imp_df_gbm = pd.DataFrame({"varname": X_train.columns, "Imp": gbm_classifier.feature_importances_ * 100})
            imp_df_gbm.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("Gradient Boosting Machine - Feature Importance:")
            st.write(imp_df_gbm)

            # XGBoost
            st.subheader("XGBoost")
            xgb_classifier = XGBClassifier(random_state=42)
            xgb_classifier.fit(X_train, y_train)
            y_train_pred_xgb = xgb_classifier.predict(X_train)
            y_test_pred_xgb = xgb_classifier.predict(X_test)

            st.write("XGBoost - Classification Report (Test Data):")
            st.text(classification_report(y_test, y_test_pred_xgb))

            # Feature Importance
            imp_df_xgb = pd.DataFrame({"varname": X_train.columns, "Imp": xgb_classifier.feature_importances_ * 100})
            imp_df_xgb.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("XGBoost - Feature Importance:")
            st.write(imp_df_xgb)

            # Button to display Random Forest Trees
            if st.button("Show Random Forest Trees"):
                estimator = rf_classifier.estimators_[0]
                dot_data = StringIO()
                export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                                special_characters=True, feature_names=X_train.columns, class_names=rf_classifier.classes_.astype(str))
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                st.graphviz_chart(graph.to_string())

            # Button to display XGBoost Trees
            if st.button("Show XGBoost Trees"):
                estimator = xgb_classifier.get_booster()
                dot_data = StringIO()
                xgb_plot_tree(estimator, num_trees=0, ax=None, rankdir='UT')
                st.pyplot(plt)

if __name__ == "__main__":
    main()
