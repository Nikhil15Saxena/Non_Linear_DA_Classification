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
    st.title("Non-Linear Classification Analysis Model_V2")

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

            # Classifier Selection
            st.header("Classifier Selection")
            classifier = st.selectbox("Select Classifier:", ["Random Forest", "Gradient Boosting Machine", "XGBoost"])

            # Model Training and Evaluation
            if st.button("Train and Evaluate"):
                X = factor_scores
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

                if classifier == "Random Forest":
                    rf_classifier = RandomForestClassifier(random_state=42)

                elif classifier == "Gradient Boosting Machine":
                    rf_classifier = GradientBoostingClassifier(random_state=42)

                elif classifier == "XGBoost":
                    rf_classifier = XGBClassifier(random_state=42)

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
                if classifier == 'XGBoost':
                    imp_df = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})
                else:
                    imp_df = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})

                imp_df.sort_values(by="Imp", ascending=False, inplace=True)
                st.write(f"{classifier} Feature Importance:")
                st.write(imp_df)

                # Button to display Tree
                if st.button(f"Show {classifier} Tree"):
                    if classifier == 'XGBoost':
                        estimator = rf_classifier.get_booster()
                        dot_data = StringIO()
                        xgb_plot_tree(estimator, num_trees=0, ax=None, rankdir='UT')
                        st.pyplot(plt)
                    else:
                        estimator = rf_classifier.estimators_[0]
                        dot_data = StringIO()
                        export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                                        special_characters=True, feature_names=X.columns, class_names=rf_classifier.classes_.astype(str))
                        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                        st.graphviz_chart(graph.to_string())

if __name__ == "__main__":
    main()
