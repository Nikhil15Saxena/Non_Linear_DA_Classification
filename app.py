#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
import graphviz

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
    
# Streamlit app
def main():
    st.title("Non-Linear Classification Analysis Model_V2")

    # Enhanced About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
            ### About this App
            This app allows for comprehensive data analysis, including filtering, factor analysis, and classification using various models.
                
            ---
            """)

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

            # Models
            models = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting Machine': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42)
            }

            model_names = st.multiselect("Select models to use:", list(models.keys()), default=['Random Forest'])

            for model_name in model_names:
                st.subheader(f"{model_name} Classifier")

                X = factor_scores
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

                # Default hyperparameters
                if model_name == 'Random Forest':
                    max_depth = 3
                    max_features = 3
                    n_estimators = 500
                elif model_name == 'Gradient Boosting Machine':
                    max_depth = 3
                    learning_rate = 0.1
                    n_estimators = 100
                elif model_name == 'XGBoost':
                    max_depth = 3
                    learning_rate = 0.1
                    n_estimators = 100

                # Toggle for GridSearchCV
                if st.checkbox(f"Use GridSearchCV for {model_name} hyperparameter tuning"):
                    if model_name == 'Random Forest':
                        max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10))
                        max_features_range = st.slider("Select max_features range", 1, X.shape[1], (1, 5))
                        # Adjust the step parameter for n_estimators slider to ensure a difference of 500
                        n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=500)

                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'max_features': list(range(max_features_range[0], max_features_range[1] + 1)),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 500))
                        }
                    elif model_name == 'Gradient Boosting Machine':
                        max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10))
                        learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.1, 0.5), step=0.1)
                        n_estimators_range = st.slider("Select n_estimators range", 50, 1000, (100, 500), step=100)

                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.1).tolist(),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100))
                        }
                    elif model_name == 'XGBoost':
                        max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10))
                        learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.1, 0.5), step=0.1)
                        n_estimators_range = st.slider("Select n_estimators range", 50, 1000, (100, 500), step=100)

                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.1).tolist(),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100))
                        }

                    clf = GridSearchCV(models[model_name], param_grid, cv=5, n_jobs=-1)
                    clf.fit(X_train, y_train)

                    st.write(f"Best Parameters for {model_name}:")
                    st.write(clf.best_params_)
                    st.write(f"Best Score for {model_name}:")
                    st.write(clf.best_score_)

                    # Update default hyperparameters
                    max_depth = clf.best_params_['max_depth']
                    if model_name == 'Random Forest':
                        max_features = clf.best_params_['max_features']
                    elif model_name in ['Gradient Boosting Machine', 'XGBoost']:
                        learning_rate = clf.best_params_['learning_rate']
                    n_estimators = clf.best_params_['n_estimators']

                # Manual hyperparameter input
                if st.checkbox(f"Manually input hyperparameters for {model_name}"):
                    if model_name == 'Random Forest':
                        max_depth = st.number_input("Enter max_depth:", 1, 20, 3, 1)
                        max_features = st.number_input("Enter max_features:", 1, X.shape[1], 3, 1)
                        n_estimators = st.number_input("Enter n_estimators:", 100, 1000, 500, 100)
                    elif model_name == 'Gradient Boosting Machine' or model_name == 'XGBoost':
                        max_depth = st.number_input("Enter max_depth:", 1, 20, 3, 1)
                        learning_rate = st.number_input("Enter learning_rate:", 0.01, 1.0, 0.1, 0.1)
                        n_estimators = st.number_input("Enter n_estimators:", 50, 1000, 100, 100)

                # Model training and evaluation
                model = models[model_name]
                if model_name == 'Random Forest':
                    model.set_params(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
                elif model_name == 'Gradient Boosting Machine':
                    model.set_params(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
                elif model_name == 'XGBoost':
                    model.set_params(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='binary:logistic')

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy for {model_name}:")
                st.write(acc)

                st.subheader(f"Confusion Matrix for {model_name}:")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

                st.subheader(f"Classification Report for {model_name}:")
                report = classification_report(y_test, y_pred)
                st.write(report)

                # Visualization of decision trees
                if st.button(f"Show {model_name} Trees"):
                    for estimator in model.estimators_:
                        dot_data = StringIO()
                        export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=X.columns)
                        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                        st.graphviz_chart(graph.create_png())

# Run the Streamlit app
if __name__ == "__main__":
    main()
