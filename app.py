#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
import graphviz
import xgboost as xgb

# CSS to inject contained in a string
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
    st.title("Non-Linear Classification Analysis Model_check")

    # Enhanced About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
            ### About this App
            This app was created by Nikhil Saxena for LMRI team use. It allows for comprehensive data analysis, including filtering, factor analysis, and random forest classification. 
                
            **Contact:** 
            - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
                
            **Features:**
            - Upload and filter datasets
            - Perform factor analysis with customizable settings
            - Train and evaluate a Random Forest classifier with optional hyperparameter tuning
            - Visualize results with ROC curves and feature importance
                
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

            # Split data
            X = factor_scores
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
                    
            # Models and Hyperparameters
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'GBM': GradientBoostingClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42)
            }
            
            default_params = {
                'RandomForest': {'max_depth': 3, 'max_features': 3, 'n_estimators': 500},
                'GBM': {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3},
                'XGBoost': {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}
            }
            
            model_selection = st.selectbox("Select Model", list(models.keys()))
            
            # Manual Hyperparameters
            manual_params = {}
            if st.checkbox(f"Manually set {model_selection} parameters"):
                if model_selection == 'RandomForest':
                    manual_params['max_depth'] = st.number_input("max_depth", min_value=1, max_value=20, value=3)
                    manual_params['max_features'] = st.number_input("max_features", min_value=1, max_value=X.shape[1], value=3)
                    manual_params['n_estimators'] = st.number_input("n_estimators", min_value=100, max_value=1000, step=100, value=500)
                elif model_selection == 'GBM':
                    manual_params['learning_rate'] = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
                    manual_params['n_estimators'] = st.number_input("n_estimators", min_value=50, max_value=500, step=50, value=100)
                    manual_params['max_depth'] = st.number_input("max_depth", min_value=1, max_value=20, value=3)
                elif model_selection == 'XGBoost':
                    manual_params['learning_rate'] = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
                    manual_params['n_estimators'] = st.number_input("n_estimators", min_value=50, max_value=500, step=50, value=100)
                    manual_params['max_depth'] = st.number_input("max_depth", min_value=1, max_value=20, value=3)
            
            # GridSearchCV
            grid_search_params = st.checkbox("Use GridSearchCV for hyperparameter tuning")
            if grid_search_params:
                if model_selection == 'RandomForest':
                    param_grid = {
                        'max_depth': [2, 3, 5, 10, 15],
                        'max_features': [1, 2, 3, 5],
                        'n_estimators': [100, 200, 500]
                    }
                elif model_selection == 'GBM':
                    param_grid = {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7]
                    }
                elif model_selection == 'XGBoost':
                    param_grid = {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7]
                    }
            
                st.write(f"Running GridSearchCV for {model_selection}...")
                grid_search = GridSearchCV(estimator=models[model_selection], param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                st.write("Best Hyperparameters found by GridSearchCV:")
                st.write(best_params)
                final_params = best_params
            else:
                final_params = manual_params if manual_params else default_params[model_selection]
            
            st.write("Current Hyperparameters used:")
            st.write(final_params)
            
            model = models[model_selection].set_params(**final_params)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            cf_train = confusion_matrix(y_train, y_train_pred)
            cf_test = confusion_matrix(y_test, y_test_pred)
            TN_train, FP_train, FN_train, TP_train = cf_train.ravel()
            TN_test, FP_test, FN_test, TP_test = cf_test.ravel()
            
            st.write("Train Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
            st.write(f"Sensitivity: {TP_train / (TP_train + FN_train)}")
            #st.write(f"Specificity: {TN_train / (TN_train + FP_train)}")
            
            st.write("Test Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
            st.write(f"Sensitivity: {TP_test / (TP_test + FN_test)}")
            #st.write(f"Specificity: {TN_test / (TN_test + FP_test)}")
            
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_test_pred))
            
            # Feature Importance
            imp_df = pd.DataFrame({"varname": X_train.columns, "Importance": model.feature_importances_ * 100})
            imp_df.sort_values(by="Importance", ascending=False, inplace=True)
            st.write("Feature Importance:")
            st.write(imp_df)
            
            # Button to display ROC Curve
            if st.button("Show ROC Curve"):
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                st.pyplot(plt)
            
            # Button to display Trees for RandomForest, GBM, and XGBoost
            if st.button("Show Trees"):
                if model_selection == 'RandomForest':
                    # Select one of the trees to display
                    estimator = model.estimators_[0]
                    dot_data = StringIO()
                    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                                    special_characters=True, feature_names=X.columns, class_names=model.classes_.astype(str))
                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                    st.graphviz_chart(graph.to_string())
            
                elif model_selection == 'GBM':
                    # Select one of the trees to display
                    estimator = model.estimators_[0, 0]
                    dot_data = StringIO()
                    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                                    special_characters=True, feature_names=X.columns, class_names=model.classes_.astype(str))
                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                    st.graphviz_chart(graph.to_string())
            
                elif model_selection == 'XGBoost':
                    # Select one of the trees to display
                    booster = model.get_booster()
                    trees = booster.get_dump()
                    dot_data = xgb.to_graphviz(booster, num_trees=0)
                    st.graphviz_chart(dot_data.source)

if __name__ == "__main__":
    main()
