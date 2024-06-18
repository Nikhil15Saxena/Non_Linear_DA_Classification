#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
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
    st.title("Non-Linear Classification Analysis Model")

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

                        rf = RandomForestClassifier(random_state=42)
                        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        st.write("Best Hyperparameters found by GridSearchCV:")
                        st.write(best_params)

                        rf_classifier = RandomForestClassifier(random_state=42, **best_params)

                    elif model_name == 'Gradient Boosting Machine':
                        max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10))
                        learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.1, 0.5), step=0.1)
                        n_estimators_range = st.slider("Select n_estimators range", 50, 500, (50, 100), step=50)

                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1], 0.1),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 50))
                        }

                        gbm = GradientBoostingClassifier(random_state=42)
                        grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        st.write("Best Hyperparameters found by GridSearchCV:")
                        st.write(best_params)

                        rf_classifier = GradientBoostingClassifier(random_state=42, **best_params)

                    elif model_name == 'XGBoost':
                        max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10))
                        learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.1, 0.5), step=0.1)
                        n_estimators_range = st.slider("Select n_estimators range", 50, 500, (50, 100), step=50)

                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1], 0.1),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 50))
                        }

                        xgb = XGBClassifier(random_state=42)
                        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        st.write("Best Hyperparameters found by GridSearchCV:")
                        st.write(best_params)

                        rf_classifier = XGBClassifier(random_state=42, **best_params)

                else:
                    # Ask if the user wants to input hyperparameters manually
                    if model_name == 'Random Forest':
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=3)
                        max_features = st.number_input("max_features", min_value=1, max_value=X.shape[1], value=3)
                        n_estimators = st.number_input("n_estimators", min_value=100, max_value=1000, step=100, value=500)
                        rf_classifier = RandomForestClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            max_features=max_features,
                            n_estimators=n_estimators
                        )

                    elif model_name == 'Gradient Boosting Machine':
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=3)
                        learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.1, value=0.1)
                        n_estimators = st.number_input("n_estimators", min_value=50, max_value=500, step=50, value=100)
                        rf_classifier = GradientBoostingClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators
                        )

                    elif model_name == 'XGBoost':
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=3)
                        learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.1, value=0.1)
                        n_estimators = st.number_input("n_estimators", min_value=50, max_value=500, step=50, value=100)
                        rf_classifier = XGBClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators
                        )

                # Display current hyperparameters
                st.write("Current Hyperparameters used:")
                if model_name == 'Random Forest' or model_name == 'Gradient Boosting Machine' or model_name == 'XGBoost':
                    if model_name == 'Random Forest':
                        st.write(f"max_depth: {max_depth}")
                        st.write(f"max_features: {max_features}")
                        st.write(f"n_estimators: {n_estimators}")
                    elif model_name == 'Gradient Boosting Machine':
                        st.write(f"max_depth: {max_depth}")
                        st.write(f"learning_rate: {learning_rate}")
                        st.write(f"n_estimators: {n_estimators}")
                    elif model_name == 'XGBoost':
                        st.write(f"max_depth: {max_depth}")
                        st.write(f"learning_rate: {learning_rate}")
                        st.write(f"n_estimators: {n_estimators}")

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
                if model_name == 'XGBoost':
                    imp_df = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})
                else:
                    imp_df = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})
                
                imp_df.sort_values(by="Imp", ascending=False, inplace=True)
                st.write(f"{model_name} Feature Importance:")
                st.write(imp_df)

                # Button to display ROC Curve
                if st.button(f"Show {model_name} ROC Curve"):
                    fpr, tpr, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(10, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)

                # Button to display Tree
                if st.button(f"Show {model_name} Tree"):
                    if model_name == 'XGBoost':
                        estimator = rf_classifier.get_booster()
                        dot_data = StringIO()
                        xgb.plot_tree(estimator, num_trees=0, ax=None, rankdir='UT', **xgb.plot_tree_args)
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
