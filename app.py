#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, plot_tree
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

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

def main():
    st.title("Non-Linear Classification Analysis Model_V2")

    # Enhanced About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
        ### About this App
        This app was created by Nikhil Saxena for LMRI team use. It allows for comprehensive data analysis, including filtering, factor analysis, and classification using Random Forest, GBM, and XGBoost models.
            
        **Contact:** 
        - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
            
        **Features:**
        - Upload and filter datasets
        - Perform factor analysis with customizable settings
        - Train and evaluate classifiers with optional hyperparameter tuning
        - Visualize results including ROC curves and feature importance
            
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
            X = filtered_df[independent_vars]
            y = filtered_df[outcome_var]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Perform statistical tests and plots
            st.header("Statistical Tests and Plots")

            # Bartlett’s Test of Sphericity
            chi2, p = calculate_bartlett_sphericity(X)
            st.write("Bartlett’s Test of Sphericity:")
            st.write(f"Chi-squared value: {chi2}, p-value: {p}")

            # Kaiser-Meyer-Olkin (KMO) Test
            kmo_values, kmo_model = calculate_kmo(X)
            st.write("Kaiser-Meyer-Olkin (KMO) Test:")
            st.write(f"KMO Test Statistic: {kmo_model}")

            # Scree Plot
            fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=X.shape[1])
            fa.fit(X)
            ev, _ = fa.get_eigenvalues()
            plt.figure(figsize=(10, 6))
            plt.scatter(range(1, X.shape[1] + 1), ev)
            plt.plot(range(1, X.shape[1] + 1), ev)
            plt.title('Scree Plot')
            plt.xlabel('Factors')
            plt.ylabel('Eigen Value')
            plt.grid()
            st.pyplot(plt)

            # Heatmap of correlation matrix
            plt.figure(figsize=(20, 10))
            sns.heatmap(X.corr(), cmap="Reds", annot=True)
            st.pyplot(plt)

            # Variance Inflation Factor (VIF)
            X_with_const = add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_with_const.columns
            vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
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

            n_factors = st.number_input("Enter the number of factors:", min_value=1, max_value=X.shape[1], value=6)
            fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
            fa.fit(X)
            fa_df = pd.DataFrame(fa.loadings_.round(2), index=X.columns)
            st.write("Factor Loadings:")
            st.write(fa_df)

            st.write("Factor Variance:")
            variance_df = pd.DataFrame(fa.get_factor_variance(), index=['Variance', 'Proportional Var', 'Cumulative Var']).T
            st.write(variance_df)

            # Communality
            st.write("Communality:")
            st.write(pd.DataFrame(fa.get_communalities(), index=X.columns, columns=["Communality"]))

            # User-defined cluster names
            cluster_titles = st.text_input("Enter cluster names (comma-separated):", value="Efficacy,Supply and Samples,Patient Benefits,Cost and Coverage,Approval,MACE")
            cluster_titles = [x.strip() for x in cluster_titles.split(",")]
            factor_scores = fa.transform(X)
            factor_scores = pd.DataFrame(factor_scores, columns=cluster_titles)
            st.write("Factor Scores:")
            st.write(factor_scores)

            # Model Selection and Training
            st.header("Model Selection and Training")
            models = ['Random Forest', 'Gradient Boosting Machine', 'XGBoost']
            selected_models = st.multiselect("Select models to train and evaluate:", models, default=models)

            for model_name in selected_models:
                st.subheader(f"{model_name} Classifier")
            
                # Default hyperparameters
                max_depth = 3
                max_features = 3
                n_estimators = 500
                learning_rate = 0.1
            
                # Toggle for GridSearchCV
                if st.checkbox(f"Use GridSearchCV for {model_name} hyperparameter tuning"):
                    if model_name == 'Random Forest':
                        max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10), key='rf_max_depth_range')
                        max_features_range = st.slider("Select max_features range", 1, X.shape[1], (1, 5), key='rf_max_features_range')
                        n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=100, key='rf_n_estimators_range')
            
                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'max_features': list(range(max_features_range[0], max_features_range[1] + 1)),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100))
                        }
            
                        rf = RandomForestClassifier(random_state=42)
                        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        st.write("Best Hyperparameters found by GridSearchCV:")
                        st.write(best_params)
            
                        classifier = RandomForestClassifier(random_state=42, **best_params)
            
                    elif model_name == 'Gradient Boosting Machine':
                        max_depth_range = st.slider("Select max_depth range", 1, 10, (1, 3), key='gbm_max_depth_range')
                        max_features_range = st.slider("Select max_features range", 1, X.shape[1], (1, 5), key='gbm_max_features_range')
                        n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=100, key='gbm_n_estimators_range')
                        learning_rate_range = st.slider("Select learning_rate range", 0.01, 0.5, (0.1, 0.2), step=0.01, key='gbm_learning_rate_range')
            
                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'max_features': list(range(max_features_range[0], max_features_range[1] + 1)),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100)),
                            'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.01)
                        }
            
                        gbm = GradientBoostingClassifier(random_state=42)
                        grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        st.write("Best Hyperparameters found by GridSearchCV:")
                        st.write(best_params)
            
                        classifier = GradientBoostingClassifier(random_state=42, **best_params)
            
                    elif model_name == 'XGBoost':
                        max_depth_range = st.slider("Select max_depth range", 1, 10, (1, 3), key='xgb_max_depth_range')
                        n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=100, key='xgb_n_estimators_range')
                        learning_rate_range = st.slider("Select learning_rate range", 0.01, 0.5, (0.1, 0.2), step=0.01, key='xgb_learning_rate_range')
            
                        param_grid = {
                            'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                            'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100)),
                            'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.01)
                        }
            
                        xgb = XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss')
                        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        st.write("Best Hyperparameters found by GridSearchCV:")
                        st.write(best_params)
            
                        classifier = XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss', **best_params)
                else:
                    if model_name == 'Random Forest':
                        classifier = RandomForestClassifier(random_state=42, max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
                    elif model_name == 'Gradient Boosting Machine':
                        classifier = GradientBoostingClassifier(random_state=42, max_depth=max_depth, max_features=max_features, n_estimators=n_estimators, learning_rate=learning_rate)
                    elif model_name == 'XGBoost':
                        classifier = XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss', max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
            
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
            
                # Evaluate model performance
                st.write(f"Model Performance for {model_name}:")
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), index=classifier.classes_, columns=classifier.classes_))
            
                # Plot ROC curve and calculate AUC
                if hasattr(classifier, "predict_proba"):
                    probas = classifier.predict_proba(X_test)[:, 1]
                else:  # use decision function
                    probas = classifier.decision_function(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas)
                roc_auc = auc(fpr, tpr)
            
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                st.pyplot(plt)
            
                # Plot feature importance (if applicable)
                if model_name in ['Random Forest', 'Gradient Boosting Machine', 'XGBoost']:
                    try:
                        if model_name == 'XGBoost':
                            plot_tree(classifier)
                            st.pyplot(plt)
                        else:
                            importances = classifier.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            plt.figure(figsize=(10, 6))
                            plt.title(f'{model_name} - Feature Importance')
                            plt.bar(range(X.shape[1]), importances[indices], align='center')
                            plt.xticks(range(X.shape[1]), np.array(X.columns)[indices], rotation=90)
                            plt.tight_layout()
                            st.pyplot(plt)
                    except Exception as e:
                        st.write(f"Feature importance plot is not available for {model_name}: {e}")

    else:
        st.info('Awaiting for CSV file to be uploaded.')

if __name__ == '__main__':
    main()
