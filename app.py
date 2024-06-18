#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, plot_tree as xgb_plot_tree
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
            This app was created by Nikhil Saxena for LMRI team use. It allows for comprehensive data analysis, including filtering, factor analysis, and classification using Random Forest, Gradient Boosting, and XGBoost.
                
            **Contact:** 
            - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
                
            **Features:**
            - Upload and filter datasets
            - Perform factor analysis with customizable settings
            - Train and evaluate classifiers (Random Forest, Gradient Boosting, XGBoost) with optional hyperparameter tuning
            - Visualize results with ROC curves, feature importance, and decision trees
                
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

            # Model selection and hyperparameter tuning
            models = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False)
            }

            st.header("Model Selection and Hyperparameter Tuning")

            selected_models = st.multiselect("Select models to train and evaluate:", list(models.keys()), default=['Random Forest'])

            for model_name in selected_models:
                st.subheader(model_name)

                # Hyperparameters selection
                use_grid_search = st.checkbox(f"Use GridSearchCV for {model_name} hyperparameter tuning")
                if use_grid_search:
                    if model_name == 'Random Forest':
                        param_grid = {
                            'max_depth': list(range(1, 21)),
                            'max_features': list(range(1, X.shape[1] + 1)),
                            'n_estimators': [100, 500, 1000]
                        }
                    elif model_name == 'Gradient Boosting':
                        param_grid = {
                            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                            'n_estimators': [50, 100, 200, 500],
                            'max_depth': list(range(1, 11)),
                            'max_features': list(range(1, X.shape[1] + 1))
                        }
                    elif model_name == 'XGBoost':
                        param_grid = {
                            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                            'n_estimators': [50, 100, 200, 500],
                            'max_depth': list(range(1, 11)),
                            'colsample_bytree': [0.5, 0.7, 0.9, 1.0]
                        }

                    grid_search = GridSearchCV(models[model_name], param_grid, cv=5, n_jobs=-1, scoring='accuracy')
                    grid_search.fit(X_train, y_train)

                    st.write(f"Best Parameters for {model_name}:")
                    st.write(grid_search.best_params_)
                    classifier = grid_search.best_estimator_
                else:
                    classifier = models[model_name]
                    classifier.fit(X_train, y_train)

                # Predictions
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")

                # Confusion Matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

                # Classification Report
                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred))

                # ROC Curve and AUC
                if hasattr(classifier, "predict_proba"):
                    st.write("ROC Curve and AUC:")
                    y_proba = classifier.predict_proba(X_test)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)

                # Feature Importance (if applicable)
                if hasattr(classifier, "feature_importances_"):
                    st.write("Feature Importance:")
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': classifier.feature_importances_
                    })
                    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                    st.write(feature_importance)

                # Decision Tree Visualization (for RandomForest and XGBoost)
                if model_name == 'Random Forest' or model_name == 'XGBoost':
                    st.write(f"{model_name} Trees Visualization:")
                    st.subheader("Tree Options")
                    if st.checkbox("Display a tree?"):
                        tree_num = st.number_input("Tree ID", 0, len(classifier.estimators_), 0)
                        if model_name == 'Random Forest':
                            if st.button('Generate', key=None):
                                dot_data = StringIO()
                                export_graphviz(classifier.estimators_[tree_num], out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=X.columns, class_names=np.unique(y).astype('str'))
                                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                                st.image(graph.create_png())
                            else:
                                st.write('Please press the Generate button to create the graph for a given Tree.')
                        elif model_name == 'XGBoost':
                            if st.button('Generate', key=None):
                                xgb.plot_tree(classifier, num_trees=tree_num)
                                plt.rcParams['figure.figsize'] = [50, 10]
                                st.pyplot()
                            else:
                                st.write('Please press the Generate button to create the graph for a given Tree.')

# Run the Streamlit app
if __name__ == "__main__":
    main()
