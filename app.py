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
from xgboost import XGBClassifier, plot_tree

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
    st.title("Non-Linear Classification Analysis Model_V2")

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
            - Train and evaluate a Random Forest, GBM, and XGBoost classifier with optional hyperparameter tuning
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

            # Random Forest Classifier
            X = factor_scores
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

            model_name = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting Machine", "XGBoost"])

            # Default hyperparameters
            max_depth = 3
            max_features = 3
            n_estimators = 500
            learning_rate = 0.1

            # Checkbox for hyperparameter tuning
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
                    max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10), key='gbm_max_depth_range')
                    n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=100, key='gbm_n_estimators_range')
                    learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.01, 0.1), step=0.01, key='gbm_learning_rate_range')

                    param_grid = {
                        'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                        'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100)),
                        'learning_rate': [round(x, 2) for x in np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.01)]
                    }

                    gbm = GradientBoostingClassifier(random_state=42)
                    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    st.write("Best Hyperparameters found by GridSearchCV:")
                    st.write(best_params)

                    classifier = GradientBoostingClassifier(random_state=42, **best_params)

                elif model_name == 'XGBoost':
                    max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10), key='xgb_max_depth_range')
                    n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=100, key='xgb_n_estimators_range')
                    learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.01, 0.1), step=0.01, key='xgb_learning_rate_range')

                    param_grid = {
                        'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                        'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100)),
                        'learning_rate': [round(x, 2) for x in np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.01)]
                    }

                    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
                    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    st.write("Best Hyperparameters found by GridSearchCV:")
                    st.write(best_params)

                    classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', **best_params)

            else:
                # Checkbox to manually set hyperparameters
                if st.checkbox(f"Manually set {model_name} parameters"):
                    if model_name == 'Random Forest':
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=3, key='rf_max_depth')
                        max_features = st.number_input("max_features", min_value=1, max_value=X.shape[1], value=3, key='rf_max_features')
                        n_estimators = st.number_input("n_estimators", min_value=100, max_value=1000, step=100, value=500, key='rf_n_estimators')

                        classifier = RandomForestClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            max_features=max_features,
                            n_estimators=n_estimators
                        )

                    elif model_name == 'Gradient Boosting Machine':
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=3, key='gbm_max_depth')
                        n_estimators = st.number_input("n_estimators", min_value=100, max_value=1000, step=100, value=500, key='gbm_n_estimators')
                        learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1, key='gbm_learning_rate')

                        classifier = GradientBoostingClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate
                        )

                    elif model_name == 'XGBoost':
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=3, key='xgb_max_depth')
                        n_estimators = st.number_input("n_estimators", min_value=100, max_value=1000, step=100, value=500, key='xgb_n_estimators')
                        learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1, key='xgb_learning_rate')

                        classifier = XGBClassifier(
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='mlogloss',
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate
                        )
                else:
                    if model_name == 'Random Forest':
                        classifier = RandomForestClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            max_features=max_features,
                            n_estimators=n_estimators
                        )
                    elif model_name == 'Gradient Boosting Machine':
                        classifier = GradientBoostingClassifier(
                            random_state=42,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate
                        )
                    elif model_name == 'XGBoost':
                        classifier = XGBClassifier(
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='mlogloss',
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate
                        )

            classifier.fit(X_train, y_train)
            y_train_pred = classifier.predict(X_train)
            y_test_pred = classifier.predict(X_test)

            # Metrics
            cf_train = confusion_matrix(y_train, y_train_pred)
            cf_test = confusion_matrix(y_test, y_test_pred)
            TN_train, FP_train, FN_train, TP_train = cf_train.ravel()
            TN_test, FP_test, FN_test, TP_test = cf_test.ravel()

            sensitivity_train = TP_train / (TP_train + FN_train) if (TP_train + FN_train) != 0 else float('nan')
            specificity_train = TN_train / (TN_train + FP_train) if (TN_train + FP_train) != 0 else float('nan')
            sensitivity_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) != 0 else float('nan')
            specificity_test = TN_test / (TN_test + FP_test) if (TN_test + FP_test) != 0 else float('nan')

            st.write("Train Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
            st.write(f"Sensitivity: {sensitivity_train}")
            st.write(f"Specificity: {specificity_train}")

            st.write("Test Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
            st.write(f"Sensitivity: {sensitivity_test}")
            st.write(f"Specificity: {specificity_test}")

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_test_pred))

            # Feature Importance
            imp_df = pd.DataFrame({"varname": X_train.columns, "Imp": classifier.feature_importances_ * 100})
            imp_df.sort_values(by="Imp", ascending=False, inplace=True)
            st.write(f"Feature Importance for {model_name}:")
            st.write(imp_df)

            # Button to display ROC Curve
            if st.button("Show ROC Curve"):
                fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Receiver Operating Characteristic for {model_name}')
                plt.legend(loc="lower right")
                st.pyplot(plt)

            # Visualize Trees for Random Forest, GBM, and XGBoost
            if model_name == 'Random Forest':
                if st.button("Show Trees for Random Forest"):
                    # Visualize a few trees from the Random Forest
                    for i in range(min(3, classifier.n_estimators)):
                        tree = classifier.estimators_[i]
                        dot_data = StringIO()
                        export_graphviz(tree, out_file=dot_data, filled=True, rounded=True,
                                        special_characters=True, feature_names=X_train.columns, class_names=True)
                        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                        st.graphviz_chart(graph.to_string())

            elif model_name == 'Gradient Boosting Machine':
                if st.button("Show Trees for GBM"):
                    # Visualize a few trees from the GBM
                    for i in range(min(3, classifier.n_estimators)):
                        tree = classifier.estimators_[i, 0]
                        dot_data = StringIO()
                        export_graphviz(tree, out_file=dot_data, filled=True, rounded=True,
                                        special_characters=True, feature_names=X_train.columns, class_names=True)
                        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                        st.graphviz_chart(graph.to_string())

            elif model_name == 'XGBoost':
                if st.button("Show Trees for XGBoost"):
                    # Visualize a few trees from the XGBoost model
                    for i in range(min(3, classifier.n_estimators)):
                        plt.figure(figsize=(10, 6))
                        plot_tree(classifier, num_trees=i)
                        plt.title(f"Tree number: {i}")
                        st.pyplot(plt)

if __name__ == "__main__":
    main()

