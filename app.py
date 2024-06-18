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
            - Train and evaluate Random Forest, GBM, and XGBoost classifiers with optional hyperparameter tuning
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

            # Classification Models
            X = factor_scores
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

            # Random Forest Classifier
            st.subheader("Random Forest Classifier")

            # Default hyperparameters
            max_depth = 3
            max_features = 3
            n_estimators = 500

            # Toggle for GridSearchCV
            if st.checkbox("Use GridSearchCV for Random Forest hyperparameter tuning"):
                max_depth_range = st.slider("Select max_depth range", 1, 20, (1, 10))
                max_features_range = st.slider("Select max_features range", 1, X.shape[1], (1, 5))
                n_estimators_range = st.slider("Select n_estimators range", 100, 1000, (100, 500), step=100)

                param_grid = {
                    'max_depth': list(range(max_depth_range[0], max_depth_range[1] + 1)),
                    'max_features': list(range(max_features_range[0], max_features_range[1] + 1)),
                    'n_estimators': list(range(n_estimators_range[0], n_estimators_range[1] + 1, 100))
                }

                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_params_rf = grid_search.best_params_
                st.write("Best Hyperparameters found by GridSearchCV for Random Forest:")
                st.write(best_params_rf)

                rf_classifier = RandomForestClassifier(random_state=42, **best_params_rf)
            else:
                rf_classifier = RandomForestClassifier(
                    random_state=42,
                    max_depth=max_depth,
                    max_features=max_features,
                    n_estimators=n_estimators
                )

            # Display current hyperparameters
            st.write("Current Hyperparameters used for Random Forest:")
            st.write(f"max_depth: {max_depth}")
            st.write(f"max_features: {max_features}")
            st.write(f"n_estimators: {n_estimators}")

            rf_classifier.fit(X_train, y_train)
            y_train_pred_rf = rf_classifier.predict(X_train)
            y_test_pred_rf = rf_classifier.predict(X_test)

            # Metrics for Random Forest
            cf_train_rf = confusion_matrix(y_train, y_train_pred_rf)
            cf_test_rf = confusion_matrix(y_test, y_test_pred_rf)
            TN_train_rf, FN_train_rf, FP_train_rf, TP_train_rf = cf_train_rf.ravel()
            TN_test_rf, FN_test_rf, FP_test_rf, TP_test_rf = cf_test_rf.ravel()

            st.write("Train Data Metrics for Random Forest:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred_rf)}")
            st.write(f"Sensitivity: {TP_train_rf / (TP_train_rf + FN_train_rf)}")
            st.write(f"Specificity: {TN_train_rf / (TN_train_rf + FP_train_rf)}")

            st.write("Test Data Metrics for Random Forest:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred_rf)}")
            st.write(f"Sensitivity: {TP_test_rf / (TP_test_rf + FN_test_rf)}")
            st.write(f"Specificity: {TN_test_rf / (TN_test_rf + FP_test_rf)}")

            st.write("Classification Report for Random Forest:")
            st.text(classification_report(y_test, y_test_pred_rf))

            # Feature Importance for Random Forest
            imp_df_rf = pd.DataFrame({"varname": X_train.columns, "Imp": rf_classifier.feature_importances_ * 100})
            imp_df_rf.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("Feature Importance for Random Forest:")
            st.write(imp_df_rf)

            # GBM Classifier
            st.subheader("Gradient Boosting Classifier (GBM)")

            # Default hyperparameters
            learning_rate_gbm = 0.1
            n_estimators_gbm = 100
            max_depth_gbm = 3
            subsample_gbm = 1.0

            # Toggle for GridSearchCV
            if st.checkbox("Use GridSearchCV for GBM hyperparameter tuning"):
                learning_rate_range = st.slider("Select learning_rate range", 0.01, 1.0, (0.1, 0.5), step=0.01)
                n_estimators_range_gbm = st.slider("Select n_estimators range for GBM", 50, 500, (100, 300), step=50)
                max_depth_range_gbm = st.slider("Select max_depth range for GBM", 1, 20, (1, 5))
                subsample_range_gbm = st.slider("Select subsample range for GBM", 0.1, 1.0, (0.5, 1.0), step=0.1)

                param_grid_gbm = {
                    'learning_rate': np.arange(learning_rate_range[0], learning_rate_range[1] + 0.01, 0.01),
                    'n_estimators': list(range(n_estimators_range_gbm[0], n_estimators_range_gbm[1] + 1, 50)),
                    'max_depth': list(range(max_depth_range_gbm[0], max_depth_range_gbm[1] + 1)),
                    'subsample': np.arange(subsample_range_gbm[0], subsample_range_gbm[1] + 0.01, 0.1)
                }

                gbm = GradientBoostingClassifier(random_state=42)
                grid_search_gbm = GridSearchCV(estimator=gbm, param_grid=param_grid_gbm, cv=5, n_jobs=1, verbose=2)
                grid_search_gbm.fit(X_train, y_train)
                best_params_gbm = grid_search_gbm.best_params_
                st.write("Best Hyperparameters found by GridSearchCV for GBM:")
                st.write(best_params_gbm)

                gbm_classifier = GradientBoostingClassifier(random_state=42, **best_params_gbm)
            else:
                gbm_classifier = GradientBoostingClassifier(
                    random_state=42,
                    learning_rate=learning_rate_gbm,
                    n_estimators=n_estimators_gbm,
                    max_depth=max_depth_gbm,
                    subsample=subsample_gbm
                )

            # Display current hyperparameters
            st.write("Current Hyperparameters used for GBM:")
            st.write(f"learning_rate: {learning_rate_gbm}")
            st.write(f"n_estimators: {n_estimators_gbm}")
            st.write(f"max_depth: {max_depth_gbm}")
            st.write(f"subsample: {subsample_gbm}")

            gbm_classifier.fit(X_train, y_train)
            y_train_pred_gbm = gbm_classifier.predict(X_train)
            y_test_pred_gbm = gbm_classifier.predict(X_test)

            # Metrics for GBM
            cf_train_gbm = confusion_matrix(y_train, y_train_pred_gbm)
            cf_test_gbm = confusion_matrix(y_test, y_test_pred_gbm)
            TN_train_gbm, FN_train_gbm, FP_train_gbm, TP_train_gbm = cf_train_gbm.ravel()
            TN_test_gbm, FN_test_gbm, FP_test_gbm, TP_test_gbm = cf_test_gbm.ravel()

            st.write("Train Data Metrics for GBM:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred_gbm)}")
            st.write(f"Sensitivity: {TP_train_gbm / (TP_train_gbm + FN_train_gbm)}")
            st.write(f"Specificity: {TN_train_gbm / (TN_train_gbm + FP_train_gbm)}")

            st.write("Test Data Metrics for GBM:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred_gbm)}")
            st.write(f"Sensitivity: {TP_test_gbm / (TP_test_gbm + FN_test_gbm)}")
            st.write(f"Specificity: {TN_test_gbm / (TN_test_gbm + FP_test_gbm)}")

            st.write("Classification Report for GBM:")
            st.text(classification_report(y_test, y_test_pred_gbm))

            # Feature Importance for GBM
            imp_df_gbm = pd.DataFrame({"varname": X_train.columns, "Imp": gbm_classifier.feature_importances_ * 100})
            imp_df_gbm.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("Feature Importance for GBM:")
            st.write(imp_df_gbm)

            # XGBoost Classifier
            st.subheader("XGBoost Classifier")

            # Default hyperparameters
            learning_rate_xgb = 0.1
            n_estimators_xgb = 100
            max_depth_xgb = 3
            subsample_xgb = 1.0

            # Toggle for GridSearchCV
            if st.checkbox("Use GridSearchCV for XGBoost hyperparameter tuning"):
                learning_rate_range_xgb = st.slider("Select learning_rate range for XGBoost", 0.01, 1.0, (0.1, 0.5), step=0.01)
                n_estimators_range_xgb = st.slider("Select n_estimators range for XGBoost", 50, 500, (100, 300), step=50)
                max_depth_range_xgb = st.slider("Select max_depth range for XGBoost", 1, 20, (1, 5))
                subsample_range_xgb = st.slider("Select subsample range for XGBoost", 0.1, 1.0, (0.5, 1.0), step=0.1)

                param_grid_xgb = {
                    'learning_rate': np.arange(learning_rate_range_xgb[0], learning_rate_range_xgb[1] + 0.01, 0.01),
                    'n_estimators': list(range(n_estimators_range_xgb[0], n_estimators_range_xgb[1] + 1, 50)),
                    'max_depth': list(range(max_depth_range_xgb[0], max_depth_range_xgb[1] + 1)),
                    'subsample': np.arange(subsample_range_xgb[0], subsample_range_xgb[1] + 0.01, 0.1)
                }

                xgb = XGBClassifier(random_state=42)
                grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=1, verbose=2)
                grid_search_xgb.fit(X_train, y_train)
                best_params_xgb = grid_search_xgb.best_params_
                st.write("Best Hyperparameters found by GridSearchCV for XGBoost:")
                st.write(best_params_xgb)

                xgb_classifier = XGBClassifier(random_state=42, **best_params_xgb)
            else:
                xgb_classifier = XGBClassifier(
                    random_state=42,
                    learning_rate=learning_rate_xgb,
                    n_estimators=n_estimators_xgb,
                    max_depth=max_depth_xgb,
                    subsample=subsample_xgb
                )

            # Display current hyperparameters
            st.write("Current Hyperparameters used for XGBoost:")
            st.write(f"learning_rate: {learning_rate_xgb}")
            st.write(f"n_estimators: {n_estimators_xgb}")
            st.write(f"max_depth: {max_depth_xgb}")
            st.write(f"subsample: {subsample_xgb}")

            xgb_classifier.fit(X_train, y_train)
            y_train_pred_xgb = xgb_classifier.predict(X_train)
            y_test_pred_xgb = xgb_classifier.predict(X_test)

            # Metrics for XGBoost
            cf_train_xgb = confusion_matrix(y_train, y_train_pred_xgb)
            cf_test_xgb = confusion_matrix(y_test, y_test_pred_xgb)
            TN_train_xgb, FN_train_xgb, FP_train_xgb, TP_train_xgb = cf_train_xgb.ravel()
            TN_test_xgb, FN_test_xgb, FP_test_xgb, TP_test_xgb = cf_test_xgb.ravel()

            st.write("Train Data Metrics for XGBoost:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred_xgb)}")
            st.write(f"Sensitivity: {TP_train_xgb / (TP_train_xgb + FN_train_xgb)}")
            st.write(f"Specificity: {TN_train_xgb / (TN_train_xgb + FP_train_xgb)}")

            st.write("Test Data Metrics for XGBoost:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred_xgb)}")
            st.write(f"Sensitivity: {TP_test_xgb / (TP_test_xgb + FN_test_xgb)}")
            st.write(f"Specificity: {TN_test_xgb / (TN_test_xgb + FP_test_xgb)}")

            st.write("Classification Report for XGBoost:")
            st.text(classification_report(y_test, y_test_pred_xgb))

            # Feature Importance for XGBoost
            imp_df_xgb = pd.DataFrame({"varname": X_train.columns, "Imp": xgb_classifier.feature_importances_ * 100})
            imp_df_xgb.sort_values(by="Imp", ascending=False, inplace=True)
            st.write("Feature Importance for XGBoost:")
            st.write(imp_df_xgb)

            # ROC Curve and AUC
            st.header("ROC Curve and AUC")

            # Random Forest ROC
            fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
            roc_auc_rf = auc(fpr_rf, tpr_rf)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            # GBM ROC
            fpr_gbm, tpr_gbm, _ = roc_curve(y_test, gbm_classifier.predict_proba(X_test)[:, 1])
            roc_auc_gbm = auc(fpr_gbm, tpr_gbm)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_gbm, tpr_gbm, color='green', lw=2, label=f'GBM (AUC = {roc_auc_gbm:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve for GBM')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            # XGBoost ROC
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_classifier.predict_proba(X_test)[:, 1])
            roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_xgb, tpr_xgb, color='purple', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            # Show all model results
            st.header("All Model Results")

            st.subheader("Random Forest Results:")
            st.write("Train Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred_rf)}")
            st.write(f"Sensitivity: {TP_train_rf / (TP_train_rf + FN_train_rf)}")
            st.write(f"Specificity: {TN_train_rf / (TN_train_rf + FP_train_rf)}")

            st.write("Test Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred_rf)}")
            st.write(f"Sensitivity: {TP_test_rf / (TP_test_rf + FN_test_rf)}")
            st.write(f"Specificity: {TN_test_rf / (TN_test_rf + FP_test_rf)}")

            st.write("Feature Importance:")
            st.write(imp_df_rf)

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_test_pred_rf))

            st.subheader("Gradient Boosting Results:")
            st.write("Train Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred_gbm)}")
            st.write(f"Sensitivity: {TP_train_gbm / (TP_train_gbm + FN_train_gbm)}")
            st.write(f"Specificity: {TN_train_gbm / (TN_train_gbm + FP_train_gbm)}")

            st.write("Test Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred_gbm)}")
            st.write(f"Sensitivity: {TP_test_gbm / (TP_test_gbm + FN_test_gbm)}")
            st.write(f"Specificity: {TN_test_gbm / (TN_test_gbm + FP_test_gbm)}")

            st.write("Feature Importance:")
            st.write(imp_df_gbm)

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_test_pred_gbm))

            st.subheader("XGBoost Results:")
            st.write("Train Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred_xgb)}")
            st.write(f"Sensitivity: {TP_train_xgb / (TP_train_xgb + FN_train_xgb)}")
            st.write(f"Specificity: {TN_train_xgb / (TN_train_xgb + FP_train_xgb)}")

            st.write("Test Data Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred_xgb)}")
            st.write(f"Sensitivity: {TP_test_xgb / (TP_test_xgb + FN_test_xgb)}")
            st.write(f"Specificity: {TN_test_xgb / (TN_test_xgb + FP_test_xgb)}")

            st.write("Feature Importance:")
            st.write(imp_df_xgb)

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_test_pred_xgb))

        else:
            st.warning("Please select both outcome variable and independent variables.")
    
    else:
        st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()
