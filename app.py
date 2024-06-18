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
            - Visualize results with feature importance
                
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

            # Hyperparameters for GridSearchCV
            rf_param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 5, 10, None],
                'max_features': ['auto', 'sqrt', 'log2'],
                'criterion': ['gini', 'entropy']
            }

            # Option to manually enter hyperparameters
            if st.checkbox("Manually Enter Random Forest Hyperparameters"):
                rf_manual_params = {}
                rf_manual_params['n_estimators'] = st.number_input("Enter n_estimators:", min_value=1, value=100, step=1)
                rf_manual_params['max_depth'] = st.selectbox("Select max_depth:", [3, 5, 10, None])
                rf_manual_params['max_features'] = st.selectbox("Select max_features:", ['auto', 'sqrt', 'log2'])
                rf_manual_params['criterion'] = st.selectbox("Select criterion:", ['gini', 'entropy'])
            else:
                rf_manual_params = None

            if rf_manual_params:
                st.write("Using Manual Hyperparameters for Random Forest:")
                st.write(rf_manual_params)

            # Perform GridSearchCV if not using manual hyperparameters
            if not rf_manual_params:
                rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, n_jobs=-1)
                rf_grid.fit(X_train, y_train)
                rf_best_params = rf_grid.best_params_
            else:
                rf_best_params = rf_manual_params

            st.write("Best Parameters for Random Forest:")
            st.write(rf_best_params)

            # Train and test with best parameters
            rf_classifier = RandomForestClassifier(random_state=42, **rf_best_params)
            rf_classifier.fit(X_train, y_train)
            y_train_pred_rf = rf_classifier.predict(X_train)
            y_test_pred_rf = rf_classifier.predict(X_test)

            st.write("Train Data Classification Report for Random Forest:")
            st.write(classification_report(y_train, y_train_pred_rf))

            st.write("Test Data Classification Report for Random Forest:")
            st.write(classification_report(y_test, y_test_pred_rf))

            st.subheader("Feature Importance for Random Forest")
            feature_imp_rf = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.write(feature_imp_rf)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_imp_rf, y=feature_imp_rf.index)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Visualizing Important Features for Random Forest")
            st.pyplot(plt)

            # Gradient Boosting Classifier
            st.subheader("Gradient Boosting Classifier")

            # Hyperparameters for GridSearchCV
            gb_param_grid = {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 10],
                'subsample': [0.8, 1.0],
                'max_features': ['auto', 'sqrt', 'log2']
            }

            # Option to manually enter hyperparameters
            if st.checkbox("Manually Enter Gradient Boosting Hyperparameters"):
                gb_manual_params = {}
                gb_manual_params['n_estimators'] = st.number_input("Enter n_estimators:", min_value=1, value=100, step=1)
                gb_manual_params['learning_rate'] = st.selectbox("Select learning_rate:", [0.01, 0.1, 0.5])
                gb_manual_params['max_depth'] = st.selectbox("Select max_depth:", [3, 5, 10])
                gb_manual_params['subsample'] = st.selectbox("Select subsample:", [0.8, 1.0])
                gb_manual_params['max_features'] = st.selectbox("Select max_features:", ['auto', 'sqrt', 'log2'])
            else:
                gb_manual_params = None

            if gb_manual_params:
                st.write("Using Manual Hyperparameters for Gradient Boosting:")
                st.write(gb_manual_params)

            # Perform GridSearchCV if not using manual hyperparameters
            if not gb_manual_params:
                gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, n_jobs=-1)
                gb_grid.fit(X_train, y_train)
                gb_best_params = gb_grid.best_params_
            else:
                gb_best_params = gb_manual_params

            st.write("Best Parameters for Gradient Boosting:")
            st.write(gb_best_params)

            # Train and test with best parameters
            gb_classifier = GradientBoostingClassifier(random_state=42, **gb_best_params)
            gb_classifier.fit(X_train, y_train)
            y_train_pred_gb = gb_classifier.predict(X_train)
            y_test_pred_gb = gb_classifier.predict(X_test)

            st.write("Train Data Classification Report for Gradient Boosting:")
            st.write(classification_report(y_train, y_train_pred_gb))

            st.write("Test Data Classification Report for Gradient Boosting:")
            st.write(classification_report(y_test, y_test_pred_gb))

            st.subheader("Feature Importance for Gradient Boosting")
            feature_imp_gb = pd.Series(gb_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.write(feature_imp_gb)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_imp_gb, y=feature_imp_gb.index)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Visualizing Important Features for Gradient Boosting")
            st.pyplot(plt)

            # XGBoost Classifier
            st.subheader("XGBoost Classifier")

            # Hyperparameters for GridSearchCV
            xgb_param_grid = {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 10],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            # Option to manually enter hyperparameters
            if st.checkbox("Manually Enter XGBoost Hyperparameters"):
                xgb_manual_params = {}
                xgb_manual_params['n_estimators'] = st.number_input("Enter n_estimators:", min_value=1, value=100, step=1)
                xgb_manual_params['learning_rate'] = st.selectbox("Select learning_rate:", [0.01, 0.1, 0.5])
                xgb_manual_params['max_depth'] = st.selectbox("Select max_depth:", [3, 5, 10])
                xgb_manual_params['subsample'] = st.selectbox("Select subsample:", [0.8, 1.0])
                xgb_manual_params['colsample_bytree'] = st.selectbox("Select colsample_bytree:", [0.8, 1.0])
            else:
                xgb_manual_params = None

            if xgb_manual_params:
                st.write("Using Manual Hyperparameters for XGBoost:")
                st.write(xgb_manual_params)

            # Perform GridSearchCV if not using manual hyperparameters
            if not xgb_manual_params:
                xgb_grid = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=5, n_jobs=-1)
                xgb_grid.fit(X_train, y_train)
                xgb_best_params = xgb_grid.best_params_
            else:
                xgb_best_params = xgb_manual_params

            st.write("Best Parameters for XGBoost:")
            st.write(xgb_best_params)

            # Train and test with best parameters
            xgb_classifier = XGBClassifier(random_state=42, **xgb_best_params)
            xgb_classifier.fit(X_train, y_train)
            y_train_pred_xgb = xgb_classifier.predict(X_train)
            y_test_pred_xgb = xgb_classifier.predict(X_test)

            st.write("Train Data Classification Report for XGBoost:")
            st.write(classification_report(y_train, y_train_pred_xgb))

            st.write("Test Data Classification Report for XGBoost:")
            st.write(classification_report(y_test, y_test_pred_xgb))

            st.subheader("Feature Importance for XGBoost")
            feature_imp_xgb = pd.Series(xgb_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.write(feature_imp_xgb)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_imp_xgb, y=feature_imp_xgb.index)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Visualizing Important Features for XGBoost")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
