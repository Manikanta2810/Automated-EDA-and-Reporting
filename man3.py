import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# --- Page Configuration ---
st.set_page_config(
    page_title="Ultimate Gen AI EDA App",
    page_icon="🌟",
    layout="wide"
)

# --- State Management ---
if 'df' not in st.session_state: # Original uploaded DataFrame
    st.session_state.df = None
if 'df_cleaned' not in st.session_state: # DataFrame after cleaning operations
    st.session_state.df_cleaned = None
if 'cleaned_df_for_report' not in st.session_state: # Snapshot for the AI report
    st.session_state.cleaned_df_for_report = None
if 'report_log' not in st.session_state: # Log of user actions and AI outputs
    st.session_state.report_log = []
if 'target_variable' not in st.session_state: # Selected target variable
    st.session_state.target_variable = None
if 'uploaded_filename' not in st.session_state: # To track the current file
    st.session_state.uploaded_filename = None

# --- Helper & AI Functions ---
def get_gemini_explanation(prompt):
    """Calls the Gemini API to get an explanation."""
    api_key = "AIzaSyDGeinwr7M57wtEE3DFswrlDHa19qrWXk0" 
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180) 
        response.raise_for_status()
        result = response.json()
        if (result.get("candidates") and result["candidates"][0].get("content") and
            result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text")):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.error(f"Could not parse LLM response. Raw response: {result}")
            return "Error: Could not parse response from LLM."
    except requests.exceptions.Timeout:
        st.error("LLM API call timed out. The request took too long to process.")
        return "Error: API call timed out."
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling LLM API: {e}")
        return f"Error: API call failed ({e})."

def add_to_log(message):
    """Adds a message to the report log in session state."""
    st.session_state.report_log.append(message)

# --- UI Pages (Full implementation for all pages) ---

def page_data_overview():
    st.header("1. Initial Data Exploration")
    if st.session_state.df is None: 
        st.warning("Please upload a dataset first using the sidebar.")
        return
    
    df = st.session_state.df
    st.markdown("#### Dataset Preview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("#### Dataset Size and Shape")
    st.write(f"Number of Rows: **{df.shape[0]}**")
    st.write(f"Number of Columns: **{df.shape[1]}**")

    st.markdown("#### Column Data Types")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ['Column', 'Data Type']
    st.dataframe(dtypes_df, use_container_width=True)

    st.markdown("#### Summary Statistics (Numerical Columns)")
    st.dataframe(df.describe(include=np.number), use_container_width=True)
    st.markdown("#### Summary Statistics (Categorical/Object Columns)")
    st.dataframe(df.describe(include=['object', 'category']), use_container_width=True)

def page_data_cleaning():
    st.header("2. Data Cleaning and Preprocessing")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first using the sidebar.")
        return

    # Initialize or use existing df_cleaned
    if st.session_state.df_cleaned is None:
        st.session_state.df_cleaned = st.session_state.df.copy()
    
    df_cleaned = st.session_state.df_cleaned

    with st.expander("Handle Missing Values", expanded=True):
        missing_values = df_cleaned.isnull().sum()
        missing_df = missing_values[missing_values > 0].reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        st.write("Columns with missing values:")
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)

            col_to_impute = st.selectbox("Select column to impute", options=[""] + missing_df['Column'].tolist(), key="impute_col_select")
            if col_to_impute:
                imputation_method = st.radio("Select imputation method", ('Mean', 'Median', 'Mode', 'Remove Rows with Missing in this Column', 'Fill with Custom Value'), key=f"impute_method_{col_to_impute}")
                custom_fill_value_na = None
                if imputation_method == 'Fill with Custom Value':
                    custom_fill_value_na = st.text_input("Enter custom value to fill NA:", key=f"custom_fill_{col_to_impute}")

                if st.button(f"Apply Imputation to '{col_to_impute}'", key=f"apply_impute_{col_to_impute}"):
                    if imputation_method == 'Remove Rows with Missing in this Column':
                        rows_before = len(df_cleaned)
                        df_cleaned.dropna(subset=[col_to_impute], inplace=True)
                        rows_after = len(df_cleaned)
                        st.success(f"Removed {rows_before - rows_after} rows with missing values in '{col_to_impute}'.")
                        add_to_log(f"- Removed {rows_before - rows_after} rows with missing values in '{col_to_impute}'.")
                    elif imputation_method == 'Fill with Custom Value':
                        if custom_fill_value_na is not None and custom_fill_value_na != "":
                            try: 
                                original_dtype = st.session_state.df[col_to_impute].dtype 
                                fill_val_typed = pd.Series([custom_fill_value_na]).astype(original_dtype)[0]
                                df_cleaned[col_to_impute].fillna(fill_val_typed, inplace=True)
                            except ValueError: 
                                st.warning(f"Could not convert '{custom_fill_value_na}' to original type of '{col_to_impute}'. Filled as string.")
                                df_cleaned[col_to_impute].fillna(custom_fill_value_na, inplace=True)
                            st.success(f"Imputed missing values in '{col_to_impute}' with custom value '{custom_fill_value_na}'.")
                            add_to_log(f"- Imputed missing values in '{col_to_impute}' with custom value '{custom_fill_value_na}'.")
                        else:
                            st.error("Please enter a custom value.")
                    else: 
                        fill_value_calculated = None
                        if pd.api.types.is_numeric_dtype(df_cleaned[col_to_impute]):
                            if imputation_method == 'Mean': fill_value_calculated = df_cleaned[col_to_impute].mean()
                            elif imputation_method == 'Median': fill_value_calculated = df_cleaned[col_to_impute].median()
                            else: 
                                mode_val = df_cleaned[col_to_impute].mode()
                                if not mode_val.empty: fill_value_calculated = mode_val[0]
                        else: 
                            mode_val = df_cleaned[col_to_impute].mode()
                            if not mode_val.empty: fill_value_calculated = mode_val[0]
                        
                        if fill_value_calculated is not None:
                            df_cleaned[col_to_impute].fillna(fill_value_calculated, inplace=True)
                            st.success(f"Imputed missing values in '{col_to_impute}' with {imputation_method.lower()} ({fill_value_calculated}).")
                            add_to_log(f"- Imputed missing values in '{col_to_impute}' with {imputation_method.lower()} ({fill_value_calculated}).")
                        else:
                            st.error(f"Could not calculate {imputation_method.lower()} for '{col_to_impute}' (e.g., all values are NaN or no unique mode).")
                    
                    st.session_state.df_cleaned = df_cleaned.copy()
                    st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                    st.rerun()
        else:
            st.success("No missing values found in the current dataset!  ")

    with st.expander("Handle Duplicates"):
        num_duplicates = df_cleaned.duplicated().sum()
        st.write(f"Number of duplicate rows found: **{num_duplicates}**")
        if num_duplicates > 0:
            if st.button("Remove Duplicate Rows", key="remove_duplicates_btn"):
                df_cleaned.drop_duplicates(inplace=True, keep='first')
                st.session_state.df_cleaned = df_cleaned.copy()
                st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                st.success(f"Successfully removed {num_duplicates} duplicate rows.")
                add_to_log(f"- Removed {num_duplicates} duplicate rows from the dataset.")
                st.rerun()
    
    with st.expander("Correct Data Types"):
        col_to_convert = st.selectbox("Select column to convert type", options=[""] + df_cleaned.columns.tolist(), key="dtype_select_cleaning")
        if col_to_convert:
            current_type = df_cleaned[col_to_convert].dtype
            st.write(f"Current data type of '{col_to_convert}': **{current_type}**")
            
            new_type_options = ['object (string)', 'int64', 'float64', 'bool', 'datetime64[ns]', 'category']
            new_type_selected = st.selectbox("Select new data type", new_type_options, key=f"new_type_cleaning_{col_to_convert}")

            if st.button(f"Convert Data Type for '{col_to_convert}'", key=f"convert_type_cleaning_{col_to_convert}"):
                try:
                    if new_type_selected == 'datetime64[ns]':
                        df_cleaned[col_to_convert] = pd.to_datetime(df_cleaned[col_to_convert], errors='coerce')
                    elif new_type_selected == 'bool':
                        df_cleaned[col_to_convert] = df_cleaned[col_to_convert].replace({
                            'true': True, 'True': True, 'TRUE': True, '1': True, 1: True,
                            'false': False, 'False': False, 'FALSE': False, '0': False, 0: False
                        }).astype(bool)
                    else:
                        df_cleaned[col_to_convert] = df_cleaned[col_to_convert].astype(new_type_selected)
                    
                    st.session_state.df_cleaned = df_cleaned.copy()
                    st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                    st.success(f"Successfully converted '{col_to_convert}' to {new_type_selected}.")
                    add_to_log(f"- Converted column '{col_to_convert}' from {current_type} to {new_type_selected}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to convert column '{col_to_convert}' to {new_type_selected}. Error: {e}")

    with st.expander("Encode Categorical Variables (One-Hot Encoding)"):
        cat_cols_to_encode = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols_to_encode:
            st.info("No categorical columns found to encode.")
        else:
            col_to_encode_ohe = st.selectbox("Select categorical column to one-hot encode", options=[""] + cat_cols_to_encode, key="ohe_select")
            if col_to_encode_ohe:
                if st.button(f"Apply One-Hot Encoding to '{col_to_encode_ohe}'", key=f"apply_ohe_{col_to_encode_ohe}"):
                    try:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_data = encoder.fit_transform(df_cleaned[[col_to_encode_ohe]])
                        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col_to_encode_ohe]), index=df_cleaned.index)
                        df_cleaned = df_cleaned.join(encoded_df).drop(columns=[col_to_encode_ohe])
                        st.session_state.df_cleaned = df_cleaned.copy()
                        st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                        st.success(f"One-hot encoded '{col_to_encode_ohe}'. New columns added.")
                        add_to_log(f"- One-hot encoded the categorical column '{col_to_encode_ohe}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during one-hot encoding of '{col_to_encode_ohe}': {e}")
        
    with st.expander("Scale Numerical Data"):
        num_cols_for_scaling = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        if not num_cols_for_scaling:
            st.info("No numerical columns available for scaling.")
        else:
            col_to_scale_values = st.selectbox("Select numerical column to scale", options=[""] + num_cols_for_scaling, key="scale_select_cleaning")
            if col_to_scale_values:
                scaler_type_selected = st.radio("Select scaling method", ('Min-Max Scaling (Normalization)', 'Standard Scaling (Standardization)'), key=f"scaler_type_cleaning_{col_to_scale_values}")
                if st.button(f"Apply Scaling to '{col_to_scale_values}'", key=f"apply_scale_cleaning_{col_to_scale_values}"):
                    if scaler_type_selected == 'Min-Max Scaling (Normalization)': scaler = MinMaxScaler()
                    else: scaler = StandardScaler()
                    
                    new_scaled_col_name = f"{col_to_scale_values}_scaled"
                    if new_scaled_col_name in df_cleaned.columns:
                        st.warning(f"Column '{new_scaled_col_name}' already exists. Scaling will overwrite it.")
                    
                    df_cleaned[new_scaled_col_name] = scaler.fit_transform(df_cleaned[[col_to_scale_values]])
                    st.session_state.df_cleaned = df_cleaned.copy()
                    st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                    st.success(f"Applied {scaler_type_selected} to '{col_to_scale_values}'. New column: '{new_scaled_col_name}'")
                    add_to_log(f"- Applied {scaler_type_selected} to '{col_to_scale_values}', creating '{new_scaled_col_name}'.")
                    st.rerun()
    
    st.markdown("---")
    st.subheader("Preview of Data After Cleaning (First 5 rows)")
    st.dataframe(st.session_state.df_cleaned.head(), use_container_width=True)

def page_visualization():
    st.header("3. Data Visualization (General)")
    if st.session_state.df_cleaned is None:
        st.warning("Please upload and process data on the 'Data Cleaning' page first.")
        return

    df = st.session_state.df_cleaned
    
    st.markdown("### Univariate Analysis")
    uni_col_viz = st.selectbox("Select a column for univariate analysis", options=[""] + df.columns.tolist(), key="uni_viz_col_main")
    
    if uni_col_viz:
        if pd.api.types.is_numeric_dtype(df[uni_col_viz]):
            plot_type_num_viz = st.radio("Select plot type for numerical data", ('Histogram', 'Box Plot', 'KDE Plot'), key=f"plot_type_num_viz_{uni_col_viz}")
            fig, ax = plt.subplots()
            if plot_type_num_viz == 'Histogram': sns.histplot(df[uni_col_viz], kde=True, ax=ax); ax.set_title(f"Histogram of {uni_col_viz}")
            elif plot_type_num_viz == 'Box Plot': sns.boxplot(x=df[uni_col_viz], ax=ax); ax.set_title(f"Box Plot of {uni_col_viz}")
            elif plot_type_num_viz == 'KDE Plot': sns.kdeplot(df[uni_col_viz], fill=True, ax=ax); ax.set_title(f"KDE Plot of {uni_col_viz}")
            st.pyplot(fig); plt.clf()
        else: 
            plot_type_cat_viz = st.radio("Select plot type for categorical data", ('Bar Chart', 'Pie Chart'), key=f"plot_type_cat_viz_{uni_col_viz}")
            fig, ax = plt.subplots()
            counts = df[uni_col_viz].value_counts()
            if len(counts) > 15 and plot_type_cat_viz == 'Pie Chart':
                st.warning("Pie chart for >15 categories can be cluttered. Showing top 15."); counts = counts.nlargest(15)
            if len(counts) > 30 and plot_type_cat_viz == 'Bar Chart':
                st.warning("Bar chart for >30 categories can be cluttered. Showing top 30."); counts = counts.nlargest(30)
            
            if plot_type_cat_viz == 'Bar Chart': counts.plot(kind='bar', ax=ax); ax.set_title(f"Bar Chart of {uni_col_viz}")
            elif plot_type_cat_viz == 'Pie Chart': counts.plot(kind='pie', autopct='%1.1f%%', ax=ax); ax.set_title(f"Pie Chart of {uni_col_viz}")
            plt.xticks(rotation=45, ha='right'); st.pyplot(fig); plt.clf()
    
    st.markdown("---")
    st.markdown("### Bivariate and Multivariate Analysis")
    
    num_cols_for_viz = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_for_viz = df.select_dtypes(include=['object', 'category']).columns.tolist()

    with st.expander("Scatter Plots (Numerical vs. Numerical)"):
        if len(num_cols_for_viz) >= 2:
            x_axis_scatter = st.selectbox("Select X-axis variable", options=[""] + num_cols_for_viz, key='x_scatter_main')
            y_axis_scatter = st.selectbox("Select Y-axis variable", options=[""] + num_cols_for_viz, key='y_scatter_main')
            hue_scatter_main = st.selectbox("Select Hue variable (optional, categorical)", options=[None] + cat_cols_for_viz, key='hue_scatter_main')
            
            if x_axis_scatter and y_axis_scatter and x_axis_scatter != y_axis_scatter:
                fig, ax = plt.subplots(); sns.scatterplot(data=df, x=x_axis_scatter, y=y_axis_scatter, hue=hue_scatter_main, ax=ax)
                ax.set_title(f"Scatter Plot: {x_axis_scatter} vs. {y_axis_scatter}"); st.pyplot(fig); plt.clf()
            elif x_axis_scatter and y_axis_scatter and x_axis_scatter == y_axis_scatter:
                st.warning("Please select different variables for X and Y axis.")
        else: st.info("At least two numerical columns are needed for scatter plots.")

    with st.expander("Pair Plots (Multiple Numerical Features)"):
        if len(num_cols_for_viz) > 1:
            default_pair_cols_main = num_cols_for_viz[:min(len(num_cols_for_viz), 4)]
            selected_pair_cols_main = st.multiselect("Select columns for Pair Plot (3-5 recommended)", options=num_cols_for_viz, default=default_pair_cols_main, key="pairplot_cols_main")
            hue_pair_main = st.selectbox("Select Hue for Pair Plot (optional, categorical)", options=[None] + cat_cols_for_viz, key='hue_pairplot_main')
            
            if len(selected_pair_cols_main) > 1:
                if st.button("Generate Pair Plot", key="gen_pairplot_main_btn"):
                    with st.spinner("Generating Pair Plot..."):
                        try:
                            fig_pair_main = sns.pairplot(df[selected_pair_cols_main], hue=hue_pair_main if hue_pair_main else None, diag_kind='kde')
                            st.pyplot(fig_pair_main); plt.clf()
                        except Exception as e: st.error(f"Error generating pair plot: {e}")
            elif selected_pair_cols_main: st.warning("Please select at least two columns.")
        else: st.info("At least two numerical columns are needed for pair plots.")

    with st.expander("Correlation Heatmap (Numerical Features)"):
         if len(num_cols_for_viz) > 1:
            corr_main = df[num_cols_for_viz].corr()
            fig, ax = plt.subplots(figsize=(max(10, len(num_cols_for_viz)*0.7), max(8, len(num_cols_for_viz)*0.5)))
            sns.heatmap(corr_main, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
            ax.set_title("Correlation Heatmap of Numerical Features"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            st.pyplot(fig); plt.clf()
         else: st.info("At least two numerical columns are needed for a correlation heatmap.")

def page_target_analysis():
    st.header("4. Target Variable Analysis")
    st.markdown("Analyze features in relation to a specific target variable. **This selection will be remembered for the Modeling Advisor.**")
    if st.session_state.df_cleaned is None:
        st.warning("Please upload and process data on the 'Data Cleaning' page first.")
        return

    df = st.session_state.df_cleaned
    all_cols_target = [""] + df.columns.tolist()
    
    current_target_idx = 0
    if st.session_state.target_variable and st.session_state.target_variable in all_cols_target:
        current_target_idx = all_cols_target.index(st.session_state.target_variable)

    selected_target_variable = st.selectbox(
        "Select your Target Variable", 
        options=all_cols_target,
        index=current_target_idx,
        key="target_var_selector_page"
    )
    st.session_state.target_variable = selected_target_variable if selected_target_variable else None

    if st.session_state.target_variable:
        target_var_name = st.session_state.target_variable
        st.success(f"`{target_var_name}` is now set as the target variable for the app.")
        add_to_log(f"\n--- Target Variable Analysis Started: {target_var_name} ---")
        
        is_target_numeric_val = pd.api.types.is_numeric_dtype(df[target_var_name])
        feature_cols_list = [col for col in df.columns if col != target_var_name]

        if is_target_numeric_val:
            st.subheader(f"Analysis for Numerical Target: `{target_var_name}`")
            st.markdown("#### Correlation of Numerical Features with Target")
            num_features_for_corr = df[feature_cols_list].select_dtypes(include=np.number).columns.tolist()
            if num_features_for_corr:
                corr_target = df[num_features_for_corr + [target_var_name]].corr()[[target_var_name]].sort_values(by=target_var_name, ascending=False)
                corr_target = corr_target.drop(target_var_name, errors='ignore') 
                fig, ax = plt.subplots(figsize=(8, max(6, len(corr_target)*0.3)))
                sns.heatmap(corr_target, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=True)
                ax.set_title(f"Feature Correlation with '{target_var_name}'"); st.pyplot(fig); plt.clf()
                add_to_log(f"- Correlation heatmap for numerical target '{target_var_name}'.")
            else: st.info("No other numerical features to correlate.")

            st.markdown("#### Scatter Plots of Numerical Features vs. Target")
            if num_features_for_corr:
                scatter_num_feat_target = st.selectbox("Select numerical feature for scatter plot vs. target", options=[""] + num_features_for_corr, key="scatter_num_target_page")
                if scatter_num_feat_target:
                    fig, ax = plt.subplots(); sns.scatterplot(data=df, x=scatter_num_feat_target, y=target_var_name, ax=ax, alpha=0.6)
                    sns.regplot(data=df, x=scatter_num_feat_target, y=target_var_name, ax=ax, scatter=False, color='red')
                    ax.set_title(f"'{scatter_num_feat_target}' vs. '{target_var_name}'"); st.pyplot(fig); plt.clf()
                    add_to_log(f"- Scatter plot: '{scatter_num_feat_target}' vs. target '{target_var_name}'.")
            else: st.info("No numerical features for scatter plots.")
        else: # Categorical Target
            st.subheader(f"Analysis for Categorical Target: `{target_var_name}`")
            st.write(f"Target classes: {', '.join(map(str, df[target_var_name].unique()))}")

            st.markdown("#### Numerical Features vs. Categorical Target (Box Plots)")
            num_feat_cat_target_list = df[feature_cols_list].select_dtypes(include=np.number).columns.tolist()
            if num_feat_cat_target_list:
                num_feat_selected = st.selectbox("Select numerical feature", options=[""] + num_feat_cat_target_list, key="num_feat_cat_target_page")
                if num_feat_selected:
                    fig, ax = plt.subplots(); sns.boxplot(data=df, x=target_var_name, y=num_feat_selected, ax=ax)
                    ax.set_title(f"'{num_feat_selected}' by '{target_var_name}'"); plt.xticks(rotation=45, ha='right'); st.pyplot(fig); plt.clf()
                    add_to_log(f"- Box plot: '{num_feat_selected}' by target '{target_var_name}'.")
            else: st.info("No numerical features for this analysis.")

            st.markdown("#### Categorical Features vs. Categorical Target (Stacked Bar & Chi-Square)")
            cat_feat_cat_target_list = df[feature_cols_list].select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_feat_cat_target_list:
                cat_feat_selected = st.selectbox("Select categorical feature", options=[""] + cat_feat_cat_target_list, key="cat_feat_cat_target_page")
                if cat_feat_selected:
                    contingency = pd.crosstab(df[cat_feat_selected], df[target_var_name])
                    fig, ax = plt.subplots(figsize=(10,6)); contingency.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f"'{cat_feat_selected}' vs. '{target_var_name}'"); plt.xticks(rotation=45, ha='right'); st.pyplot(fig); plt.clf()
                    
                    st.write(f"**Chi-Square Test ('{cat_feat_selected}' vs. '{target_var_name}')**")
                    try:
                        chi2, p, dof, expected = chi2_contingency(contingency)
                        st.write(f"P-value: **{p:.4f}**")
                        if p < 0.05: st.success("Significant association (p < 0.05).")
                        else: st.warning("No significant association (p >= 0.05).")
                        add_to_log(f"- Stacked bar & Chi-Square: '{cat_feat_selected}' vs. target '{target_var_name}'. P-value: {p:.4f}")
                    except ValueError as e: st.error(f"Chi-Square error: {e}. (May be due to low expected frequencies).")
            else: st.info("No other categorical features for this analysis.")
    else:
        st.info("Select a target variable to begin analysis.")

def page_outlier_detection():
    st.header("5. Outlier Detection")
    st.markdown("Identify potential outliers in numerical data using the IQR method.")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please upload and process data on the 'Data Cleaning' page first.")
        return

    df = st.session_state.df_cleaned
    num_cols_outlier = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols_outlier:
        st.info("No numerical columns for outlier detection.")
        return

    col_check_outlier = st.selectbox("Select numerical column for outliers", options=[""] + num_cols_outlier, key="outlier_col_main_select")

    if col_check_outlier:
        st.markdown(f"### Outlier Analysis for: `{col_check_outlier}`")
        Q1 = df[col_check_outlier].quantile(0.25)
        Q3 = df[col_check_outlier].quantile(0.75)
        IQR = Q3 - Q1
        lower_b = Q1 - 1.5 * IQR
        upper_b = Q3 + 1.5 * IQR

        st.write(f"**Q1:** {Q1:.2f}, **Q3:** {Q3:.2f}, **IQR:** {IQR:.2f}")
        st.write(f"**Lower Bound:** {lower_b:.2f}, **Upper Bound:** {upper_b:.2f}")

        outliers_found_df = df[(df[col_check_outlier] < lower_b) | (df[col_check_outlier] > upper_b)]
        num_found_outliers = len(outliers_found_df)
        st.write(f"Potential outliers detected: **{num_found_outliers}**")

        if num_found_outliers > 0:
            st.markdown("#### Detected Outliers:"); st.dataframe(outliers_found_df, use_container_width=True)
            add_to_log(f"- Detected {num_found_outliers} outliers in '{col_check_outlier}' (IQR method).")
            st.markdown("#### Box Plot (showing outliers)")
            fig, ax = plt.subplots(); sns.boxplot(x=df[col_check_outlier], ax=ax)
            ax.set_title(f"Box Plot for '{col_check_outlier}'"); st.pyplot(fig); plt.clf()
        else:
            st.success("No outliers detected by 1.5 * IQR rule."); add_to_log(f"- No outliers in '{col_check_outlier}' (IQR method).")
    else:
        st.info("Select a numerical column.")

def page_hypothesis_testing():
    st.header("6. Hypothesis Testing (General)")
    if st.session_state.df_cleaned is None:
        st.warning("Please upload and process data on the 'Data Cleaning' page first.")
        return
    df = st.session_state.df_cleaned
    
    test_tabs_main = st.tabs(["Independent T-test", "Chi-Square Test of Independence", "ANOVA (One-Way)"])

    with test_tabs_main[0]: 
        st.markdown("#### Independent Samples T-test"); st.write("Compares means of a numerical variable between two groups of a categorical variable.")
        num_cols_ht_main = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_ht_main = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        num_var_ttest_main = st.selectbox("Numerical variable", options=[""] + num_cols_ht_main, key='ttest_num_main')
        cat_var_ttest_main = st.selectbox("Categorical variable (2 groups)", options=[""] + cat_cols_ht_main, key='ttest_cat_main')

        if st.button("Run T-test", key="run_ttest_main_btn"):
            if num_var_ttest_main and cat_var_ttest_main:
                groups = df[cat_var_ttest_main].dropna().unique()
                if len(groups) == 2:
                    g1 = df[df[cat_var_ttest_main] == groups[0]][num_var_ttest_main].dropna()
                    g2 = df[df[cat_var_ttest_main] == groups[1]][num_var_ttest_main].dropna()
                    if len(g1) < 2 or len(g2) < 2: st.error("Not enough data in one/both groups.")
                    else:
                        stat, p = ttest_ind(g1, g2, equal_var=False) 
                        st.write(f"**T-statistic:** {stat:.4f}, **P-value:** {p:.4f}")
                        if p < 0.05: st.success("Significant difference (p < 0.05).")
                        else: st.warning("No significant difference (p >= 0.05).")
                        add_to_log(f"- T-test: '{num_var_ttest_main}' by '{cat_var_ttest_main}'. P-value: {p:.4f}.")
                else: st.error(f"'{cat_var_ttest_main}' has {len(groups)} groups. T-test needs 2.")
            else: st.error("Select both variables.")

    with test_tabs_main[1]: 
        st.markdown("#### Chi-Square Test of Independence"); st.write("Tests association between two categorical variables.")
        cat_cols_chi_main = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols_chi_main) < 2: st.info("Need at least two categorical columns.")
        else:
            cat1_chi = st.selectbox("First categorical variable", options=[""] + cat_cols_chi_main, key='chi_cat1_main')
            cat2_chi = st.selectbox("Second categorical variable", options=[""] + cat_cols_chi_main, key='chi_cat2_main')
            if st.button("Run Chi-Square Test", key="run_chi_main_btn"):
                 if cat1_chi and cat2_chi and cat1_chi != cat2_chi:
                     try:
                        table = pd.crosstab(df[cat1_chi], df[cat2_chi]); st.write("Contingency Table:"); st.dataframe(table)
                        chi2, p, dof, expected = chi2_contingency(table)
                        st.write(f"**Chi2:** {chi2:.4f}, **P-value:** {p:.4f}, **DoF:** {dof}")
                        if p < 0.05: st.success("Significant association (p < 0.05).")
                        else: st.warning("No significant association (p >= 0.05).")
                        add_to_log(f"- Chi-Square: '{cat1_chi}' vs '{cat2_chi}'. P-value: {p:.4f}.")
                     except ValueError as e: st.error(f"Chi-Square error: {e}.")
                 else: st.error("Select two different variables.")
        
    with test_tabs_main[2]:
        st.markdown("#### ANOVA (One-Way)"); st.write("Compares means of a numerical variable across >2 groups of a categorical variable.")
        num_cols_anova_main = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_anova_main = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_var_anova_main = st.selectbox("Numerical variable", options=[""] + num_cols_anova_main, key='anova_num_main')
        cat_var_anova_main = st.selectbox("Categorical variable (>2 groups)", options=[""] + cat_cols_anova_main, key='anova_cat_main')

        if st.button("Run ANOVA", key="run_anova_main_btn"):
            if num_var_anova_main and cat_var_anova_main:
                groups_a = df[cat_var_anova_main].dropna().unique()
                if len(groups_a) > 2:
                    samples = [df[df[cat_var_anova_main] == g][num_var_anova_main].dropna() for g in groups_a]
                    samples_filt = [s for s in samples if len(s) >= 2]
                    if len(samples_filt) < 2 : st.error(f"ANOVA needs >=2 groups with >=2 samples each. Found {len(samples_filt)}.")
                    else:
                        f_stat, p_val = f_oneway(*samples_filt)
                        st.write(f"**F-statistic:** {f_stat:.4f}, **P-value:** {p_val:.4f}")
                        if p_val < 0.05: st.success("Significant difference between group means (p < 0.05).")
                        else: st.warning("No significant difference between group means (p >= 0.05).")
                        add_to_log(f"- ANOVA: '{num_var_anova_main}' by '{cat_var_anova_main}'. P-value: {p_val:.4f}.")
                else: st.error(f"'{cat_var_anova_main}' has {len(groups_a)} groups. ANOVA needs >2. For 2, use T-test.")
            else: st.error("Select both variables.")

def page_modeling_advisor():
    st.header("7. AI Modeling Advisor")
    st.markdown("Ask questions about potential machine learning models for your dataset.")
    if st.session_state.df_cleaned is None:
        st.warning("Please upload and process your data first.")
        return
    df = st.session_state.df_cleaned
    
    target_var_advisor = st.session_state.get('target_variable', None)
    if not target_var_advisor:
        st.error("Please select a target variable on the '4. Target Variable Analysis' page first.")
        return
    st.info(f"Current target variable: **`{target_var_advisor}`**")

    user_question_model = st.text_area("Ask a modeling question", 
        f"Given the target '{target_var_advisor}', what are suitable models? Compare Random Forest and Logistic Regression (if classification) or Linear Regression (if regression).",
        height=150, key="model_advisor_q_main")

    if st.button("Get AI Modeling Advice", type="primary", key="get_model_advice_main_btn"):
        with st.spinner("AI advisor is thinking..."):
            problem_type_advisor = "Unknown"
            if pd.api.types.is_numeric_dtype(df[target_var_advisor]): problem_type_advisor = "Regression"
            elif df[target_var_advisor].nunique() / len(df) < 0.5: problem_type_advisor = f"Classification ({df[target_var_advisor].nunique()} classes)"

            context_summary_model = f"""
            - Dataset Shape: {df.shape}
            - Target Variable: '{target_var_advisor}' (Type: {df[target_var_advisor].dtype}, Problem: {problem_type_advisor})
            - Columns & Data Types:\n{df.dtypes.to_string()}
            - Descriptive Stats:\n{df.describe(include='all').to_string()}
            """
            prompt_model = f"""
            You are a Data Science Modeling Advisor. Based on the dataset context and user's question, provide guidance.
            Dataset Context:\n{context_summary_model}
            User's Question: "{user_question_model}"
            Your Task: Answer the user's question. Explain reasoning, mention prep steps, compare models if asked (pros/cons for this data), and conclude. Use markdown.
            """
            advice_model = get_gemini_explanation(prompt_model)
            st.markdown("### AI Advisor's Recommendation"); st.markdown(advice_model)
            add_to_log(f"\n--- AI Modeling Advice (Target: {target_var_advisor}, Q: '{user_question_model}') ---\n{advice_model}")

def page_ai_summary():
    st.header("8. AI Summary & Reporting")
    current_df_report = st.session_state.cleaned_df_for_report if st.session_state.cleaned_df_for_report is not None else st.session_state.df
    if current_df_report is None: st.warning("Please upload data first."); return

    st.markdown("Generate a comprehensive AI report based on the dataset and actions taken.")
    if st.button("Generate Full AI EDA Report", type="primary", key="gen_ai_final_report_main_btn"):
        with st.spinner("AI is generating the full EDA report..."):
            summary_for_ai_report = f"""
            Dataset Shape: {current_df_report.shape}
            Data Types:\n{current_df_report.dtypes.to_string()}
            Descriptive Stats (Numerical):\n{current_df_report.describe(include=np.number).to_string()}
            Descriptive Stats (Categorical):\n{current_df_report.describe(include=['object', 'category']).to_string()}
            Missing Values (Final):\n{current_df_report.isnull().sum().to_string()}
            """
            log_for_ai_report = "\n".join(st.session_state.report_log) if st.session_state.report_log else "No specific actions logged."
            target_info_for_report = f"Target Variable Selected for Analysis (if any): {st.session_state.target_variable}" if st.session_state.target_variable else "No specific target variable was focused on in the 'Target Analysis' section."

            prompt_final_report = f"""
            You are an expert data scientist writing a final EDA report.
            Dataset Summary:\n{summary_for_ai_report}
            {target_info_for_report}
            Log of Actions & Key Findings:\n{log_for_ai_report}
            Your Task: Write a comprehensive, narrative EDA report covering:
            1. Introduction & Dataset Overview.
            2. Data Quality & Cleaning Insights (based on logs and final missing values).
            3. Key Univariate Insights (from stats).
            4. Key Bivariate/Multivariate Insights (infer from typical EDA and logged target analysis/tests).
            5. Outlier Observations (if logged).
            6. Hypothesis Testing Summary (if logged).
            7. Modeling Considerations (if target variable was set and modeling advice logged).
            8. Potential Limitations & Biases.
            9. Actionable Recommendations & Next Steps.
            Structure clearly with markdown. Be insightful.
            """
            report_final_content = get_gemini_explanation(prompt_final_report)
            st.markdown("### AI Generated EDA Report:"); st.markdown(report_final_content, unsafe_allow_html=True)
            add_to_log(f"\n--- Full AI EDA Report ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n{report_final_content}")

    st.markdown("---"); st.markdown("### Download EDA Log & Report")
    if st.session_state.report_log:
        download_content_final = "\n".join(st.session_state.report_log)
        st.download_button("Download Full Log & Reports (TXT)", download_content_final, "full_eda_report.txt", "text/plain", key="download_final_report_btn")
    else: st.info("No actions logged yet.")

# --- Main App ---
st.sidebar.title("🌟 Gen AI EDA Tool")
st.sidebar.markdown("Your comprehensive EDA assistant.")

uploaded_file_main = st.sidebar.file_uploader("Upload your CSV file", type="csv", key="main_csv_uploader")

# CORRECTED FILE UPLOAD LOGIC TO PRESERVE STATE
if uploaded_file_main is not None:
    # Check if this is a new file upload
    if st.session_state.uploaded_filename != uploaded_file_main.name:
        try:
            st.sidebar.info("New file detected. Loading data...")
            df_load = pd.read_csv(uploaded_file_main)
            # Reset all state for the new file
            st.session_state.df = df_load.copy()
            st.session_state.df_cleaned = df_load.copy() 
            st.session_state.cleaned_df_for_report = df_load.copy()
            st.session_state.target_variable = None 
            st.session_state.report_log = [f"- Dataset '{uploaded_file_main.name}' loaded: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}."]
            st.session_state.uploaded_filename = uploaded_file_main.name
            st.sidebar.success(f"'{uploaded_file_main.name}' loaded!")
            st.rerun() # Rerun to reflect the new state immediately
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            keys_to_clear = ['df', 'df_cleaned', 'cleaned_df_for_report', 'target_variable', 'report_log', 'uploaded_filename']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
else:
    # This block allows clearing the file uploader widget without losing state
    # A more advanced implementation might have a "Clear Data" button
    pass

if 'df' in st.session_state and st.session_state.df is not None:
    page_options_main = [
        "1. Data Overview", "2. Data Cleaning & Preprocessing", "3. Data Visualization (General)",
        "4. Target Variable Analysis", "5. Outlier Detection", "6. Hypothesis Testing (General)",
        "7. AI Modeling Advisor", "8. AI Summary & Reporting"
    ]
    page_selected = st.sidebar.radio("Select Analysis Stage:", page_options_main, key="main_page_nav_radio")

    # Routing logic
    if page_selected == page_options_main[0]: page_data_overview()
    elif page_selected == page_options_main[1]: page_data_cleaning()
    elif page_selected == page_options_main[2]: page_visualization()
    elif page_selected == page_options_main[3]: page_target_analysis()
    elif page_selected == page_options_main[4]: page_outlier_detection()
    elif page_selected == page_options_main[5]: page_hypothesis_testing()
    elif page_selected == page_options_main[6]: page_modeling_advisor() 
    elif page_selected == page_options_main[7]: page_ai_summary()
else:
    st.info("⬆️ Upload a CSV file using the sidebar to begin your Exploratory Data Analysis journey!")

