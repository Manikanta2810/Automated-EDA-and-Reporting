import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import io # For capturing print outputs
import contextlib # For redirecting stdout
import sys # To access stdout
from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler # These are correctly imported
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
load_dotenv()
# --- Page Configuration ---
st.set_page_config(
    page_title="Ultimate Gen AI EDA App",
    page_icon="�",
    layout="wide"
)

# --- State Management ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'cleaned_df_for_report' not in st.session_state:
    st.session_state.cleaned_df_for_report = None
if 'report_log' not in st.session_state:
    st.session_state.report_log = []
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Helper & AI Functions ---
def get_gemini_explanation(prompt: str) -> str:
    """Calls the Gemini API to get an explanation."""
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("API Key not found. Please set GEMINI_API_KEY in your .env file or environment variables.")
    return "Error: API Key not configured"
    # Corrected: Should be empty for env-injected keys or replaced by user for local.
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

def add_to_log(message: str):
    """Adds a message to the application's report log."""
    st.session_state.report_log.append(message)

# --- EDA UI Pages (Full Implementations) ---

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
                        rows_before = len(df_cleaned); df_cleaned.dropna(subset=[col_to_impute], inplace=True); rows_after = len(df_cleaned)
                        st.success(f"Removed {rows_before - rows_after} rows with missing values in '{col_to_impute}'.")
                        add_to_log(f"- Removed {rows_before - rows_after} rows with missing values in '{col_to_impute}'.")
                    elif imputation_method == 'Fill with Custom Value':
                        if custom_fill_value_na is not None and custom_fill_value_na != "":
                            try: 
                                original_dtype = st.session_state.df[col_to_impute].dtype 
                                fill_val_typed = pd.Series([custom_fill_value_na]).astype(original_dtype)[0]
                                df_cleaned[col_to_impute].fillna(fill_val_typed, inplace=True)
                            except ValueError: 
                                st.warning(f"Could not convert '{custom_fill_value_na}' to original type. Filled as string.")
                                df_cleaned[col_to_impute].fillna(custom_fill_value_na, inplace=True)
                            st.success(f"Imputed missing values in '{col_to_impute}' with '{custom_fill_value_na}'.")
                            add_to_log(f"- Imputed missing in '{col_to_impute}' with custom value '{custom_fill_value_na}'.")
                        else: st.error("Please enter a custom value.")
                    else: 
                        fill_val = None
                        if pd.api.types.is_numeric_dtype(df_cleaned[col_to_impute]):
                            if imputation_method == 'Mean': fill_val = df_cleaned[col_to_impute].mean()
                            elif imputation_method == 'Median': fill_val = df_cleaned[col_to_impute].median()
                            else: mode_val = df_cleaned[col_to_impute].mode(); fill_val = mode_val[0] if not mode_val.empty else None
                        else: mode_val = df_cleaned[col_to_impute].mode(); fill_val = mode_val[0] if not mode_val.empty else None
                        if fill_val is not None:
                            df_cleaned[col_to_impute].fillna(fill_val, inplace=True)
                            st.success(f"Imputed missing in '{col_to_impute}' with {imputation_method.lower()} ({fill_val}).")
                            add_to_log(f"- Imputed missing in '{col_to_impute}' with {imputation_method.lower()} ({fill_val}).")
                        else: st.error(f"Could not calculate {imputation_method.lower()} for '{col_to_impute}'.")
                    st.session_state.df_cleaned = df_cleaned.copy(); st.session_state.cleaned_df_for_report = df_cleaned.copy(); st.rerun()
        else: st.success("No missing values found! 🎉")

    with st.expander("Handle Duplicates"):
        num_duplicates = df_cleaned.duplicated().sum()
        st.write(f"Duplicate rows found: **{num_duplicates}**")
        if num_duplicates > 0 and st.button("Remove Duplicate Rows", key="remove_dup_btn"):
            df_cleaned.drop_duplicates(inplace=True, keep='first')
            st.session_state.df_cleaned = df_cleaned.copy(); st.session_state.cleaned_df_for_report = df_cleaned.copy()
            st.success(f"Removed {num_duplicates} duplicate rows."); add_to_log(f"- Removed {num_duplicates} duplicates."); st.rerun()
    
    with st.expander("Correct Data Types"):
        col_conv = st.selectbox("Select column to convert type", [""] + df_cleaned.columns.tolist(), key="dtype_conv_select")
        if col_conv:
            curr_type = df_cleaned[col_conv].dtype; st.write(f"Current type of '{col_conv}': **{curr_type}**")
            new_type_opts = ['object (string)', 'int64', 'float64', 'bool', 'datetime64[ns]', 'category']
            new_type_sel = st.selectbox("Select new data type", new_type_opts, key=f"new_type_sel_{col_conv}")
            if st.button(f"Convert '{col_conv}' to {new_type_sel}", key=f"conv_btn_{col_conv}"):
                try:
                    if new_type_sel == 'datetime64[ns]': df_cleaned[col_conv] = pd.to_datetime(df_cleaned[col_conv], errors='coerce')
                    elif new_type_sel == 'bool':
                        df_cleaned[col_conv] = df_cleaned[col_conv].replace({'true': True, 'True': True, 'TRUE': True, '1': True, 1: True,'false': False, 'False': False, 'FALSE': False, '0': False, 0: False}).astype(bool)
                    else: df_cleaned[col_conv] = df_cleaned[col_conv].astype(new_type_sel)
                    st.session_state.df_cleaned = df_cleaned.copy(); st.session_state.cleaned_df_for_report = df_cleaned.copy()
                    st.success(f"Converted '{col_conv}' to {new_type_sel}."); add_to_log(f"- Converted '{col_conv}' to {new_type_sel}."); st.rerun()
                except Exception as e: st.error(f"Conversion failed: {e}")

    # --- CORRECTED AND REFINED SECTIONS ---
    with st.expander("Encode Categorical Variables (One-Hot Encoding)"):
        # Filter for object or category type columns that are not already one-hot encoded (heuristic: many unique values or specific naming)
        potential_cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not potential_cat_cols:
            st.info("No suitable categorical columns found for one-hot encoding.")
        else:
            col_ohe = st.selectbox("Select categorical column to one-hot encode", [""] + potential_cat_cols, key="ohe_col_select")
            if col_ohe :
                if st.button(f"One-Hot Encode '{col_ohe}'", key=f"ohe_btn_{col_ohe}"):
                    try:
                        if col_ohe not in df_cleaned.columns:
                             st.error(f"Column '{col_ohe}' not found. It might have been removed or renamed.")
                        elif df_cleaned[col_ohe].nunique() > 50: # Heuristic: warn if too many unique values
                            st.warning(f"Column '{col_ohe}' has many unique values ({df_cleaned[col_ohe].nunique()}). One-hot encoding might create many new columns.")
                        
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_data = encoder.fit_transform(df_cleaned[[col_ohe]])
                        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col_ohe]), index=df_cleaned.index)
                        
                        df_cleaned = df_cleaned.drop(columns=[col_ohe]) # Drop original column
                        df_cleaned = df_cleaned.join(encoded_df) # Join new encoded columns
                        
                        st.session_state.df_cleaned = df_cleaned.copy()
                        st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                        st.success(f"One-hot encoded '{col_ohe}'. Original column dropped and new encoded columns added.")
                        add_to_log(f"- One-hot encoded '{col_ohe}'. Original dropped, {len(encoded_df.columns)} new columns added.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"One-Hot Encoding failed for '{col_ohe}': {e}")
        
    with st.expander("Scale Numerical Data"):
        numerical_cols_for_scaling = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        # Exclude columns that already seem to be scaled to prevent re-scaling the same data
        unscaled_numerical_cols = [col for col in numerical_cols_for_scaling if not col.endswith("_scaled")]

        if not unscaled_numerical_cols:
            st.info("No unscaled numerical columns available for scaling, or all numerical columns appear to have been scaled already.")
        else:
            col_to_scale_values = st.selectbox(
                "Select numerical column to scale", 
                options=[""] + unscaled_numerical_cols, # Offer only unscaled columns
                key="scale_select_cleaning"
            )
            if col_to_scale_values:
                scaler_type_selected = st.radio(
                    "Select scaling method", 
                    ('Min-Max Scaling (Normalization)', 'Standard Scaling (Standardization)'), 
                    key=f"scaler_type_cleaning_{col_to_scale_values}"
                )
                new_scaled_col_name = f"{col_to_scale_values}_scaled" # Define new column name

                if st.button(f"Scale '{col_to_scale_values}' into '{new_scaled_col_name}'", key=f"apply_scale_cleaning_{col_to_scale_values}"):
                    if new_scaled_col_name in df_cleaned.columns:
                        st.warning(f"Column '{new_scaled_col_name}' already exists and will be overwritten.")
                    
                    if scaler_type_selected == 'Min-Max Scaling (Normalization)':
                        scaler = MinMaxScaler()
                    else: # Standard Scaling
                        scaler = StandardScaler()
                    
                    try:
                        df_cleaned[new_scaled_col_name] = scaler.fit_transform(df_cleaned[[col_to_scale_values]])
                        st.session_state.df_cleaned = df_cleaned.copy()
                        st.session_state.cleaned_df_for_report = st.session_state.df_cleaned.copy()
                        st.success(f"Applied {scaler_type_selected} to '{col_to_scale_values}'. New column created: '{new_scaled_col_name}'")
                        add_to_log(f"- Applied {scaler_type_selected} to '{col_to_scale_values}', creating '{new_scaled_col_name}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Scaling failed for '{col_to_scale_values}': {e}")
    # --- END OF CORRECTED AND REFINED SECTIONS ---

    st.markdown("---"); st.subheader("Preview of Data After Cleaning"); st.dataframe(df_cleaned.head(), use_container_width=True)

def page_visualization():
    st.header("3. Data Visualization (General)")
    if st.session_state.df_cleaned is None: st.warning("Please clean data first."); return
    df = st.session_state.df_cleaned
    st.markdown("### Univariate Analysis")
    uni_col = st.selectbox("Select column for univariate analysis", [""] + df.columns.tolist(), key="uni_viz_select")
    if uni_col:
        if pd.api.types.is_numeric_dtype(df[uni_col]):
            plot_t_num = st.radio("Plot type", ('Histogram', 'Box Plot', 'KDE Plot'), key=f"plot_num_{uni_col}")
            fig, ax = plt.subplots()
            if plot_t_num == 'Histogram': sns.histplot(df[uni_col], kde=True, ax=ax); ax.set_title(f"Histogram: {uni_col}")
            elif plot_t_num == 'Box Plot': sns.boxplot(x=df[uni_col], ax=ax); ax.set_title(f"Box Plot: {uni_col}")
            else: sns.kdeplot(df[uni_col], fill=True, ax=ax); ax.set_title(f"KDE Plot: {uni_col}")
            st.pyplot(fig); plt.clf()
        else:
            plot_t_cat = st.radio("Plot type", ('Bar Chart', 'Pie Chart'), key=f"plot_cat_{uni_col}")
            fig, ax = plt.subplots(); counts = df[uni_col].value_counts()
            if len(counts) > 15 and plot_t_cat == 'Pie Chart': st.warning("Top 15 shown for Pie Chart."); counts = counts.nlargest(15)
            if len(counts) > 30 and plot_t_cat == 'Bar Chart': st.warning("Top 30 shown for Bar Chart."); counts = counts.nlargest(30)
            if plot_t_cat == 'Bar Chart': counts.plot(kind='bar', ax=ax); ax.set_title(f"Bar Chart: {uni_col}")
            else: counts.plot(kind='pie', autopct='%1.1f%%', ax=ax); ax.set_title(f"Pie Chart: {uni_col}")
            plt.xticks(rotation=45, ha='right'); st.pyplot(fig); plt.clf()

    st.markdown("---"); st.markdown("### Bivariate and Multivariate Analysis")
    num_cols_viz_bi = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_viz_bi = df.select_dtypes(include=['object', 'category']).columns.tolist()
    with st.expander("Scatter Plots"):
        if len(num_cols_viz_bi) >= 2:
            x_sc = st.selectbox("X-axis", [""] + num_cols_viz_bi, key='x_sc_main'); y_sc = st.selectbox("Y-axis", [""] + num_cols_viz_bi, key='y_sc_main')
            hue_sc = st.selectbox("Hue (optional)", [None] + cat_cols_viz_bi, key='hue_sc_main')
            if x_sc and y_sc and x_sc != y_sc:
                fig, ax = plt.subplots(); sns.scatterplot(data=df, x=x_sc, y=y_sc, hue=hue_sc, ax=ax)
                ax.set_title(f"{x_sc} vs. {y_sc}"); st.pyplot(fig); plt.clf()
            elif x_sc and y_sc: st.warning("Select different X and Y.")
        else: st.info("Need >=2 numerical columns.")
    with st.expander("Pair Plots"):
        if len(num_cols_viz_bi) > 1:
            sel_pair = st.multiselect("Select columns (3-5 recommended)", num_cols_viz_bi, default=num_cols_viz_bi[:min(len(num_cols_viz_bi),4)], key="pair_sel_main")
            hue_pair = st.selectbox("Hue (optional)", [None] + cat_cols_viz_bi, key='hue_pair_main')
            if len(sel_pair) > 1 and st.button("Generate Pair Plot", key="gen_pair_main_btn"):
                with st.spinner("Generating..."):
                    try: fig_p = sns.pairplot(df[sel_pair], hue=hue_pair if hue_pair else None, diag_kind='kde'); st.pyplot(fig_p); plt.clf()
                    except Exception as e: st.error(f"Pair plot error: {e}")
            elif sel_pair: st.warning("Select >=2 columns.")
        else: st.info("Need >=2 numerical columns.")
    with st.expander("Correlation Heatmap"):
         if len(num_cols_viz_bi) > 1:
            corr = df[num_cols_viz_bi].corr()
            fig, ax = plt.subplots(figsize=(max(10,len(num_cols_viz_bi)*0.7), max(8,len(num_cols_viz_bi)*0.5)))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size":8})
            ax.set_title("Correlation Heatmap"); plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0); st.pyplot(fig); plt.clf()
         else: st.info("Need >=2 numerical columns.")

def page_target_analysis():
    st.header("4. Target Variable Analysis")
    st.markdown("Analyze features relative to a target. **Selection remembered for Modeling Advisor.**")
    if st.session_state.df_cleaned is None: st.warning("Please clean data first."); return
    df = st.session_state.df_cleaned; all_cols = [""] + df.columns.tolist()
    curr_idx = all_cols.index(st.session_state.target_variable) if st.session_state.target_variable and st.session_state.target_variable in all_cols else 0
    sel_target = st.selectbox("Select Target Variable", all_cols, index=curr_idx, key="target_sel_page_main")
    st.session_state.target_variable = sel_target if sel_target else None
    if st.session_state.target_variable:
        target = st.session_state.target_variable; st.success(f"`{target}` set as target.")
        add_to_log(f"\n--- Target Analysis: {target} ---")
        is_num_target = pd.api.types.is_numeric_dtype(df[target])
        features = [col for col in df.columns if col != target]
        if is_num_target:
            st.subheader(f"Numerical Target: `{target}`"); st.markdown("#### Correlation with Target")
            num_feats_corr = df[features].select_dtypes(include=np.number).columns.tolist()
            if num_feats_corr:
                corr_df = df[num_feats_corr + [target]].corr()[[target]].sort_values(by=target, ascending=False).drop(target, errors='ignore')
                fig, ax = plt.subplots(figsize=(8,max(6,len(corr_df)*0.3))); sns.heatmap(corr_df,annot=True,fmt=".2f",cmap="viridis",ax=ax,cbar=True)
                ax.set_title(f"Feature Correlation with '{target}'"); st.pyplot(fig); plt.clf(); add_to_log(f"- Corr heatmap for num target '{target}'.")
            else: st.info("No other numerical features.")
            st.markdown("#### Scatter Plots vs. Target")
            if num_feats_corr:
                sc_feat = st.selectbox("Select numerical feature for scatter", [""]+num_feats_corr, key="sc_num_target_main")
                if sc_feat:
                    fig,ax=plt.subplots(); sns.scatterplot(data=df,x=sc_feat,y=target,ax=ax,alpha=0.6); sns.regplot(data=df,x=sc_feat,y=target,ax=ax,scatter=False,color='red')
                    ax.set_title(f"'{sc_feat}' vs. '{target}'"); st.pyplot(fig); plt.clf(); add_to_log(f"- Scatter: '{sc_feat}' vs. target '{target}'.")
            else: st.info("No numerical features for scatter.")
        else: # Categorical Target
            st.subheader(f"Categorical Target: `{target}`"); st.write(f"Classes: {', '.join(map(str, df[target].unique()))}")
            st.markdown("#### Numerical Features vs. Categorical Target (Box Plots)")
            num_feats_cat_t = df[features].select_dtypes(include=np.number).columns.tolist()
            if num_feats_cat_t:
                sel_num_feat = st.selectbox("Select numerical feature", [""]+num_feats_cat_t, key="num_feat_cat_t_main")
                if sel_num_feat:
                    fig,ax=plt.subplots(); sns.boxplot(data=df,x=target,y=sel_num_feat,ax=ax); ax.set_title(f"'{sel_num_feat}' by '{target}'")
                    plt.xticks(rotation=45,ha='right'); st.pyplot(fig); plt.clf(); add_to_log(f"- Box plot: '{sel_num_feat}' by target '{target}'.")
            else: st.info("No numerical features.")
            st.markdown("#### Categorical Features vs. Categorical Target (Stacked Bar & Chi-Square)")
            cat_feats_cat_t = df[features].select_dtypes(include=['object','category']).columns.tolist()
            if cat_feats_cat_t:
                sel_cat_feat = st.selectbox("Select categorical feature", [""]+cat_feats_cat_t, key="cat_feat_cat_t_main")
                if sel_cat_feat:
                    cont_table = pd.crosstab(df[sel_cat_feat], df[target])
                    fig,ax=plt.subplots(figsize=(10,6)); cont_table.plot(kind='bar',stacked=True,ax=ax); ax.set_title(f"'{sel_cat_feat}' vs. '{target}'")
                    plt.xticks(rotation=45,ha='right'); st.pyplot(fig); plt.clf()
                    st.write(f"**Chi-Square ('{sel_cat_feat}' vs. '{target}')**")
                    try:
                        chi2,p_val_chi_target,dof,exp = chi2_contingency(cont_table) # Renamed p
                        st.write(f"P-value: **{p_val_chi_target:.4f}**")
                        if p_val_chi_target < 0.05:
                            st.success("Significant association.")
                        else:
                            st.warning("No significant association.")
                        add_to_log(f"- Stacked bar & Chi2: '{sel_cat_feat}' vs. target '{target}'. P: {p_val_chi_target:.4f}")
                    except ValueError as e: st.error(f"Chi2 error: {e}.")
            else: st.info("No other categorical features.")
    else: st.info("Select a target variable.")

def page_outlier_detection():
    st.header("5. Outlier Detection")
    st.markdown("Identify outliers using the IQR method.")
    if st.session_state.df_cleaned is None: st.warning("Please clean data first."); return
    df = st.session_state.df_cleaned; num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols: st.info("No numerical columns."); return
    col_check = st.selectbox("Select column for outliers", [""]+num_cols, key="outlier_sel_main")
    if col_check:
        st.markdown(f"### Outlier Analysis for: `{col_check}`"); Q1=df[col_check].quantile(0.25); Q3=df[col_check].quantile(0.75); IQR=Q3-Q1
        low_b=Q1-1.5*IQR; upp_b=Q3+1.5*IQR
        st.write(f"**Q1:** {Q1:.2f}, **Q3:** {Q3:.2f}, **IQR:** {IQR:.2f}, **Lower:** {low_b:.2f}, **Upper:** {upp_b:.2f}")
        outliers = df[(df[col_check]<low_b)|(df[col_check]>upp_b)]; num_out = len(outliers)
        st.write(f"Outliers detected: **{num_out}**")
        if num_out > 0:
            st.markdown("#### Detected Outliers:"); st.dataframe(outliers, use_container_width=True); add_to_log(f"- {num_out} outliers in '{col_check}'.")
            st.markdown("#### Box Plot"); fig,ax=plt.subplots(); sns.boxplot(x=df[col_check],ax=ax); ax.set_title(f"Box Plot: {col_check}"); st.pyplot(fig); plt.clf()
        else: st.success("No outliers by 1.5*IQR rule."); add_to_log(f"- No outliers in '{col_check}'.")
    else: st.info("Select a column.")

def page_hypothesis_testing():
    st.header("6. Hypothesis Testing (General)")
    if st.session_state.df_cleaned is None: st.warning("Please clean data first."); return
    df = st.session_state.df_cleaned
    tabs = st.tabs(["T-test", "Chi-Square", "ANOVA"])
    num_cols_ht = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_ht = df.select_dtypes(include=['object','category']).columns.tolist()
    with tabs[0]:
        st.markdown("#### Independent T-test"); st.write("Compares means of a numerical var between 2 groups of a cat var.")
        num_var_t = st.selectbox("Numerical var", [""]+num_cols_ht, key='t_num_main'); cat_var_t = st.selectbox("Categorical var (2 groups)", [""]+cat_cols_ht, key='t_cat_main')
        if st.button("Run T-test", key="run_t_main_btn"):
            if num_var_t and cat_var_t:
                grps = df[cat_var_t].dropna().unique()
                if len(grps)==2:
                    d1=df[df[cat_var_t]==grps[0]][num_var_t].dropna(); d2=df[df[cat_var_t]==grps[1]][num_var_t].dropna()
                    if len(d1)<2 or len(d2)<2: st.error("Not enough data.")
                    else:
                        stat,p_val_ttest=ttest_ind(d1,d2,equal_var=False)
                        st.write(f"**T-stat:** {stat:.4f}, **P-val:** {p_val_ttest:.4f}")
                        if p_val_ttest<0.05:
                            st.success("Significant (p<0.05).")
                        else:
                            st.warning("Not significant (p>=0.05).")
                        add_to_log(f"- T-test: '{num_var_t}' by '{cat_var_t}'. P: {p_val_ttest:.4f}.")
                else: st.error(f"'{cat_var_t}' has {len(grps)} groups. T-test needs 2.")
            else: st.error("Select variables.")
    with tabs[1]:
        st.markdown("#### Chi-Square Test"); st.write("Tests association between 2 cat vars.")
        if len(cat_cols_ht)<2: st.info("Need >=2 cat columns.")
        else:
            c1=st.selectbox("Cat var 1",[""]+cat_cols_ht,key='chi1_main'); c2=st.selectbox("Cat var 2",[""]+cat_cols_ht,key='chi2_main')
            if st.button("Run Chi-Square",key="run_chi2_main_btn"):
                if c1 and c2 and c1!=c2:
                    try:
                        tbl=pd.crosstab(df[c1],df[c2]); st.write("Contingency:"); st.dataframe(tbl)
                        chi2,p_val_chi2,dof,exp=chi2_contingency(tbl)
                        st.write(f"**Chi2:** {chi2:.4f}, **P-val:** {p_val_chi2:.4f}, **DoF:** {dof}")
                        if p_val_chi2<0.05:
                            st.success("Significant (p<0.05).")
                        else:
                            st.warning("Not significant (p>=0.05).")
                        add_to_log(f"- Chi2: '{c1}' vs '{c2}'. P: {p_val_chi2:.4f}.")
                    except ValueError as e: st.error(f"Chi2 error: {e}.")
                else: st.error("Select 2 different vars.")
    with tabs[2]:
        st.markdown("#### ANOVA (One-Way)"); st.write("Compares means of a num var across >2 groups of a cat var.")
        num_var_a=st.selectbox("Num var",[""]+num_cols_ht,key='a_num_main'); cat_var_a=st.selectbox("Cat var (>2 groups)",[""]+cat_cols_ht,key='a_cat_main')
        if st.button("Run ANOVA",key="run_a_main_btn"):
            if num_var_a and cat_var_a:
                grps_a=df[cat_var_a].dropna().unique()
                if len(grps_a)>2:
                    samps=[df[df[cat_var_a]==g][num_var_a].dropna() for g in grps_a]; samps_f=[s for s in samps if len(s)>=2]
                    if len(samps_f)<2: st.error(f"ANOVA needs >=2 groups with >=2 samples. Found {len(samps_f)}.")
                    else:
                        f,p_val_anova=f_oneway(*samps_f)
                        st.write(f"**F-stat:** {f:.4f}, **P-val:** {p_val_anova:.4f}")
                        if p_val_anova<0.05:
                            st.success("Significant (p<0.05).")
                        else:
                            st.warning("Not significant (p>=0.05).")
                        add_to_log(f"- ANOVA: '{num_var_a}' by '{cat_var_a}'. P: {p_val_anova:.4f}.")
                else: st.error(f"'{cat_var_a}' has {len(grps_a)} groups. ANOVA needs >2.")
            else: st.error("Select variables.")

def page_modeling_advisor():
    st.header("7. AI Modeling Advisor")
    st.markdown("Ask questions about potential ML models.")
    if st.session_state.df_cleaned is None: st.warning("Please clean data first."); return
    df=st.session_state.df_cleaned; target_adv=st.session_state.get('target_variable',None)
    if not target_adv: st.error("Select target on 'Target Analysis' page."); return
    st.info(f"Target: **`{target_adv}`**")
    q_model=st.text_area("Ask modeling question",f"For target '{target_adv}', suggest models. Compare Random Forest & Logistic/Linear Regression.",height=150,key="model_q_main")
    if st.button("Get AI Modeling Advice",type="primary",key="get_model_adv_main_btn"):
        with st.spinner("AI thinking..."):
            prob_t="Unknown"
            if pd.api.types.is_numeric_dtype(df[target_adv]): prob_t="Regression"
            elif df[target_adv].nunique()/len(df)<0.5: prob_t=f"Classification ({df[target_adv].nunique()} classes)"
            ctx_sum=f"- Shape: {df.shape}\n- Target: '{target_adv}' ({df[target_adv].dtype}, {prob_t})\n- Dtypes:\n{df.dtypes.to_string()}\n- Stats:\n{df.describe(include='all').to_string()}"
            prompt_m=f"As Data Science Advisor. Context:\n{ctx_sum}\nUser Q: \"{q_model}\"\nTask: Answer. Explain why, prep steps, compare models (pros/cons for this data), conclude. Markdown."
            adv=get_gemini_explanation(prompt_m); st.markdown("### AI Recommendation"); st.markdown(adv)
            add_to_log(f"\n--- AI Model Advice (Target: {target_adv}, Q: '{q_model}') ---\n{adv}")

def page_ai_summary():
    st.header("8. AI Summary & Reporting")
    df_rep=st.session_state.cleaned_df_for_report if st.session_state.cleaned_df_for_report is not None else st.session_state.df
    if df_rep is None: st.warning("Upload data first."); return
    st.markdown("Generate AI report based on data and actions.")
    if st.button("Generate Full AI EDA Report",type="primary",key="gen_ai_final_rep_main_btn"):
        with st.spinner("AI generating report..."):
            sum_rep=f"Shape: {df_rep.shape}\nDtypes:\n{df_rep.dtypes.to_string()}\nNum Stats:\n{df_rep.describe(include=np.number).to_string()}\nCat Stats:\n{df_rep.describe(include=['object','category']).to_string()}\nMissing (Final):\n{df_rep.isnull().sum().to_string()}"
            log_rep="\n".join(st.session_state.report_log) if st.session_state.report_log else "No actions logged."
            target_rep=f"Target for Analysis: {st.session_state.target_variable}" if st.session_state.target_variable else "No target focused on."
            prompt_rep=f"Expert data scientist EDA report.\nSummary:\n{sum_rep}\n{target_rep}\nLog & Findings:\n{log_rep}\nTask: Write narrative EDA report: Intro, Quality/Cleaning, Univariate, Bi/Multivariate, Outliers, Hypothesis Tests, Modeling (if target/advice logged), Limitations, Recommendations. Markdown."
            content_rep=get_gemini_explanation(prompt_rep); st.markdown("### AI EDA Report:"); st.markdown(content_rep,unsafe_allow_html=True)
            add_to_log(f"\n--- Full AI EDA Report ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n{content_rep}")
    st.markdown("---"); st.markdown("### Download Log & Report")
    if st.session_state.report_log:
        dl_content="\n".join(st.session_state.report_log)
        st.download_button("Download Log & Reports (TXT)",dl_content,"full_eda_report.txt","text/plain",key="dl_final_rep_btn")
    else: st.info("No actions logged.")

# --- LEARNING ZONE CONTENT AND FUNCTION ---
sample_data_learning = {'col1': [10, 20, 30, 40, 50], 'col2': ['A', 'B', 'A', 'C', 'B'], 'col3': [1.1, 2.2, 3.3, 4.4, 5.5]}
sample_df_for_learning = pd.DataFrame(sample_data_learning)

learning_content = {
    "NumPy": {
        "Intro & Setup": {"explanation": "NumPy (Numerical Python) is for numerical computation, providing N-dimensional array objects, broadcasting, and tools for linear algebra, Fourier transform, etc.", "example_code": "import numpy as np", "default_code": "import numpy as np\nprint(f\"NumPy version: {np.__version__}\")"},
        "Array Creation": {"explanation": "From lists/tuples (`np.array()`), or functions like `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.random.rand()`.", "example_code": "import numpy as np\narr1 = np.array([1,2,3])\narr_zeros = np.zeros((2,3))\narr_range = np.arange(5,15,3)\nprint(arr1)\nprint(arr_zeros)\nprint(arr_range)", "default_code": "import numpy as np\n# Create a 2x4 array of random numbers between 0 and 1\nrandom_arr = np.random.rand(2,4)\nprint(random_arr)"},
        "Attributes & Shape Manipulation": {"explanation": "Attributes: `shape`, `dtype`, `ndim`, `size`. Reshape with `reshape()`.", "example_code": "import numpy as np\narr = np.arange(12)\nprint('Original:', arr)\nprint('Shape:', arr.shape)\narr_reshaped = arr.reshape(3,4)\nprint('Reshaped (3x4):\\n', arr_reshaped)", "default_code": "import numpy as np\na = np.array([[1,2],[3,4],[5,6]])\n# Print its ndim and then reshape it to 2x3\n# print(a.ndim)\n# print(a.reshape(2,3))"},
        "Indexing & Slicing": {"explanation": "Access elements like Python lists. Slicing `[start:stop:step]`. Boolean indexing.", "example_code": "import numpy as np\narr = np.arange(10,20)\nprint('Array:', arr)\nprint('Element at index 3:', arr[3])\nprint('Slice from index 2 to 5:', arr[2:6])\nprint('Elements > 15:', arr[arr > 15])", "default_code": "import numpy as np\narr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])\n# Select the element in the 2nd row, 3rd column (value 6)\n# print(arr2d[1,2])\n# Select the first two rows\n# print(arr2d[:2, :])"},
        "Universal Functions (ufuncs)": {"explanation": "Element-wise operations: `np.sqrt()`, `np.exp()`, `np.sin()`, arithmetic ops.", "example_code": "import numpy as np\narr = np.array([1, 4, 9, 16])\nprint('Sqrt:', np.sqrt(arr))\nprint('Exp:', np.exp(arr / 10))", "default_code": "import numpy as np\na = np.array([0, np.pi/2, np.pi])\n# Calculate sin of each element\n# print(np.sin(a))"},
        "Aggregation": {"explanation": "Functions like `sum()`, `mean()`, `std()`, `min()`, `max()`. Can be applied over axes.", "example_code": "import numpy as np\narr = np.array([[1,5,2],[8,3,6]])\nprint('Sum of all:', arr.sum())\nprint('Mean of columns:', arr.mean(axis=0))\nprint('Max of rows:', arr.max(axis=1))", "default_code": "import numpy as np\ndata = np.random.randint(1, 100, size=(4,5))\n# Calculate the standard deviation of the entire array\n# print(data.std())"},
    },
    "Pandas": {
        "Intro & Setup": {"explanation": "Pandas is for data analysis and manipulation, with Series (1D) and DataFrame (2D) as primary structures.", "example_code": "import pandas as pd", "default_code": "import pandas as pd\nprint(f\"Pandas version: {pd.__version__}\")"},
        "Series": {"explanation": "A 1D labeled array capable of holding any data type.", "example_code": "import pandas as pd\ns = pd.Series([10,20,30], index=['a','b','c'])\nprint(s)\nprint('Value at label b:', s['b'])", "default_code": "import pandas as pd\n# Create a Series from a dictionary\ndata = {'apple': 5, 'banana': 8, 'cherry': 3}\n# fruit_series = pd.Series(data)\n# print(fruit_series)"},
        "DataFrame Creation": {"explanation": "From dicts, lists of dicts/lists, NumPy arrays, CSVs.", "example_code": "import pandas as pd\ndata = {'colA': [1,2], 'colB': [3,4]}\ndf = pd.DataFrame(data)\nprint(df)", "default_code": "import pandas as pd\n# Create a DataFrame with 3 rows and 2 columns from random numbers\n# df_random = pd.DataFrame(np.random.rand(3,2), columns=['X','Y'])\n# print(df_random)"},
        "Viewing Data": {"explanation": "Inspect with `.head()`, `.tail()`, `.info()`, `.describe()`, `.shape`, `.columns`, `.index`.", "example_code": f"import pandas as pd\ndf = sample_df_for_learning\nprint('Info:'); df.info()\nprint('\\nDescribe:\\n', df.describe(include='all'))", "default_code": f"import pandas as pd\ndf = sample_df_for_learning\n# Show the data types of each column\n# print(df.dtypes)"},
        "Selection (loc, iloc, conditional)": {"explanation": "`.loc[]` (label-based), `.iloc[]` (integer-based), boolean indexing.", "example_code": f"import pandas as pd\ndf = sample_df_for_learning\nprint('Row index 1, col2 (loc):', df.loc[1, 'col2'])\nprint('\\nRows 0-2, col index 1 (iloc):\\n', df.iloc[0:3, 1])\nprint(\"\\nRows where col1 > 30:\\n\", df[df['col1'] > 30])", "default_code": f"import pandas as pd\ndf = sample_df_for_learning\n# Select rows where 'col2' is 'A' and show only 'col1' and 'col3'\n# print(df.loc[df['col2'] == 'A', ['col1', 'col3']])"},
        "Operations & Apply": {"explanation": "Arithmetic, string methods, `.apply()` for custom functions.", "example_code": f"import pandas as pd\ndf = sample_df_for_learning.copy()\ndf['col1_plus_100'] = df['col1'] + 100\ndf['col2_upper'] = df['col2'].str.upper()\ndef custom_func(x): return x * 2\ndf['col3_doubled'] = df['col3'].apply(custom_func)\nprint(df)", "default_code": f"import pandas as pd\ndf = sample_df_for_learning.copy()\n# Create a new column 'col1_category' which is 'High' if col1 > 25, else 'Low'\n# df['col1_category'] = df['col1'].apply(lambda x: 'High' if x > 25 else 'Low')\n# print(df)"},
        "Grouping (groupby)": {"explanation": "Split data into groups, apply a function, combine results.", "example_code": f"import pandas as pd\ndf = sample_df_for_learning\ngrouped_sum = df.groupby('col2')['col1'].sum()\nprint('Sum of col1 grouped by col2:\\n', grouped_sum)\ngrouped_mean_agg = df.groupby('col2').agg({{'col1':'mean', 'col3':'max'}})\nprint('\\nMean of col1 & Max of col3 by col2:\\n', grouped_mean_agg)", "default_code": f"import pandas as pd\ndf = sample_df_for_learning\n# Find the number of occurrences for each unique value in 'col2'\n# print(df.groupby('col2').size())"},
        "Merging & Joining": {"explanation": "`pd.merge()` for SQL-like joins, `pd.concat()` for stacking.", "example_code": "import pandas as pd\ndf1 = pd.DataFrame({{'key': ['A','B','C'], 'val1': [1,2,3]}})\ndf2 = pd.DataFrame({{'key': ['B','C','D'], 'val2': [4,5,6]}})\nmerged_inner = pd.merge(df1, df2, on='key', how='inner')\nprint('Inner Merge:\\n', merged_inner)", "default_code": "import pandas as pd\ndf_left = pd.DataFrame({{'ID': [1,2,3], 'Name': ['X','Y','Z']}})\ndf_right = pd.DataFrame({{'ID': [1,3,4], 'Score': [100,90,80]}})\n# Perform a left merge on 'ID'\n# merged_df = pd.merge(df_left, df_right, on='ID', how='left')\n# print(merged_df)"},
    },
    "Matplotlib": {
        "Intro & Setup": {"explanation": "Matplotlib for static, interactive, and animated visualizations. Pyplot makes it work like MATLAB.", "example_code": "import matplotlib.pyplot as plt", "default_code": "import matplotlib.pyplot as plt\nprint(\"Matplotlib ready!\")"},
        "Line Plot": {"explanation": "Plot lines from lists/arrays. Customize with `title`, `xlabel`, `ylabel`, `color`, `linestyle`, `marker`.", "example_code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3,4], [1,4,9,16], marker='o', linestyle='--', color='g')\nplt.title('Squared Values'); plt.xlabel('Base'); plt.ylabel('Square')", "default_code": "import matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(0, 2 * np.pi, 50)\ny1 = np.sin(x)\ny2 = np.cos(x)\n# Plot y1 and y2 on the same axes with different colors and labels\n# plt.plot(x, y1, label='sin(x)')\n# plt.plot(x, y2, label='cos(x)')\n# plt.legend()"},
        "Scatter Plot": {"explanation": "Show relationship between two numerical variables.", "example_code": "import matplotlib.pyplot as plt\nplt.scatter([1,2,3,4,5], [5,3,6,2,7], color='red', s=50) # s is size\nplt.title('Sample Scatter Plot')", "default_code": "import matplotlib.pyplot as plt\nimport numpy as np\n# Generate 50 random x and y coordinates and a random size array\nx_rand = np.random.rand(50)\ny_rand = np.random.rand(50)\nsizes = 100 * np.random.rand(50)\n# Create a scatter plot with varying sizes\n# plt.scatter(x_rand, y_rand, s=sizes, alpha=0.5) # alpha for transparency"},
        "Bar Chart": {"explanation": "Compare quantities across categories.", "example_code": "import matplotlib.pyplot as plt\ncategories = ['A','B','C']; values = [10,25,15]\nplt.bar(categories, values, color=['skyblue','lightcoral','lightgreen'])\nplt.title('Category Comparison')", "default_code": "import matplotlib.pyplot as plt\nlabels = ['G1','G2','G3','G4']\nmen_means = [20,35,30,35]\nwomen_means = [25,32,34,20]\n# Create a grouped bar chart for men and women\n# x = np.arange(len(labels))\n# width = 0.35\n# rects1 = plt.bar(x - width/2, men_means, width, label='Men')\n# rects2 = plt.bar(x + width/2, women_means, width, label='Women')\n# plt.ylabel('Scores'); plt.title('Scores by group and gender')\n# plt.xticks(x, labels); plt.legend()"},
        "Histogram": {"explanation": "Visualize distribution of a single numerical variable.", "example_code": "import matplotlib.pyplot as plt\nimport numpy as np\ndata_hist = np.random.randn(1000) # Standard normal distribution\nplt.hist(data_hist, bins=30, edgecolor='black')\nplt.title('Data Distribution')", "default_code": "import matplotlib.pyplot as plt\nimport numpy as np\n# Generate 200 data points from a gamma distribution\n# gamma_data = np.random.gamma(2., 2., 200)\n# plt.hist(gamma_data, bins=20, color='purple', alpha=0.7)\n# plt.title('Gamma Distribution')"},
        "Subplots": {"explanation": "`plt.subplot()` or `plt.subplots()` to create multiple plots in one figure.", "example_code": "import matplotlib.pyplot as plt\nimport numpy as np\nx_sub = np.arange(5)\nplt.figure(figsize=(10,4))\nplt.subplot(1,2,1); plt.plot(x_sub, x_sub**2); plt.title('y = x^2')\nplt.subplot(1,2,2); plt.plot(x_sub, x_sub**3); plt.title('y = x^3')", "default_code": "import matplotlib.pyplot as plt\nimport numpy as np\n# Create a 2x2 grid of subplots\n# Plot different functions (e.g., sin, cos, tan, exp) in each subplot\n# fig, axs = plt.subplots(2, 2, figsize=(8,8))\n# x = np.linspace(0,10,100)\n# axs[0,0].plot(x, np.sin(x)); axs[0,0].set_title('Sin')\n# ... fill other subplots"},
    },
    "Seaborn": {
        "Intro & Setup": {"explanation": "Seaborn, based on Matplotlib, for attractive statistical graphics. Works well with Pandas.", "example_code": "import seaborn as sns\nimport matplotlib.pyplot as plt", "default_code": "import seaborn as sns\nimport pandas as pd\nprint(f\"Seaborn version: {sns.__version__}\")\ndf_sb = pd.DataFrame({'x':range(5),'y':[i*2 for i in range(5)],'cat':['A','B','A','B','A']})\n# print(df_sb)"},
        "Distribution Plots": {"explanation": "`histplot`, `kdeplot` (Kernel Density Estimate), `displot` (figure-level for distributions), `boxplot`, `violinplot`.", "example_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\ndf = sample_df_for_learning\nsns.histplot(data=df, x='col1', hue='col2', kde=True)\nplt.title('Histogram of col1 by col2')", "default_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\ndf = sample_df_for_learning\n# Create a violin plot of 'col3' grouped by 'col2'\n# sns.violinplot(data=df, x='col2', y='col3')\n# plt.title('Violin Plot')"},
        "Categorical Plots": {"explanation": "`barplot`, `countplot`, `boxplot` (for categorical x), `stripplot`, `swarmplot`.", "example_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\ndf = sample_df_for_learning\nsns.countplot(data=df, x='col2', palette='viridis')\nplt.title('Count of Categories in col2')", "default_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\ndf = sample_df_for_learning\n# Create a bar plot showing the mean of 'col1' for each category in 'col2'\n# sns.barplot(data=df, x='col2', y='col1', estimator=np.mean)\n# plt.title('Mean of col1 by col2')"},
        "Relational Plots": {"explanation": "`scatterplot` (can include hue, size, style), `lineplot` (for trends, often with time).", "example_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\ndf = sample_df_for_learning\nsns.scatterplot(data=df, x='col1', y='col3', hue='col2', size='col1', sizes=(50,200))\nplt.title('Scatter of col1 vs col3')", "default_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd\ntime_data = pd.DataFrame({{'time': range(10), 'value': np.random.randn(10).cumsum()}})\n# Create a line plot of 'value' over 'time'\n# sns.lineplot(data=time_data, x='time', y='value', marker='o')\n# plt.title('Time Series Trend')"},
        "Matrix Plots": {"explanation": "`heatmap` (for correlation matrices or other grid data), `clustermap`.", "example_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\nimport numpy as np\n# Create a sample correlation matrix\ncorr_data = np.corrcoef(np.random.rand(5,5))\nsns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Sample Heatmap')", "default_code": f"import seaborn as sns\nimport matplotlib.pyplot as plt\ndf = sample_df_for_learning.copy()\n# Calculate the correlation matrix of the numerical columns in sample_df_for_learning\n# num_df = df.select_dtypes(include=np.number)\n# if not num_df.empty:\n#    sns.heatmap(num_df.corr(), annot=True, cmap='magma')\n#    plt.title('Correlation Heatmap')\n# else:\n#    print('No numerical columns in sample_df_for_learning for heatmap.')"},
    }
}

def execute_user_code(user_code, setup_code="", is_plot=False):
    """Executes user code, captures output/plots."""
    full_code = setup_code + "\n" + user_code
    stdout_capture = io.StringIO()
    fig_object = None
    error_message = None
    local_namespace = {}
    # Make necessary modules and sample data available in the execution scope
    global_namespace = {
        'np': np, 
        'pd': pd, 
        'plt': plt, 
        'sns': sns, 
        'sample_df_for_learning': sample_df_for_learning.copy() # Pass a copy to avoid modification
    }

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(full_code, global_namespace, local_namespace)
        if is_plot:
            fig_object = plt.gcf() # Get current figure
            # Check if figure has any axes, meaning something was plotted
            if not fig_object.get_axes(): 
                fig_object = None # No actual plot was made
            else:
                # If a plot was made, ensure it's not empty
                is_empty_plot = all(not ax.has_data() for ax in fig_object.get_axes())
                if is_empty_plot:
                    fig_object = None


    except Exception as e:
        error_message = f"Error executing code:\n{type(e).__name__}: {e}"
    
    output_text = stdout_capture.getvalue()
    stdout_capture.close()
    
    return output_text, fig_object, error_message


def page_learning_zone():
    """Displays the AI Learning Zone for Python data science libraries."""
    st.header("📚 AI Learning Zone")
    st.markdown("Interactive tutorials for NumPy, Pandas, Matplotlib, and Seaborn. Experiment with code and see results instantly!")
    st.warning("⚠️ **Note:** The code execution environment is for learning purposes. Avoid running complex or potentially harmful code.")

    lib_tabs = st.tabs(["NumPy", "Pandas", "Matplotlib", "Seaborn"])
    lib_names = ["NumPy", "Pandas", "Matplotlib", "Seaborn"]

    for i, lib_name in enumerate(lib_names):
        with lib_tabs[i]:
            st.subheader(f"Learn {lib_name}")
            
            topics = learning_content[lib_name]
            selected_topic_name = st.selectbox(f"Select a {lib_name} Topic:", list(topics.keys()), key=f"select_{lib_name}")

            if selected_topic_name:
                topic_data = topics[selected_topic_name]
                st.markdown(f"#### {selected_topic_name}")
                if isinstance(topic_data["explanation"], list):
                    for item in topic_data["explanation"]:
                        st.markdown(item)
                else:
                    st.markdown(topic_data["explanation"])
                
                st.markdown("##### Example Code:")
                st.code(topic_data["example_code"], language="python")

                st.markdown("##### Try it Yourself!")
                
                # Use a unique key for each text_area based on library and topic
                code_editor_key = f"user_code_{lib_name}_{selected_topic_name.replace(' ','_').replace('&','_').replace('(','_').replace(')','_')}"
                
                # Initialize text area content from default_code or previous entry
                if code_editor_key not in st.session_state:
                    st.session_state[code_editor_key] = topic_data.get("default_code", f"# Write your {lib_name} code here\n")
                
                user_code_input = st.text_area(
                    "Your Code:", 
                    value=st.session_state[code_editor_key], 
                    height=250, 
                    key=code_editor_key # This ensures state is preserved per topic
                )

                if st.button("Run Code", key=f"run_btn_{lib_name}_{selected_topic_name.replace(' ','_').replace('&','_').replace('(','_').replace(')','_')}"):
                    is_plot_expected = lib_name in ["Matplotlib", "Seaborn"]
                    
                    # Base imports always included for convenience
                    base_setup_code = "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
                    # Add sample DataFrame for Pandas examples
                    if lib_name == "Pandas":
                         base_setup_code += f"sample_data_learning = {sample_data_learning}\nsample_df_for_learning = pd.DataFrame(sample_data_learning)\n"
                    
                    # Add topic-specific setup code if any
                    topic_setup_code = topic_data.get("setup_code", "")
                    final_setup_code = base_setup_code + topic_setup_code

                    text_output, plot_output, error_output = execute_user_code(user_code_input, setup_code=final_setup_code, is_plot=is_plot_expected)
                    
                    st.markdown("##### Output:")
                    if error_output:
                        st.error(error_output)
                    if text_output:
                        # Use st.text for preformatted text, or st.markdown if it contains markdown
                        st.text(text_output) 
                    if plot_output:
                        st.pyplot(plot_output)
                        plt.clf() # Clear the figure after displaying to avoid overlap with subsequent plots
                    
                    # Handle cases where no explicit output is generated
                    if not error_output and not text_output and not plot_output:
                        if is_plot_expected:
                            st.info("No plot was generated, or the plot was empty. Make sure your Matplotlib/Seaborn code actively creates a figure (e.g., using `plt.plot()` or `sns.histplot()`). Forgetting `plt.show()` is fine here as Streamlit handles display.")
                        else:
                            st.info("Code executed. No text output (e.g., from print statements) was captured or no error occurred.")


# --- Main App ---
st.sidebar.title("🌟 Gen AI EDA Tool")
st.sidebar.markdown("Your comprehensive EDA assistant.")

uploaded_file_main = st.sidebar.file_uploader("Upload your CSV file", type="csv", key="main_csv_uploader")

if uploaded_file_main is not None:
    if st.session_state.uploaded_filename != uploaded_file_main.name: # Check if it's a new file
        try:
            st.sidebar.info("New file detected. Loading data...")
            df_load = pd.read_csv(uploaded_file_main)
            # Reset all relevant session state variables for the new file
            st.session_state.df = df_load.copy()
            st.session_state.df_cleaned = df_load.copy() 
            st.session_state.cleaned_df_for_report = df_load.copy()
            st.session_state.target_variable = None 
            st.session_state.report_log = [f"- Dataset '{uploaded_file_main.name}' loaded: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}."]
            st.session_state.uploaded_filename = uploaded_file_main.name # Update the stored filename
            st.sidebar.success(f"'{uploaded_file_main.name}' loaded!")
            st.rerun() # Rerun to reflect the new state immediately and clear old data from view
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            # Clear potentially problematic state variables on error
            keys_to_clear_on_error = ['df', 'df_cleaned', 'cleaned_df_for_report', 'target_variable', 'report_log', 'uploaded_filename']
            for key_to_clear in keys_to_clear_on_error:
                if key_to_clear in st.session_state: del st.session_state[key_to_clear]
else:
    # This part handles the case where the uploader might be cleared by the user
    # or if no file was uploaded initially. We don't want to lose state if a file was previously processed.
    pass

# Navigation logic based on whether a DataFrame is loaded
eda_pages_available = 'df' in st.session_state and st.session_state.df is not None

if eda_pages_available:
    page_options_main = [
        "1. Data Overview", "2. Data Cleaning & Preprocessing", "3. Data Visualization (General)",
        "4. Target Variable Analysis", "5. Outlier Detection", "6. Hypothesis Testing (General)",
        "7. AI Modeling Advisor", "8. AI Summary & Reporting", "📚 AI Learning Zone"
    ]
    default_index_eda = 0 # Default to Data Overview if df is loaded
else:
    page_options_main = ["📚 AI Learning Zone"] # Only Learning Zone if no df
    default_index_eda = 0 # Default to Learning Zone

page_selected = st.sidebar.radio(
    "Select Option:", 
    page_options_main, 
    index=default_index_eda, 
    key="main_page_nav_radio"
)

# Routing to the selected page
if page_selected == "1. Data Overview": page_data_overview()
elif page_selected == "2. Data Cleaning & Preprocessing": page_data_cleaning()
elif page_selected == "3. Data Visualization (General)": page_visualization()
elif page_selected == "4. Target Variable Analysis": page_target_analysis()
elif page_selected == "5. Outlier Detection": page_outlier_detection()
elif page_selected == "6. Hypothesis Testing (General)": page_hypothesis_testing()
elif page_selected == "7. AI Modeling Advisor": page_modeling_advisor()
elif page_selected == "8. AI Summary & Reporting": page_ai_summary()
elif page_selected == "📚 AI Learning Zone": page_learning_zone()

# Message if trying to access EDA pages without data
if not eda_pages_available and page_selected != "📚 AI Learning Zone":
     st.info("⬆️ Upload a CSV file using the sidebar to access full EDA features, or explore the AI Learning Zone.")
