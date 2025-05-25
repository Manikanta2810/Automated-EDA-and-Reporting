import streamlit as st
import os # Make sure os is imported

# <========================= TEMPORARY DEBUG CODE START =========================>
st.subheader("Environment Variable Debugging Info:")
api_key_value_from_env = os.getenv("GEMINI_API_KEY")

if api_key_value_from_env:
    st.write("GEMINI_API_KEY was found in environment variables.")
    # For security, let's not display the full key, just its presence and length or first/last few chars
    st.write(f"Length of key found: {len(api_key_value_from_env)}")
    st.write(f"First 5 chars: {api_key_value_from_env[:5]}")
    st.write(f"Last 5 chars: {api_key_value_from_env[-5:]}")
else:
    st.error("GEMINI_API_KEY was NOT found in environment variables by os.getenv().")

# Optionally, list all environment variables to see what's available (BE CAREFUL - MIGHT EXPOSE OTHER SENSITIVE INFO)
# This is for extreme debugging; remove after checking.
# show_all_vars = st.checkbox("Show all environment variables (for admin debug)")
# if show_all_vars:
# st.text("All Environment Variables (Be careful with this data):")
# st.json(dict(os.environ))
st.markdown("---") # Separator
# <========================= TEMPORARY DEBUG CODE END ===========================>

# ... (rest of your existing imports and code)
# from dotenv import load_dotenv # Not needed for Streamlit Cloud deployment if secrets are set
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
# from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# load_dotenv() # This line is mainly for local development
