import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def show_preprocessing():
    st.title("Step 2: Pre-processing & Feature Engineering")

    if st.session_state["raw_data"] is None:
        st.warning("Please go to Data Ingestion and select a dataset first.")
        return

    df = st.session_state["raw_data"].copy()
    target_col = st.session_state["target_col"]

    st.markdown("""
    **Pre-processing and Feature Engineering** involves cleaning our data and transforming it into a format that a Machine Learning model can understand. This often includes handling missing values, scaling numbers so they are comparable, and encoding text/categories into numbers.
    """)

    with st.container(border=True):
        # Handling Missing Values
        st.header("Handling Missing Values")
        st.markdown("Real-world data is messy. It often contains gaps or 'Null' values. Let's see if we have any, and if not, let's artificially introduce some for learning purposes!")

        # Check for missing values
        missing_data = df.isnull().sum()

        if missing_data.sum() == 0:
            st.info("Our dataset is perfectly clean! To demonstrate imputation, we'll randomly remove 10% of the data from one of the columns.")

            # Introduce missing values in the first numeric feature column
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if target_col in num_cols:
                num_cols.remove(target_col)

            col_to_corrupt = num_cols[0]

            # Corrupt 10%
            np.random.seed(42)
            mask = np.random.rand(len(df)) < 0.1
            df.loc[mask, col_to_corrupt] = np.nan

        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({'Column': missing_data.index, 'Missing Count': missing_data.values})
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.markdown("### Missing Values Code")
                code = f'''import pandas as pd
from sklearn.impute import SimpleImputer

# Find columns with missing data
missing_cols = df.isnull().sum()
print(missing_cols[missing_cols > 0])

# Impute missing values using the mean (average)
imputer = SimpleImputer(strategy='mean')
df['{missing_df.iloc[0]['Column']}'] = imputer.fit_transform(df[['{missing_df.iloc[0]['Column']}']])
'''
                st.code(code, language='python')

        with col2:
            with st.container(border=True):
                st.markdown("### Visual Output")
                fig_missing = px.bar(missing_df, x='Column', y='Missing Count',
                                     title="Missing Values Count (Before Imputation)",
                                     color_discrete_sequence=['#ef4444'])
                st.plotly_chart(fig_missing, use_container_width=True)

                if st.button("Apply Missing Value Imputation"):
                    imputer = SimpleImputer(strategy='mean')
                    for col in missing_df['Column']:
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = imputer.fit_transform(df[[col]])
                    st.success("Missing values imputed successfully using the mean.")

                    # Show that there are no more missing values
                    new_missing = pd.DataFrame({'Column': df.columns, 'Missing Count': df.isnull().sum()})
                    new_missing = new_missing[new_missing['Missing Count'] > 0]

                    if new_missing.empty:
                        st.info("No more missing values remaining.")
                    else:
                        fig_missing_after = px.bar(new_missing, x='Column', y='Missing Count',
                                             title="Missing Values Count (After Imputation)",
                                             color_discrete_sequence=['#10b981'])
                        st.plotly_chart(fig_missing_after, use_container_width=True)

    st.markdown("---")

    with st.container(border=True):
        # Categorical Encoding
        st.header("Encoding Categorical Variables (One-Hot Encoding)")
        st.markdown("Machine learning models require numbers, not text. We use **One-Hot Encoding** to convert categorical text columns into multiple binary (0 or 1) columns.")

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        if not cat_cols:
            st.info("No categorical features found in this dataset (excluding the target). This step is not needed.")
        else:
            selected_cat = st.selectbox("Select a categorical feature to encode:", cat_cols)

            col3, col4 = st.columns([1, 1])

            with col3:
                with st.container(border=True):
                    st.markdown("### Encoding Code")
                    code = f'''import pandas as pd

# Original column: {selected_cat}
# Apply One-Hot Encoding
df = pd.get_dummies(df, columns=['{selected_cat}'], drop_first=True)
'''
                    st.code(code, language='python')

            with col4:
                with st.container(border=True):
                    st.markdown("### Visual Output")

                    # Show distribution before encoding
                    val_counts = df[selected_cat].value_counts().reset_index()
                    val_counts.columns = [selected_cat, 'Count']
                    fig = px.bar(val_counts, x=selected_cat, y='Count', title=f"Distribution of {selected_cat} (Before Encoding)")
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button("Apply One-Hot Encoding"):
                        df = pd.get_dummies(df, columns=[selected_cat], drop_first=True)
                        st.success(f"Column '{selected_cat}' has been one-hot encoded.")

                        # Show new columns
                        new_cols = [c for c in df.columns if selected_cat in c]
                        st.write("Here are the new binary columns created:")
                        st.dataframe(df[new_cols].head(), use_container_width=True)

    st.markdown("---")

    with st.container(border=True):
        # Feature Scaling
        st.header("Feature Scaling (Standardization)")
        st.markdown("Features often have different scales (e.g., Age ranges from 0-100, while Salary ranges from 30,000-200,000). Many ML algorithms perform better when all numerical features are scaled to have a similar range. **Standardization** shifts the data to have a mean of 0 and standard deviation of 1.")

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        if not num_cols:
            st.warning("No numerical columns found to scale.")
        else:
            selected_num = st.selectbox("Select a numerical feature to scale:", num_cols)

            col5, col6 = st.columns([1, 1])

            with col5:
                with st.container(border=True):
                    st.markdown("### Scaling Code")
                    code = f'''from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit and transform the selected column
df['{selected_num}_scaled'] = scaler.fit_transform(df[['{selected_num}']])
'''
                    st.code(code, language='python')

            with col6:
                with st.container(border=True):
                    st.markdown("### Visual Output")

                    # Before scaling
                    fig1 = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num} (Before Scaling)", nbins=30)
                    st.plotly_chart(fig1, use_container_width=True)

                    if st.button("Apply Standardization"):
                        scaler = StandardScaler()
                        scaled_vals = scaler.fit_transform(df[[selected_num]])
                        df[f'{selected_num}_scaled'] = scaled_vals
                        st.success(f"Column '{selected_num}' has been scaled.")

                        # After scaling
                        fig2 = px.histogram(df, x=f'{selected_num}_scaled', title=f"Distribution of {selected_num} (After Scaling)", nbins=30, color_discrete_sequence=['#10b981'])
                        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    with st.container(border=True):
        st.header("Apply and Finalize Full Pipeline")
        st.markdown("""
            In real Machine Learning projects, we build automated pipelines that perform **all** the above steps simultaneously on the entire dataset.
            Click the button below to cleanly apply Imputation, Encoding, and Scaling across all features instantly to prepare the data for Modeling.
        """)

        if st.button("Finalize All Pre-processing Pipeline", type="primary", use_container_width=True):
            # Auto-apply all transformations for the rest of the app seamlessly
            final_df = st.session_state["raw_data"].copy()

            # 1. Impute
            num_cols_f = final_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if target_col in num_cols_f:
                num_cols_f.remove(target_col)

            imputer = SimpleImputer(strategy='mean')
            final_df[num_cols_f] = imputer.fit_transform(final_df[num_cols_f])

            # 2. Encode Categoricals
            cat_cols_f = final_df.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_col in cat_cols_f:
                cat_cols_f.remove(target_col)

            if cat_cols_f:
                final_df = pd.get_dummies(final_df, columns=cat_cols_f, drop_first=True)

            # 3. Scale Numericals
            scaler = StandardScaler()
            final_df[num_cols_f] = scaler.fit_transform(final_df[num_cols_f])

            st.session_state["processed_data"] = final_df
            st.session_state["pipeline_scaler"] = scaler
            st.session_state["num_cols_pipeline"] = num_cols_f
            st.session_state["cat_cols_pipeline"] = cat_cols_f

            # For deployment, we need to know what dummy columns were created
            features_after_encode = [c for c in final_df.columns if c != target_col]
            st.session_state["encoded_columns"] = features_after_encode

            st.success("Pre-processing Pipeline Applied. The cleaned data is now ready for the **Modeling** phase.")
            st.dataframe(final_df.head(), use_container_width=True)

    st.markdown("---")
    with st.expander("Test Your Understanding"):
        st.markdown("### Conceptual Question")
        q2 = st.radio(
            "Why do we use One-Hot Encoding for categorical features rather than just assigning a number to each category (e.g., Apple=1, Banana=2, Orange=3)?",
            options=[
                "A) Because computers only understand 1s and 0s.",
                "B) To avoid implying a mathematical or ordinal relationship (like Banana is 'greater' than Apple) where none exists.",
                "C) It reduces the number of columns in the dataset.",
                "D) Models automatically assume any text column is categorical, so we don't strictly need to encode it."
            ],
            index=None
        )
        if q2:
            if q2.startswith("B"):
                st.success("Correct! Label encoding assigns arbitrary numerical order, confusing the algorithm. One-Hot encoding treats them independently.")
            else:
                st.error("Not quite. The primary reason is to prevent the model from misinterpreting arbitrary numbers as ordered values.")
