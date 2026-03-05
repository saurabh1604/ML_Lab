import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
import plotly.express as px

@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Ames Housing (Regression)":
        # Load Ames Housing dataset from OpenML
        housing = fetch_openml(name="house_prices", as_frame=True, version=1, parser='auto')
        df = housing.frame

        # Taking a smaller subset of columns to make visualizations cleaner
        # Focusing on the most important numerical and categorical columns for intuition
        cols_to_keep = ['LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea',
                        'FullBath', 'GarageArea', 'MSZoning', 'Neighborhood', 'BldgType',
                        'HouseStyle', 'SalePrice']

        # It's possible some columns differ slightly in older versions, safe check:
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        df = df[cols_to_keep].copy()
        target_col = 'SalePrice'
        problem_type = 'regression'

        return df, target_col, problem_type

    elif dataset_name == "Breast Cancer (Classification)":
        data = load_breast_cancer(as_frame=True)
        df = data.frame

        # Rename target for clarity
        df.rename(columns={'target': 'Diagnosis'}, inplace=True)
        # Breast Cancer has many features, let's select a few for cleaner UI
        cols_to_keep = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                        'mean smoothness', 'mean compactness', 'Diagnosis']
        df = df[cols_to_keep].copy()

        # Map target to strings for better visual understanding
        df['Diagnosis'] = df['Diagnosis'].map({0: 'Malignant', 1: 'Benign'})
        target_col = 'Diagnosis'
        problem_type = 'classification'

        return df, target_col, problem_type

def show_data_ingestion():
    st.title("Step 1: Data Ingestion")

    st.markdown("""
    **Data Ingestion** is the very first step of any Machine Learning pipeline. This is where we load our raw data from a source (like a CSV file, a database, or an API) into our programming environment so we can begin exploring it.
    """)

    with st.container(border=True):
        st.header("Select a Dataset")
        dataset_name = st.selectbox(
            "Choose a dataset to explore:",
            ["Select a Dataset...", "Ames Housing (Regression)", "Breast Cancer (Classification)"]
        )

    if dataset_name != "Select a Dataset...":
        df, target_col, problem_type = load_data(dataset_name)

        # Save to session state
        st.session_state["dataset_name"] = dataset_name
        st.session_state["raw_data"] = df
        st.session_state["target_col"] = target_col
        st.session_state["problem_type"] = problem_type
        st.session_state["features"] = [col for col in df.columns if col != target_col]

        st.success(f"Successfully loaded the **{dataset_name}** dataset.")

        # Quick Overview Metrics
        st.markdown("### Dataset Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", f"{df.shape[1]}")
        m3.metric("Target Variable", target_col)
        m4.metric("Problem Type", problem_type.title())

        st.header("View the Raw Data")

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.markdown("### Python Source Code")
                st.markdown("Here is the Python code using the `pandas` library to load and inspect our dataset.")
                code = '''import pandas as pd

# Load the dataset
df = pd.read_csv("dataset.csv")

# Display the first 5 rows
print(df.head())

# Get the shape of the dataset (rows, columns)
print(df.shape)'''
                st.code(code, language='python')

        with col2:
            with st.container(border=True):
                st.markdown("### Visual Output")
                st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown(f"The target variable we are trying to predict is **`{target_col}`**.")

        with st.container(border=True):
            col3, col4 = st.columns([1, 1])

            with col3:
                st.markdown("### Code: Analyzing the Target")
                if problem_type == "regression":
                    eda_code = f'''import plotly.express as px

# Create a histogram to see the distribution of the target variable
fig = px.histogram(df, x="{target_col}",
                   title="Distribution of {target_col}")
fig.show()'''
                else:
                    eda_code = f'''import plotly.express as px

# Create a bar chart to see the class balance
class_counts = df["{target_col}"].value_counts().reset_index()
fig = px.bar(class_counts, x="{target_col}", y="count",
             title="Class Balance of {target_col}",
             color="{target_col}")
fig.show()'''
                st.code(eda_code, language='python')

            with col4:
                st.markdown("### Visual Output")
                if problem_type == "regression":
                    fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}", nbins=50,
                                       color_discrete_sequence=['#3b82f6'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    class_counts = df[target_col].value_counts().reset_index()
                    class_counts.columns = [target_col, 'count']
                    fig = px.bar(class_counts, x=target_col, y='count', color=target_col,
                                 title=f"Class Balance of {target_col}",
                                 color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("Advanced Exploratory Data Analysis")
        st.markdown("Understanding relationships between features is crucial. Let's explore correlations and distributions.")

        tab1, tab2 = st.tabs(["Correlation Heatmap", "Feature Distributions"])

        with tab1:
            st.markdown("A **Correlation Heatmap** shows how strongly numerical features are related to one another. Values close to 1 or -1 indicate a strong relationship, while values near 0 indicate weak or no relationship.")
            num_df = df.select_dtypes(include=['float64', 'int64'])
            if not num_df.empty:
                corr = num_df.corr()
                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                     color_continuous_scale='RdBu_r',
                                     title="Feature Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No numerical features available for correlation.")

        with tab2:
            st.markdown("A **Box Plot** helps visualize the distribution of a numerical feature and identify potential outliers.")
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if num_cols:
                selected_feature = st.selectbox("Select a feature to visualize:", num_cols, key="eda_boxplot_feat")
                if problem_type == "classification":
                    fig_box = px.box(df, x=target_col, y=selected_feature, color=target_col,
                                     title=f"{selected_feature} Distribution by {target_col}",
                                     color_discrete_sequence=['#3b82f6', '#10b981'])
                else:
                    fig_box = px.box(df, y=selected_feature,
                                     title=f"Distribution of {selected_feature}",
                                     color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No numerical features available for box plots.")

        st.info("Tip: In regression, we want to predict a continuous number (like price). In classification, we predict a category (like Benign or Malignant). Now that our data is loaded, let's move on to Step 2: Pre-processing & Feature Engineering.")

        st.markdown("---")
        with st.expander("Test Your Understanding"):
            st.markdown("### Conceptual Question")
            q1 = st.radio(
                "Why is it important to perform Exploratory Data Analysis (EDA) before modeling?",
                options=[
                    "A) To immediately start training the most complex model.",
                    "B) To understand data distributions, find missing values, and discover relationships between features.",
                    "C) To deploy the model to production.",
                    "D) To increase the size of the dataset."
                ],
                index=None
            )
            if q1:
                if q1.startswith("B"):
                    st.success("Correct! EDA helps us understand the shape and quality of our data, which dictates our pre-processing steps.")
                else:
                    st.error("Not quite. EDA is about understanding the data before we even think about modeling or deployment.")
