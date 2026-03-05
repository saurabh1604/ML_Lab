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
    st.title("📥 Step 1: Data Ingestion")

    st.markdown("""
    **Data Ingestion** is the very first step of any Machine Learning pipeline. This is where we load our raw data from a source (like a CSV file, a database, or an API) into our programming environment so we can begin exploring it.
    """)

    st.header("1. Select a Dataset")
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

        st.success(f"Successfully loaded the **{dataset_name}** dataset!")

        st.header("2. View the Raw Data")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Python Code")
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
            st.markdown("### Visual Output")
            st.write(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
            st.dataframe(df.head(10), use_container_width=True)

        st.header("3. Basic Exploratory Data Analysis (EDA)")
        st.markdown(f"The target variable we are trying to predict is **`{target_col}`**.")

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
                fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}", nbins=50)
                st.plotly_chart(fig, use_container_width=True)
            else:
                class_counts = df[target_col].value_counts().reset_index()
                class_counts.columns = [target_col, 'count']
                fig = px.bar(class_counts, x=target_col, y='count', color=target_col,
                             title=f"Class Balance of {target_col}")
                st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Tip**: In regression, we want to predict a continuous number (like price). In classification, we predict a category (like Benign or Malignant). Now that our data is loaded, let's move on to Step 2: Pre-processing & Feature Engineering!")
