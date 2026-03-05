import streamlit as st
from data_ingestion import show_data_ingestion
from preprocessing import show_preprocessing
from modeling import show_modeling
from deployment import show_deployment

def main():
    st.set_page_config(
        page_title="Visual ML Pipeline",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for passing data between steps
    if "dataset_name" not in st.session_state:
        st.session_state["dataset_name"] = None
    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = None
    if "processed_data" not in st.session_state:
        st.session_state["processed_data"] = None
    if "target_col" not in st.session_state:
        st.session_state["target_col"] = None
    if "problem_type" not in st.session_state:
        st.session_state["problem_type"] = None
    if "model" not in st.session_state:
        st.session_state["model"] = None
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = None
    if "X_train" not in st.session_state:
        st.session_state["X_train"] = None
    if "y_train" not in st.session_state:
        st.session_state["y_train"] = None
    if "X_test" not in st.session_state:
        st.session_state["X_test"] = None
    if "y_test" not in st.session_state:
        st.session_state["y_test"] = None
    if "features" not in st.session_state:
        st.session_state["features"] = None

    st.sidebar.title("🧠 ML Pipeline Guide")
    st.sidebar.markdown("Learn the entire Machine Learning pipeline visually, from data ingestion to deployment.")

    stages = [
        "Data Ingestion",
        "Pre-processing & Feature Eng",
        "Modeling",
        "Deployment"
    ]

    choice = st.sidebar.radio("Navigate Pipeline:", stages)

    st.sidebar.markdown("---")
    if st.session_state["dataset_name"]:
        st.sidebar.success(f"**Current Dataset:** {st.session_state['dataset_name']}")
    else:
        st.sidebar.warning("No dataset selected. Start at Step 1.")

    if choice == "Data Ingestion":
        show_data_ingestion()
    elif choice == "Pre-processing & Feature Eng":
        show_preprocessing()
    elif choice == "Modeling":
        show_modeling()
    elif choice == "Deployment":
        show_deployment()

if __name__ == "__main__":
    main()
