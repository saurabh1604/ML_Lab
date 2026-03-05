import streamlit as st
from data_ingestion import show_data_ingestion
from preprocessing import show_preprocessing
from modeling import show_modeling
from deployment import show_deployment

def main():
    st.set_page_config(
        page_title="Visual ML Pipeline",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject deeply custom, production-grade CSS for a sleek, corporate ML dashboard look
    st.markdown("""
        <style>
        /* Global Backgrounds */
        .stApp {
            background-color: #f8fafc; /* Very light slate for main body */
        }
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        /* Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1 {
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.05em;
            margin-bottom: 1.5rem !important;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }
        h2 {
            color: #1e293b;
            font-weight: 600;
            letter-spacing: -0.025em;
            margin-top: 2rem !important;
        }
        h3 {
            color: #334155;
            font-weight: 600;
            font-size: 1.25rem;
        }
        p, li {
            color: #475569;
            line-height: 1.6;
        }

        /* Containers (Cards) */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
            border: 1px solid #cbd5e1;
            background-color: #ffffff;
            color: #0f172a;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .stButton>button:hover {
            border-color: #94a3b8;
            background-color: #f8fafc;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        /* Primary Buttons */
        .stButton>button[data-baseweb="button"] {
            /* Handled dynamically by Streamlit for type="primary" */
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #2563eb !important;
            color: white !important;
            border: none !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #1d4ed8 !important;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 700;
            font-size: 2.25rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.875rem;
        }

        /* Code blocks */
        .stCodeBlock {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #1e293b;
            background-color: #f1f5f9;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }

        /* Sidebar Nav Styling */
        .stRadio > div[role="radiogroup"] > label {
            background-color: transparent !important;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 0.25rem;
            transition: background-color 0.2s;
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            background-color: #f1f5f9 !important;
        }
        </style>
    """, unsafe_allow_html=True)

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

    st.sidebar.title("ML Pipeline Guide")
    st.sidebar.markdown("Interactive Machine Learning pipeline from data ingestion to deployment.")

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
