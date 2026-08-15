import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google import genai
from dotenv import load_dotenv
import os

from streamlit_mic_recorder import speech_to_text
# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )
else:
    client = None


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---- App shell ---- */

.stApp {
    background-color: #F4F6FA;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}

[data-testid="stVerticalBlock"] > div {
    gap: 0.35rem;
}

hr {
    margin: 0.6rem 0;
}

h1, h2, h3 {
    margin-top: 0.4rem;
    margin-bottom: 0.35rem;
}

/* ---- Sidebar ---- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2563EB 100%);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
    color: #E8EEF7 !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
    font-weight: 600;
    font-size: 0.9rem;
}

.sidebar-title {
    color: #FFFFFF;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: 0.02em;
}

.sidebar-description {
    color: #D6E4FF;
    text-align: center;
    font-size: 0.88rem;
    line-height: 1.4;
    margin: 0 0 0.75rem 0;
}

.sidebar-section-label {
    color: #FFFFFF;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0.75rem 0 0.35rem 0;
}

.default-dataset-card {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 10px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.75rem;
}

.default-dataset-card p {
    color: #E8EEF7;
    font-size: 0.82rem;
    margin: 0 0 0.5rem 0;
    line-height: 1.35;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background-color: #0F172A;
    color: white;
    border: 1px solid #0F172A;
    border-radius: 8px;
    min-height: 42px;
    font-size: 0.95rem;
    font-weight: 600;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background-color: #020617;
    border-color: #020617;
    color: white;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    border: 1px solid #059669;
    box-shadow: 0 2px 8px rgba(5, 150, 105, 0.35);
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%);
    border-color: #047857;
    color: white;
}

/* ---- Header ---- */

.dashboard-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    border-radius: 12px;
    padding: 1.35rem 1.25rem 1.15rem 1.25rem;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 14px rgba(30, 58, 95, 0.18);
}

.main-title {
    text-align: center;
    color: #FFFFFF;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.3;
    padding-top: 0.15rem;
}

.main-subtitle {
    text-align: center;
    color: #DBEAFE;
    font-size: 0.95rem;
    margin: 0.35rem 0 0 0;
    line-height: 1.4;
}

/* ---- Cards & sections ---- */

.upload-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
}

.upload-card-title {
    color: #1E293B;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.15rem 0;
}

.upload-card-help {
    color: #64748B;
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.45;
}

.status-badge {
    display: inline-block;
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

.section-title {
    color: #1E293B;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0.5rem 0 0.25rem 0;
    padding-bottom: 0.15rem;
    border-bottom: 2px solid #2563EB;
    display: inline-block;
}

.dashboard-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.35rem;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
}

.plot-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    margin: 0.35rem 0;
}

/* ---- Widgets ---- */

[data-testid="stFileUploader"] {
    background: #F8FAFC;
    border: 2px dashed #93C5FD;
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
}

[data-testid="stFileUploader"] label {
    font-weight: 600 !important;
    color: #1E293B !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.45rem 0.65rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    color: #64748B !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    color: #1E293B !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.stDownloadButton > button,
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background-color: #2563EB;
    color: white;
    border: 1px solid #2563EB;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton > button:hover,
.stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. DEFAULT DATASET
# ============================================================

default_df = pd.DataFrame({

    "case_id": [
        1, 2, 3, 4, 5, 6, 7, 8
    ],

    "continent": [
        "Asia",
        "Europe",
        "Asia",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Africa"
    ],

    "education": [
        "Master",
        "Bachelor",
        "Master",
        "PhD",
        "Bachelor",
        "Master",
        "PhD",
        "Bachelor"
    ],

    "prevailing_wage": [
        50000,
        60000,
        45000,
        70000,
        55000,
        80000,
        90000,
        65000
    ],

    "case_status": [
        "Certified",
        "Denied",
        "Certified",
        "Certified",
        "Denied",
        "Certified",
        "Certified",
        "Denied"
    ]
})


# ============================================================
# 5. SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "data_source" not in st.session_state:
    st.session_state.data_source = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None

if "plot_generated" not in st.session_state:
    st.session_state.plot_generated = False

if "plot_type" not in st.session_state:
    st.session_state.plot_type = None

if "selected_column" not in st.session_state:
    st.session_state.selected_column = None

if "gemini_history" not in st.session_state:
    st.session_state.gemini_history = []

if "gemini_interaction_id" not in st.session_state:
    st.session_state.gemini_interaction_id = None


# ============================================================
# 6. MAIN TITLE
# ============================================================

st.markdown(
    '<div class="dashboard-header">'
    '<div class="main-title">'
    'Exploratory Data Analysis (EDA) Dashboard'
    '</div>'
    '<div class="main-subtitle">'
    'Upload a dataset or use the sample data, then explore '
    'statistics, visualizations, and AI-assisted insights.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 7. UPLOAD DATASET
# ============================================================

upload_col, upload_help_col = st.columns(
    [1.6, 1],
    gap="medium"
)

with upload_col:

    st.markdown(
        '<div class="upload-card">'
        '<p class="upload-card-title">'
        '📂 Upload Dataset'
        '</p>'
        '<p class="upload-card-help">'
        'Supported formats: CSV and Excel (.xlsx). '
        'Column names are read automatically from your file.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        key="dataset_uploader",
        help="Upload any tabular dataset to begin analysis."
    )

with upload_help_col:

    status_html = ""

    if st.session_state.df is not None:

        source_label = (
            "Uploaded file"
            if st.session_state.data_source == "uploaded"
            else "Default sample dataset"
        )

        status_html = (
            f'<span class="status-badge">'
            f'✓ Active: {source_label}'
            f'</span>'
        )

    st.markdown(
        '<div class="upload-card">'
        '<p class="upload-card-title">'
        '🚀 Quick Start'
        '</p>'
        '<p class="upload-card-help">'
        'No file ready? Click '
        '<strong>Use Default Dataset</strong> '
        'in the sidebar to load the built-in sample data.'
        '</p>'
        f'{status_html}'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 8. HANDLE UPLOADED DATASET
# ============================================================

if uploaded_file is not None:

    current_signature = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        st.session_state.uploaded_signature
        != current_signature
    ):

        try:

            if uploaded_file.name.lower().endswith(".csv"):

                new_df = pd.read_csv(
                    uploaded_file
                )

            else:

                new_df = pd.read_excel(
                    uploaded_file
                )

            if new_df.empty:

                st.error(
                    "The uploaded dataset is empty."
                )

            else:

                st.session_state.df = (
                    new_df.copy()
                )

                st.session_state.data_source = (
                    "uploaded"
                )

                st.session_state.uploaded_signature = (
                    current_signature
                )

                st.session_state.plot_generated = False

                st.session_state.plot_type = None

                st.session_state.selected_column = None

                st.session_state.gemini_history = []

                st.session_state.gemini_interaction_id = None

                st.success(
                    "✅ Dataset uploaded successfully!"
                )

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        'Navigation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Load data, choose a plot, and generate visualizations.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DEFAULT DATASET BUTTON
    # --------------------------------------------------------

    st.markdown(
        '<div class="default-dataset-card">'
        '<p><strong>Sample data</strong> — try the app instantly '
        'without uploading a file.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "📊 Use Default Dataset",
        type="primary",
        use_container_width=True,
        key="default_dataset_button"
    ):

        st.session_state.df = (
            default_df.copy()
        )

        st.session_state.data_source = (
            "default"
        )

        st.session_state.uploaded_signature = None

        st.session_state.plot_generated = False

        st.session_state.plot_type = None

        st.session_state.selected_column = None

        st.session_state.gemini_history = []

        st.session_state.gemini_interaction_id = None

        st.rerun()


    st.markdown("---")


    # --------------------------------------------------------
    # CURRENT DATASET
    # --------------------------------------------------------

    df = st.session_state.df


    # --------------------------------------------------------
    # PLOT SELECTION
    # --------------------------------------------------------

    if df is not None:

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            df.select_dtypes(
                exclude=np.number
            )
            .columns
            .tolist()
        )


        st.markdown(
            '<div class="sidebar-section-label">'
            'Visualization Controls'
            '</div>',
            unsafe_allow_html=True
        )


        plot_options = [
            "Bar",
            "Pie",
            "Histogram",
            "Box",
            "Scatter",
            "Line"
        ]


        plot_type = st.selectbox(
            "Plot type",
            plot_options,
            index=None,
            placeholder="Choose a plot type",
            key="sidebar_plot_type"
        )


        if plot_type in [
            "Bar",
            "Pie"
        ]:

            column_options = (
                categorical_columns
            )

            placeholder_text = (
                "Select Categorical Column"
            )

        elif plot_type in [
            "Histogram",
            "Box"
        ]:

            column_options = (
                numeric_columns
            )

            placeholder_text = (
                "Select Numerical Column"
            )

        elif plot_type in [
            "Scatter",
            "Line"
        ]:

            column_options = (
                numeric_columns
            )

            placeholder_text = (
                "Select Numerical Column"
            )

        else:

            column_options = []

            placeholder_text = (
                "Select Column"
            )


        if len(column_options) > 0:

            selected_column = st.selectbox(
                "Column",
                column_options,
                index=None,
                placeholder=placeholder_text,
                key="sidebar_column"
            )

        else:

            selected_column = None

            st.selectbox(
                "Column",
                ["Select a plot type first"],
                disabled=True,
                label_visibility="collapsed",
                key="disabled_column"
            )


        # ----------------------------------------------------
        # GENERATE PLOT BUTTON
        # ----------------------------------------------------

        if st.button(
            "Generate Plot",
            use_container_width=True,
            key="generate_plot_button"
        ):

            if plot_type is None:

                st.warning(
                    "Please select a plot type first."
                )

            elif selected_column is None:

                st.warning(
                    "Please select a column."
                )

            else:

                st.session_state.plot_type = (
                    plot_type
                )

                st.session_state.selected_column = (
                    selected_column
                )

                st.session_state.plot_generated = True

                st.rerun()


    else:

        st.info(
            "Upload a dataset or click "
            "'Use Default Dataset' to begin."
        )


# ============================================================
# 10. GET CURRENT DATASET
# ============================================================

df = st.session_state.df


# ============================================================
# 11. STOP IF NO DATASET
# ============================================================

if df is None:

    st.info(
        "👆 Upload a CSV/XLSX file above, or click "
        "**Use Default Dataset** in the sidebar to begin."
    )

    st.stop()


# ============================================================
# 12. DATASET PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Dataset Loaded'
    '</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df,
    use_container_width=True,
    height=400
)

st.write(
    f"Dataset Shape: "
    f"{df.shape[0]} rows × "
    f"{df.shape[1]} columns"
)


# ============================================================
# 13. DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Dataset Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        int(
            df.isnull()
            .sum()
            .sum()
        )
    )


# ============================================================
# 14. GENERATE VISUALIZATION
# ============================================================

if st.session_state.plot_generated:

    plot_type = (
        st.session_state.plot_type
    )

    selected_column = (
        st.session_state.selected_column
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    st.markdown(
        '<div class="plot-panel">'
        '<div class="section-title">'
        'Generated Visualization'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    try:

        # ----------------------------------------------------
        # BAR
        # ----------------------------------------------------

        if plot_type == "Bar":

            counts = (
                df[selected_column]
                .value_counts(
                    dropna=False
                )
                .reset_index()
            )

            counts.columns = [
                selected_column,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=selected_column,
                y="Count",
                title=(
                    f"Bar Chart - "
                    f"{selected_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PIE
        # ----------------------------------------------------

        elif plot_type == "Pie":

            counts = (
                df[selected_column]
                .value_counts(
                    dropna=False
                )
            )

            fig = px.pie(
                values=counts.values,
                names=counts.index.astype(str),
                title=(
                    f"Pie Chart - "
                    f"{selected_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------

        elif plot_type == "Histogram":

            fig = px.histogram(
                df,
                x=selected_column,
                title=(
                    f"Histogram - "
                    f"{selected_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # BOX
        # ----------------------------------------------------

        elif plot_type == "Box":

            fig = px.box(
                df,
                y=selected_column,
                title=(
                    f"Box Plot - "
                    f"{selected_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # SCATTER
        # ----------------------------------------------------

        elif plot_type == "Scatter":

            if len(numeric_columns) >= 2:

                x_column = st.selectbox(
                    "Select X-axis",
                    numeric_columns,
                    index=None,
                    placeholder="Select X-axis",
                    key="scatter_x"
                )

                y_column = st.selectbox(
                    "Select Y-axis",
                    numeric_columns,
                    index=None,
                    placeholder="Select Y-axis",
                    key="scatter_y"
                )


                if (
                    x_column is not None
                    and y_column is not None
                ):

                    if x_column == y_column:

                        st.warning(
                            "Please select two "
                            "different columns."
                        )

                    else:

                        fig = px.scatter(
                            df,
                            x=x_column,
                            y=y_column,
                            title=(
                                f"{x_column} "
                                f"vs {y_column}"
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

            else:

                st.warning(
                    "Scatter plot requires at least "
                    "two numerical columns."
                )


        # ----------------------------------------------------
        # LINE
        # ----------------------------------------------------

        elif plot_type == "Line":

            if len(numeric_columns) >= 2:

                x_column = st.selectbox(
                    "Select X-axis",
                    numeric_columns,
                    index=None,
                    placeholder="Select X-axis",
                    key="line_x"
                )

                y_column = st.selectbox(
                    "Select Y-axis",
                    numeric_columns,
                    index=None,
                    placeholder="Select Y-axis",
                    key="line_y"
                )


                if (
                    x_column is not None
                    and y_column is not None
                ):

                    if x_column == y_column:

                        st.warning(
                            "Please select two "
                            "different columns."
                        )

                    else:

                        fig = px.line(
                            df,
                            x=x_column,
                            y=y_column,
                            title=(
                                f"{x_column} "
                                f"vs {y_column}"
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

            else:

                st.warning(
                    "Line chart requires at least "
                    "two numerical columns."
                )


    except Exception as error:

        st.error(
            f"Unable to generate the plot: {error}"
        )


# ============================================================
# 15. EDA ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "🔍 Exploratory Data Analysis"
)


# ============================================================
# 15.1 DATASET OVERVIEW
# ============================================================

with st.expander(
    "📋 Dataset Overview",
    expanded=True
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Number of Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Number of Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Numerical Columns",
            len(
                df.select_dtypes(
                    include=np.number
                ).columns
            )
        )

    with col4:

        st.metric(
            "Categorical Columns",
            len(
                df.select_dtypes(
                    exclude=np.number
                ).columns
            )
        )


# ============================================================
# 15.2 DATA TYPES
# ============================================================

with st.expander(
    "🔎 Data Types"
):

    dtype_df = pd.DataFrame({

        "Column":
            df.columns,

        "Data Type":
            [
                str(dtype)
                for dtype in df.dtypes
            ]

    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )


# ============================================================
# 15.3 STATISTICAL SUMMARY
# ============================================================

with st.expander(
    "📊 Statistical Summary"
):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
    )

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe(),
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns available."
        )


# ============================================================
# 15.4 MISSING VALUE ANALYSIS
# ============================================================

with st.expander(
    "❌ Missing Value Analysis"
):

    missing_df = pd.DataFrame({

        "Column":
            df.columns,

        "Missing Values":
            [
                df[col].isnull().sum()
                for col in df.columns
            ],

        "Missing Percentage":
            [
                round(
                    df[col]
                    .isnull()
                    .mean() * 100,
                    2
                )
                for col in df.columns
            ]

    })


    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]


    if missing_df.empty:

        st.success(
            "✅ No missing values found."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True
        )


# ============================================================
# 16. UNIVARIATE ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "📈 Univariate Analysis"
)

st.write(
    "Univariate analysis studies one variable at a time."
)


univariate_column = st.selectbox(
    "Select Column",
    df.columns.tolist(),
    index=None,
    placeholder="Select Column",
    key="univariate_column"
)


if univariate_column is not None:

    if pd.api.types.is_numeric_dtype(
        df[univariate_column]
    ):

        fig = px.histogram(
            df,
            x=univariate_column,
            title=(
                f"Distribution of "
                f"{univariate_column}"
            ),
            marginal="box"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Mean",
                round(
                    df[univariate_column]
                    .mean(),
                    2
                )
            )

        with col2:

            st.metric(
                "Median",
                round(
                    df[univariate_column]
                    .median(),
                    2
                )
            )

        with col3:

            st.metric(
                "Minimum",
                round(
                    df[univariate_column]
                    .min(),
                    2
                )
            )

        with col4:

            st.metric(
                "Maximum",
                round(
                    df[univariate_column]
                    .max(),
                    2
                )
            )


        fig_box = px.box(
            df,
            y=univariate_column,
            title=(
                f"Box Plot - "
                f"{univariate_column}"
            )
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )


    else:

        counts = (
            df[univariate_column]
            .value_counts(
                dropna=False
            )
            .reset_index()
        )

        counts.columns = [
            univariate_column,
            "Count"
        ]


        fig_bar = px.bar(
            counts,
            x=univariate_column,
            y="Count",
            title=(
                f"Count - "
                f"{univariate_column}"
            )
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


        fig_pie = px.pie(
            counts,
            names=univariate_column,
            values="Count",
            title=(
                f"Distribution - "
                f"{univariate_column}"
            )
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )


        st.dataframe(
            counts,
            use_container_width=True
        )


# ============================================================
# 17. BIVARIATE ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "📊 Bivariate Analysis"
)

st.write(
    "Bivariate analysis studies the relationship "
    "between two variables."
)


x_column = st.selectbox(
    "Select X-axis Column",
    df.columns.tolist(),
    index=None,
    placeholder="Select X-axis Column",
    key="bivariate_x"
)


y_column = st.selectbox(
    "Select Y-axis Column",
    df.columns.tolist(),
    index=None,
    placeholder="Select Y-axis Column",
    key="bivariate_y"
)


if (
    x_column is not None
    and y_column is not None
):

    if x_column == y_column:

        st.warning(
            "Please select two different columns."
        )

    else:

        x_is_numeric = (
            pd.api.types.is_numeric_dtype(
                df[x_column]
            )
        )

        y_is_numeric = (
            pd.api.types.is_numeric_dtype(
                df[y_column]
            )
        )


        # ----------------------------------------------------
        # NUMERICAL VS NUMERICAL
        # ----------------------------------------------------

        if (
            x_is_numeric
            and y_is_numeric
        ):

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=(
                    f"{x_column} "
                    f"vs {y_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


            correlation = (
                df[
                    [x_column, y_column]
                ]
                .corr()
                .iloc[0, 1]
            )


            st.metric(
                "Correlation",
                round(
                    correlation,
                    3
                )
            )


            if correlation > 0:

                st.info(
                    "Positive relationship."
                )

            elif correlation < 0:

                st.info(
                    "Negative relationship."
                )

            else:

                st.info(
                    "Approximately no linear relationship."
                )


        # ----------------------------------------------------
        # CATEGORICAL VS NUMERICAL
        # ----------------------------------------------------

        elif (
            not x_is_numeric
            and y_is_numeric
        ):

            fig = px.box(
                df,
                x=x_column,
                y=y_column,
                title=(
                    f"{y_column} "
                    f"by {x_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # NUMERICAL VS CATEGORICAL
        # ----------------------------------------------------

        elif (
            x_is_numeric
            and not y_is_numeric
        ):

            fig = px.box(
                df,
                x=y_column,
                y=x_column,
                title=(
                    f"{x_column} "
                    f"by {y_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # CATEGORICAL VS CATEGORICAL
        # ----------------------------------------------------

        else:

            cross_table = pd.crosstab(
                df[x_column],
                df[y_column]
            )

            st.subheader(
                "Cross Tabulation"
            )

            st.dataframe(
                cross_table,
                use_container_width=True
            )


            fig = px.bar(
                cross_table,
                barmode="group",
                title=(
                    f"{x_column} "
                    f"vs {y_column}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 18. CORRELATION ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "🔥 Correlation Analysis"
)

numeric_columns = (
    df.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


if len(numeric_columns) >= 2:

    correlation_df = (
        df[numeric_columns]
        .corr()
    )


    st.subheader(
        "Correlation Matrix"
    )

    st.dataframe(
        correlation_df.round(2),
        use_container_width=True
    )


    st.subheader(
        "🔥 Correlation Heatmap"
    )


    heatmap_fig = px.imshow(
        correlation_df,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap"
    )


    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )


else:

    st.info(
        "At least two numerical columns are "
        "required for correlation analysis."
    )


# ============================================================
# 19. OUTLIER DETECTION
# ============================================================

st.markdown("---")

st.header(
    "🚨 Outlier Detection"
)

st.write(
    "Outliers are detected using the "
    "IQR method."
)


if len(numeric_columns) > 0:

    outlier_results = []


    for column in numeric_columns:

        data = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
        )


        if len(data) == 0:

            continue


        Q1 = data.quantile(
            0.25
        )

        Q3 = data.quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower_bound = (
            Q1 - 1.5 * IQR
        )

        upper_bound = (
            Q3 + 1.5 * IQR
        )


        outliers = data[
            (data < lower_bound)
            |
            (data > upper_bound)
        ]


        outlier_count = len(
            outliers
        )


        percentage = (
            outlier_count
            / len(data)
        ) * 100


        outlier_results.append({

            "Column":
                column,

            "Q1":
                round(Q1, 2),

            "Q3":
                round(Q3, 2),

            "IQR":
                round(IQR, 2),

            "Lower Bound":
                round(
                    lower_bound,
                    2
                ),

            "Upper Bound":
                round(
                    upper_bound,
                    2
                ),

            "Outlier Count":
                outlier_count,

            "Outlier Percentage":
                round(
                    percentage,
                    2
                )

        })


    if outlier_results:

        outlier_df = pd.DataFrame(
            outlier_results
        )


        st.dataframe(
            outlier_df,
            use_container_width=True
        )


        fig = px.bar(
            outlier_df,
            x="Column",
            y="Outlier Count",
            title="Outlier Count by Column"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.info(
            "No numerical data available."
        )

else:

    st.info(
        "No numerical columns found."
    )


# ============================================================
# 20. DATA CLEANING
# ============================================================

st.markdown("---")

st.header(
    "🧹 Data Cleaning"
)

st.write(
    "Select a column containing missing values "
    "and choose a method to handle them."
)


missing_columns = (
    df.columns[
        df.isnull().any()
    ]
    .tolist()
)


if len(missing_columns) == 0:

    st.success(
        "✅ No missing values found."
    )

else:

    missing_summary = pd.DataFrame({

        "Column":
            missing_columns,

        "Missing Values":
            [
                df[col]
                .isnull()
                .sum()
                for col in missing_columns
            ],

        "Missing Percentage":
            [
                round(
                    df[col]
                    .isnull()
                    .mean() * 100,
                    2
                )
                for col in missing_columns
            ]

    })


    st.subheader(
        "Missing Value Summary"
    )


    st.dataframe(
        missing_summary,
        use_container_width=True
    )


    selected_cleaning_column = st.selectbox(
        "Select Column",
        missing_columns,
        index=None,
        placeholder="Select Column",
        key="cleaning_column"
    )


    if selected_cleaning_column is not None:

        cleaning_method = st.selectbox(
            "Select Cleaning Method",
            [
                "Mean",
                "Median",
                "Mode",
                "Drop Rows"
            ],
            index=None,
            placeholder="Select Cleaning Method",
            key="cleaning_method"
        )


        if cleaning_method is not None:

            missing_count = (
                df[
                    selected_cleaning_column
                ]
                .isnull()
                .sum()
            )


            st.info(
                f"Column: "
                f"{selected_cleaning_column} | "
                f"Missing Values: "
                f"{missing_count}"
            )


            if st.button(
                "Apply Cleaning",
                type="primary",
                key="apply_cleaning"
            ):

                try:

                    if cleaning_method == "Mean":

                        if pd.api.types.is_numeric_dtype(
                            df[
                                selected_cleaning_column
                            ]
                        ):

                            value = (
                                df[
                                    selected_cleaning_column
                                ]
                                .mean()
                            )

                            df[
                                selected_cleaning_column
                            ] = (
                                df[
                                    selected_cleaning_column
                                ]
                                .fillna(value)
                            )

                        else:

                            st.error(
                                "Mean can only be used "
                                "with numerical columns."
                            )

                            st.stop()


                    elif cleaning_method == "Median":

                        if pd.api.types.is_numeric_dtype(
                            df[
                                selected_cleaning_column
                            ]
                        ):

                            value = (
                                df[
                                    selected_cleaning_column
                                ]
                                .median()
                            )

                            df[
                                selected_cleaning_column
                            ] = (
                                df[
                                    selected_cleaning_column
                                ]
                                .fillna(value)
                            )

                        else:

                            st.error(
                                "Median can only be used "
                                "with numerical columns."
                            )

                            st.stop()


                    elif cleaning_method == "Mode":

                        mode_value = (
                            df[
                                selected_cleaning_column
                            ]
                            .mode()
                        )


                        if not mode_value.empty:

                            df[
                                selected_cleaning_column
                            ] = (
                                df[
                                    selected_cleaning_column
                                ]
                                .fillna(
                                    mode_value.iloc[0]
                                )
                            )


                    elif cleaning_method == "Drop Rows":

                        df = (
                            df
                            .dropna(
                                subset=[
                                    selected_cleaning_column
                                ]
                            )
                            .reset_index(
                                drop=True
                            )
                        )


                    st.session_state.df = (
                        df.copy()
                    )


                    st.session_state.plot_generated = False

                    st.session_state.gemini_history = []

                    st.session_state.gemini_interaction_id = None

                    st.success(
                        "✅ Data cleaning completed!"
                    )

                    st.rerun()


                except Exception as error:

                    st.error(
                        f"Cleaning error: {error}"
                    )


# ============================================================
# 21. CURRENT DATASET
# ============================================================

st.subheader(
    "👀 Current Dataset"
)

st.dataframe(
    st.session_state.df,
    use_container_width=True,
    height=320
)


# ============================================================
# 22. DOWNLOAD CLEANED DATASET
# ============================================================

st.markdown("---")

st.header(
    "⬇️ Download Cleaned Dataset"
)

st.write(
    "Download the current dataset as a CSV file."
)


csv_data = (
    st.session_state.df
    .to_csv(
        index=False
    )
    .encode("utf-8")
)


st.download_button(
    label="⬇️ Download Cleaned Dataset",
    data=csv_data,
    file_name="cleaned_dataset.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_cleaned_dataset"
)


# ============================================================
# 23. INTERACTIVE DATA FILTERING
# ============================================================

st.markdown("---")

st.header(
    "🔎 Data Filtering"
)

st.write(
    "Select a column and filter the dataset "
    "based on its values."
)


filter_columns = (
    df.columns.tolist()
)


selected_filter_column = st.selectbox(
    "Select column to filter",
    filter_columns,
    index=None,
    placeholder="Choose a column",
    key="filter_column"
)


if selected_filter_column is not None:

    unique_values = (
        df[
            selected_filter_column
        ]
        .dropna()
        .unique()
        .tolist()
    )


    unique_values = sorted(
        [str(value) for value in unique_values]
    )


    selected_filter_value = st.selectbox(
        "Select value",
        unique_values,
        index=None,
        placeholder="Choose a value",
        key="filter_value"
    )


    if selected_filter_value is not None:

        filtered_data = df[
            df[
                selected_filter_column
            ]
            .astype(str)
            == selected_filter_value
        ]


        st.subheader(
            "📊 Filtered Dataset"
        )


        st.write(
            f"Showing {len(filtered_data)} rows "
            f"where **{selected_filter_column} = "
            f"{selected_filter_value}**"
        )


        st.dataframe(
            filtered_data,
            use_container_width=True,
            height=320
        )


        filtered_csv = (
            filtered_data
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )


        st.download_button(
            label="⬇️ Download Filtered Dataset",
            data=filtered_csv,
            file_name="filtered_dataset.csv",
            mime="text/csv",
            key="download_filtered"
        )


# ============================================================
# 24. DUPLICATE VALUE ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "♻️ Duplicate Value Analysis"
)

st.write(
    "Identify and remove duplicate rows."
)


duplicate_count = (
    df.duplicated().sum()
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Total Rows",
        len(df)
    )


with col2:

    st.metric(
        "Duplicate Rows",
        int(duplicate_count)
    )


if duplicate_count > 0:

    duplicate_data = df[
        df.duplicated(
            keep=False
        )
    ]


    st.subheader(
        "🔍 Duplicate Rows"
    )


    st.dataframe(
        duplicate_data,
        use_container_width=True,
        height=320
    )


    if st.button(
        "🗑️ Remove Duplicate Rows",
        key="remove_duplicates"
    ):

        df = (
            df
            .drop_duplicates()
            .reset_index(
                drop=True
            )
        )


        st.session_state.df = (
            df.copy()
        )


        st.session_state.gemini_history = []

        st.session_state.gemini_interaction_id = None


        st.success(
            "✅ Duplicate rows removed!"
        )


        st.rerun()


else:

    st.success(
        "✅ No duplicate rows found."
    )


# ============================================================
# 25. AUTOMATIC OUTLIER DETECTION
# ============================================================

st.markdown("---")

st.header(
    "📌 Detailed Outlier Detection"
)

st.write(
    "Select a numerical column to inspect "
    "its outliers using the IQR method."
)


numerical_columns = (
    df.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


if len(numerical_columns) > 0:

    selected_outlier_column = st.selectbox(
        "Select numerical column",
        numerical_columns,
        index=None,
        placeholder="Choose a numerical column",
        key="detailed_outlier_column"
    )


    if selected_outlier_column is not None:

        data = (
            df[
                selected_outlier_column
            ]
            .dropna()
        )


        Q1 = data.quantile(
            0.25
        )

        Q3 = data.quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower_bound = (
            Q1 - 1.5 * IQR
        )

        upper_bound = (
            Q3 + 1.5 * IQR
        )


        outlier_data = df[
            (
                df[
                    selected_outlier_column
                ] < lower_bound
            )
            |
            (
                df[
                    selected_outlier_column
                ] > upper_bound
            )
        ]


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Q1",
                round(Q1, 2)
            )


        with col2:

            st.metric(
                "Q3",
                round(Q3, 2)
            )


        with col3:

            st.metric(
                "IQR",
                round(IQR, 2)
            )


        with col4:

            st.metric(
                "Outliers",
                len(outlier_data)
            )


        st.write(
            f"**Lower Bound:** "
            f"{lower_bound:.2f}"
        )

        st.write(
            f"**Upper Bound:** "
            f"{upper_bound:.2f}"
        )


        if len(outlier_data) > 0:

            st.warning(
                f"⚠️ {len(outlier_data)} "
                f"outlier row(s) found."
            )


            st.dataframe(
                outlier_data,
                use_container_width=True,
                height=320
            )


        else:

            st.success(
                "✅ No outliers found."
            )


        # Box plot

        fig = px.box(
            df,
            y=selected_outlier_column,
            title=(
                f"Box Plot - "
                f"{selected_outlier_column}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


else:

    st.info(
        "No numerical columns found."
    )
# ============================================================
# 26. GEMINI AI DATA ASSISTANT
# ============================================================

from streamlit_mic_recorder import speech_to_text


st.markdown("---")

st.header("🤖 Gemini AI EDA Assistant")

st.write(
    "Ask questions about your dataset using text or voice."
)


# ============================================================
# LANGUAGE SELECTION
# ============================================================

language_options = {
    "English": "en-US",
    "Telugu": "te-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "Urdu": "ur-IN"
}


selected_language = st.selectbox(
    "🌐 Select your language",
    list(language_options.keys()),
    index=0,
    key="gemini_language"
)


selected_language_code = language_options[
    selected_language
]


st.caption(
    f"🎤 Voice and Gemini responses will use: **{selected_language}**"
)


# ============================================================
# CHECK GEMINI API
# ============================================================

if client is None:

    st.error(
        "Gemini API key is not configured. "
        "Please check your .env file or Render Environment Variables."
    )

else:

    try:

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        rows = df.shape[0]

        columns = df.shape[1]

        column_names = df.columns.tolist()

        data_types = (
            df.dtypes
            .astype(str)
            .to_dict()
        )


        # ====================================================
        # MISSING VALUES
        # ====================================================

        missing_values = df.isnull().sum()

        missing_info = {
            column: int(value)
            for column, value in missing_values.items()
            if value > 0
        }


        if not missing_info:

            missing_info = "No missing values found."


        # ====================================================
        # NUMERICAL COLUMNS
        # ====================================================

        numerical_columns = (
            df.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )


        # ====================================================
        # CATEGORICAL COLUMNS
        # ====================================================

        categorical_columns = (
            df.select_dtypes(
                exclude=np.number
            )
            .columns
            .tolist()
        )


        # ====================================================
        # STATISTICAL SUMMARY
        # ====================================================

        if numerical_columns:

            statistical_summary = (
                df[numerical_columns]
                .describe()
                .round(2)
                .to_string()
            )

        else:

            statistical_summary = (
                "No numerical columns found."
            )


        # ====================================================
        # CORRELATION
        # ====================================================

        if len(numerical_columns) >= 2:

            correlation_matrix = (
                df[numerical_columns]
                .corr()
                .round(2)
            )

            correlation_info = (
                correlation_matrix.to_string()
            )

        else:

            correlation_info = (
                "Correlation cannot be calculated "
                "because fewer than two numerical "
                "columns are available."
            )


        # ====================================================
        # OUTLIER INFORMATION
        # ====================================================

        outlier_information = []


        for column in numerical_columns:

            data = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
                .dropna()
            )


            if len(data) == 0:
                continue


            Q1 = data.quantile(0.25)

            Q3 = data.quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR

            upper_bound = Q3 + 1.5 * IQR


            outlier_count = int(
                (
                    (data < lower_bound)
                    |
                    (data > upper_bound)
                ).sum()
            )


            outlier_information.append({

                "column": column,

                "outlier_count":
                    outlier_count,

                "lower_bound":
                    round(
                        lower_bound,
                        2
                    ),

                "upper_bound":
                    round(
                        upper_bound,
                        2
                    )

            })


        if outlier_information:

            outlier_info = str(
                outlier_information
            )

        else:

            outlier_info = (
                "No numerical columns available "
                "for outlier analysis."
            )


        # ====================================================
        # EDA CONTEXT
        # ====================================================

        eda_context = f"""

You are an AI Data Analysis Assistant
inside an Exploratory Data Analysis (EDA)
application.

Analyze the user's dataset using ONLY
the information provided below.

==================================================
DATASET INFORMATION
==================================================

Number of rows:
{rows}

Number of columns:
{columns}

Column names:
{column_names}

Data types:
{data_types}

==================================================
NUMERICAL COLUMNS
==================================================

{numerical_columns}

==================================================
CATEGORICAL COLUMNS
==================================================

{categorical_columns}

==================================================
MISSING VALUES
==================================================

{missing_info}

==================================================
STATISTICAL SUMMARY
==================================================

{statistical_summary}

==================================================
CORRELATION
==================================================

{correlation_info}

==================================================
OUTLIER INFORMATION
==================================================

{outlier_info}

==================================================

IMPORTANT RULES:

1. Answer using the dataset information above.

2. Do not invent dataset values.

3. If the required information is not available,
   clearly tell the user.

4. Explain the answer in simple language.

5. The user is a student learning EDA.

6. Explain concepts when appropriate.

7. Give practical EDA suggestions when useful.

"""


        # ====================================================
        # CHAT HISTORY
        # ====================================================

        if "chat_history" not in st.session_state:

            st.session_state.chat_history = []


        # ====================================================
        # DISPLAY PREVIOUS CHAT
        # ====================================================

        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )


        # ====================================================
        # VOICE INPUT
        # ====================================================

        st.subheader("🎤 Voice Question")


        voice_text = speech_to_text(

            language=selected_language_code,

            start_prompt="🎤 Start Speaking",

            stop_prompt="⏹️ Stop Speaking",

            just_once=True,

            use_container_width=True,

            key="gemini_voice_input"

        )


        # ====================================================
        # SHOW VOICE INPUT
        # ====================================================

        if voice_text:

            st.success(
                f"🎤 You said: {voice_text}"
            )


        # ====================================================
        # TEXT CHAT INPUT
        # ====================================================

        st.subheader("💬 Text Question")


        user_question = st.chat_input(

            "Ask Gemini about your dataset...",

            key="gemini_chat_input"

        )


        # ====================================================
        # SELECT QUESTION
        # ====================================================

        question = None


        if voice_text:

            question = voice_text

        elif user_question:

            question = user_question


        # ====================================================
        # PROCESS QUESTION
        # ====================================================

        if question:

            # ------------------------------------------------
            # DISPLAY USER QUESTION
            # ------------------------------------------------

            with st.chat_message(
                "user"
            ):

                st.write(
                    question
                )


            # ------------------------------------------------
            # SAVE USER QUESTION
            # ------------------------------------------------

            st.session_state.chat_history.append({

                "role": "user",

                "content": question

            })


            # =================================================
            # GEMINI PROMPT
            # =================================================

            final_prompt = f"""

{eda_context}

==================================================
USER QUESTION
==================================================

{question}

==================================================

LANGUAGE REQUIREMENT
==================================================

The user selected:

{selected_language}

Answer the user ONLY in {selected_language}.

If the question was spoken in another language
but the selected language is {selected_language},
still answer in {selected_language}.

Do not answer in English unless English
was selected.

==================================================

ANSWER STYLE
==================================================

Give a clear and useful answer.

If the question asks about the dataset,
use the actual dataset information.

If the question asks for a calculation,
calculate it from the available dataset information.

If the question asks about an EDA concept,
explain the concept simply and then relate it
to the dataset when possible.

The user is learning data analysis, so use
step-by-step explanations when appropriate.

"""


            # =================================================
            # CALL GEMINI
            # =================================================

            try:

                with st.spinner(
                    "🤖 Gemini is thinking..."
                ):

                    interaction = (
                        client.interactions.create(

                            model="gemini-3.6-flash",

                            input=final_prompt

                        )
                    )


                    answer = (
                        interaction.output_text
                    )


                # =================================================
                # DISPLAY GEMINI ANSWER
                # =================================================

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        answer
                    )


                # =================================================
                # SAVE GEMINI ANSWER
                # =================================================

                st.session_state.chat_history.append({

                    "role":
                        "assistant",

                    "content":
                        answer

                })


            except Exception as error:

                with st.chat_message(
                    "assistant"
                ):

                    st.error(
                        f"Gemini error: {error}"
                    )


    except Exception as error:

        st.error(
            f"Unable to prepare EDA information: {error}"
        )