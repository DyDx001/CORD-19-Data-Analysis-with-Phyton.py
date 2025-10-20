import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Page Configuration ---
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide"
)

# --- Title and Description ---
st.title("🔬 CORD-19 Research Paper Explorer")
st.markdown("""
This application provides a simple, interactive exploration of the CORD-19 research dataset.
The data shown here is a **50,000-paper sample** from 2019-2022.
""")

# --- Data Loading ---
# Use st.cache_data to load data only once
@st.cache_data
def load_data():
    """
    Loads the cleaned, sampled data.
    """
    try:
        df = pd.read_csv('cleaned_metadata_sample.csv')
        # Ensure correct dtypes after loading from CSV
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        df['year'] = pd.to_numeric(df['year']).astype(int)
        return df
    except FileNotFoundError:
        st.error(
            "**ERROR: `cleaned_metadata_sample.csv` not found.** "
            "Please run the `process_data.py` script "
            "first to generate the data file."
        )
        return None

df = load_data()

# Stop execution if data failed to load
if df is None:
    st.stop()


# --- Helper Functions for Plotting ---
# We use helper functions to keep the main app logic clean
# We also use st.cache_data on plots for performance
@st.cache_data
def plot_publications_over_time(data):
    """
    Generates a bar chart of publications by year.
    """
    year_counts = data['year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax)
    ax.set_title('Publications Over Time', fontsize=16)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Papers')
    return fig

@st.cache_data
def plot_top_journals(data):
    """
    Generates a horizontal bar chart of top 15 journals.
    """
    top_j = data['journal'].value_counts().nlargest(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(y=top_j.index, x=top_j.values, orient='h', ax=ax)
    ax.set_title('Top 15 Publishing Journals', fontsize=16)
    ax.set_xlabel('Number of Papers')
    ax.set_ylabel('Journal')
    return fig

@st.cache_data
def plot_source_distribution(data):
    """
    Generates a pie chart of the top 10 paper sources.
    """
    source_counts = data['source_x'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title('Distribution of Top 10 Sources', fontsize=16)
    ax.axis('equal')
    return fig

@st.cache_data
def plot_title_word_cloud(data):
    """
    Generates a word cloud from paper titles.
    """
    # Define stop words
    stop_words = ['covid', '19', 'coronavirus', 'sars', 'cov', '2']
    
    text = " ".join(title for title in data.title.dropna())
    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color='white',
        width=800,
        height=400,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Words in Paper Titles', fontsize=16)
    return fig

# --- Interactive Sidebar ---
st.sidebar.header("Filter Data")

min_year = int(df['year'].min())
max_year = int(df['year'].max())

# Interactive Slider
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Interactive Multiselect for Source
all_sources = df['source_x'].unique()
selected_sources = st.sidebar.multiselect(
    "Select Source(s):",
    options=all_sources,
    default=all_sources
)

# --- Filter Data Based on Widgets ---
filtered_df = df[
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1]) &
    (df['source_x'].isin(selected_sources))
]

# --- Main Page Layout ---

# 1. Show a sample of the filtered data
st.header(f"Exploring {len(filtered_df)} Papers")
st.markdown(f"Displaying data for years **{year_range[0]}** to **{year_range[1]}**.")
st.dataframe(filtered_df[['title', 'journal', 'publish_time', 'source_x', 'year']].head(20))

st.divider()

# 2. Display Visualizations in columns
st.header("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Publications Over Time")
    if not filtered_df.empty:
        st.pyplot(plot_publications_over_time(filtered_df))
    else:
        st.warning("No data for this selection.")

with col2:
    st.subheader("Top 15 Publishing Journals")
    if not filtered_df.empty:
        st.pyplot(plot_top_journals(filtered_df))
    else:
        st.warning("No data for this selection.")

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribution of Top 10 Sources")
    if not filtered_df.empty:
        st.pyplot(plot_source_distribution(filtered_df))
    else:
        st.warning("No data for this selection.")

with col4:
    st.subheader("Title Word Cloud")
    if not filtered_df.title.dropna().empty:
        st.pyplot(plot_title_word_cloud(filtered_df))
    else:
        st.warning("No titles to generate word cloud.")
