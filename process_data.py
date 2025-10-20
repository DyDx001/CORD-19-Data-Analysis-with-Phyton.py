# -----------------------------------------------
# Cell 1: Imports
# -----------------------------------------------
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re

# Set plot style
sns.set_theme(style="whitegrid")
print("Libraries imported successfully.")

#
# -----------------------------------------------
# Cell 2: Part 1 - Data Loading and Exploration
# -----------------------------------------------
#
print("--- Part 1: Data Loading & Exploration ---")

# Define the file and columns to use.
# We only load relevant columns to save memory.
FILE_PATH = 'metadata.csv'
COLS_TO_USE = ['title', 'abstract', 'publish_time', 'journal', 'source_x']

# Load the dataset (use low_memory=False for mixed types)
try:
    df = pd.read_csv(FILE_PATH, usecols=COLS_TO_USE, low_memory=False)
    print(f"Successfully loaded {FILE_PATH}")
except FileNotFoundError:
    print(f"ERROR: {FILE_PATH} not found.")
    print("Please download it from Kaggle and place it in the same directory.")
    # Stop execution if file isn't found
    raise

# Examine the first few rows
print("\n[1.1] Data Head:")
print(df.head())

# Check DataFrame dimensions
print(f"\n[1.2] Data Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

# Check data types and missing values
print("\n[1.3] Data Info (Types & Missing Values):")
df.info()

#
# -----------------------------------------------
# Cell 3: Part 2 - Data Cleaning and Preparation
# -----------------------------------------------
#
print("\n--- Part 2: Data Cleaning & Preparation ---")

# Handle missing data
# We need title, abstract, and publish_time to be useful.
df_clean = df.dropna(subset=['title', 'abstract', 'publish_time']).copy()
print(f"Dropped rows with missing title, abstract, or publish_time. New shape: {df_clean.shape}")

# Fill missing journal names
df_clean['journal'] = df_clean['journal'].fillna('Unknown')

# Convert date column to datetime format
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
# Drop any rows where date conversion failed
df_clean = df_clean.dropna(subset=['publish_time'])

# Extract year
df_clean['year'] = df_clean['publish_time'].dt.year

# Create new column: abstract word count
df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))

# Filter for relevant years (e.g., 2019-2022) to remove older, irrelevant papers
df_clean = df_clean[df_clean['year'].between(2019, 2022)]

print(f"Cleaned and filtered data. Shape: {df_clean.shape}")
print("\n[2.1] Cleaned Data Head:")
print(df_clean.head())

#
# -----------------------------------------------
# Cell 4: Create a Sample for the Streamlit App
# -----------------------------------------------
#
# The dataset is still too large. Let's create a 50,000-row sample
# for the Streamlit app to use.
SAMPLE_SIZE = 50000
OUTPUT_FILE = 'cleaned_metadata_sample.csv'

if len(df_clean) > SAMPLE_SIZE:
    df_sample = df_clean.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df_clean.copy()

# Save the sample to a new CSV
df_sample.to_csv(OUTPUT_FILE, index=False)

print(f"\n--- Data Sampled ---")
print(f"Saved {len(df_sample)} rows to {OUTPUT_FILE} for the Streamlit app.")

#
# -----------------------------------------------
# Cell 5: Part 3 - Data Analysis (on the Sample)
# -----------------------------------------------
#
print(f"\n--- Part 3: Data Analysis (on {SAMPLE_SIZE} sample) ---")

# 1. Count papers by publication year
papers_by_year = df_sample['year'].value_counts().sort_index()
print("\n[3.1] Papers by Year:")
print(papers_by_year)

# 2. Identify top journals
top_journals = df_sample['journal'].value_counts().nlargest(15)
print("\n[3.2] Top 15 Journals:")
print(top_journals)

# 3. Find most frequent words in titles (using scikit-learn)
# We remove common 'covid' words to find more interesting topics.
stop_words = list(CountVectorizer(stop_words='english').get_stop_words())
stop_words.extend(['covid', '19', 'coronavirus', 'sars', 'cov', '2'])

vec = CountVectorizer(stop_words=stop_words, max_features=25)
word_matrix = vec.fit_transform(df_sample['title'])
word_freq = pd.DataFrame(
    word_matrix.toarray(),
    columns=vec.get_feature_names_out()
).sum().sort_values(ascending=False)

print("\n[3.3] Top 25 Words in Titles (excluding 'covid'):")
print(word_freq)

#
# -----------------------------------------------
# Cell 6: Part 3 - Data Visualization (on the Sample)
# -----------------------------------------------
#
print("\n--- Part 3: Visualizations (on Sample) ---")
print("Generating and displaying plots...")

# 1. Plot: Number of publications over time
plt.figure(figsize=(10, 6))
sns.barplot(x=papers_by_year.index, y=papers_by_year.values)
plt.title('Publications by Year (from Sample)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.show()

# 2. Plot: Bar chart of top publishing journals
plt.figure(figsize=(10, 8))
sns.barplot(y=top_journals.index, x=top_journals.values, orient='h')
plt.title('Top 15 Publishing Journals (from Sample)', fontsize=16)
plt.xlabel('Number of Papers')
plt.ylabel('Journal')
plt.show()

# 3. Plot: Word cloud of paper titles
text = " ".join(title for title in df_sample.title.dropna())
wordcloud = WordCloud(
    background_color='white',
    width=800,
    height=400,
    stopwords=stop_words
).generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Paper Titles (from Sample)', fontsize=16)
plt.show()

# 4. Plot: Distribution of paper counts by source
top_sources = df_sample['source_x'].value_counts().nlargest(10)
plt.figure(figsize=(10, 8))
plt.pie(top_sources, labels=top_sources.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Top 10 Sources (from Sample)', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

print("\n--- Analysis Notebook Complete! ---")
