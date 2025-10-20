# Frameworks_Assignment: CORD-19 Data Explorer

This project analyzes the CORD-19 research dataset (specifically `metadata.csv`) and presents the findings in an interactive Streamlit web application.

## Project Structure

* `analysis_and_preprocessing.ipynb`: (Parts 1-3) A Jupyter Notebook used for initial data loading, exploration, cleaning, and analysis of the full `metadata.csv` file. **Its most important output is `cleaned_metadata_sample.csv`**, which is a much smaller, cleaned file for the app.
* `app.py`: (Part 4) The Streamlit web application. This script loads the `cleaned_metadata_sample.csv` file to generate interactive visualizations.
* `requirements.txt`: A list of all required Python packages.
* `README.md`: This file (Part 5).

---

## 🚀 How to Run This Project

### Prerequisites

* Python 3.7+
* The `metadata.csv` file from the [CORD-19 Kaggle dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).

### Step-by-Step Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd Frameworks_Assignment
    ```

2.  **Download the Data:**
    * Download `metadata.csv` from the Kaggle link above.
    * Place the file in the root of your `Frameworks_Assignment` directory.

3.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Preprocessing Notebook (CRITICAL STEP):**
    * Start Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
    * Open `analysis_and_preprocessing.ipynb`.
    * Run all cells in the notebook. This will read `metadata.csv`, perform the analysis, and save the `cleaned_metadata_sample.csv` file.

6.  **Run the Streamlit Application:**
    * In your terminal, run:
        ```bash
        streamlit run app.py
        ```
    * Your web browser will automatically open to the application.

---

## Part 5: Reflection and Findings

### Summary of Findings

* **Publication Trends:** There was an exponential explosion of COVID-19-related research in 2020. The number of publications in 2020 dwarfed those from previous years, showing a massive global pivot by the scientific community.
* **Top Journals:** A small number of preprint servers and journals (like *bioRxiv*, *medRxiv*, and *PLOS ONE*) were responsible for a significant portion of the publications, highlighting the need for rapid dissemination of findings.
* **Top Sources:** The "Elsevier" and "CZI" (Chan Zuckerberg Initiative) sources were dominant, indicating their large-scale aggregation efforts.
* **Title Keywords:** Common themes in paper titles included "COVID-19," "virus," "SARS-CoV-2," "patients," "pandemic," and "detection," reflecting the primary focus of the research.

### Challenges and Learning

* **Challenge: Data Scale:** The primary challenge was the size of `metadata.csv`. At over 1.7GB, loading the entire file into memory for a Streamlit app is not feasible. It causes slow load times and high memory usage.
* **Solution:** The workflow was split. The heavy-lifting (loading, cleaning, and filtering) was done *once* in a Jupyter Notebook. This notebook then saved a much smaller (50,000-row sample) CSV file. The Streamlit app *only* loads this lightweight file, making it fast and responsive.
* **Learning:** This project was an excellent exercise in the standard data science workflow: **Explore -> Clean -> Analyze -> Present**. It highlights that a "dashboard" is often the final layer on top of a robust data processing pipeline. Streamlit makes it incredibly simple to build this final presentation layer, while `pandas` remains the workhorse for analysis.
