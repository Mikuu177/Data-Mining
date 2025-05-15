# Student Mental Health Sentiment Analysis

This repository contains a data mining and sentiment analysis project focused on analyzing student counseling dialogues to identify mental health patterns and provide early intervention support.

## Project Overview

This project implements a sentiment analysis system for student mental health monitoring, using Natural Language Processing (NLP) techniques and machine learning models to analyze counseling dialogues, diaries, and forum posts. The goal is to automatically detect emotional patterns and potential mental health concerns in student communications.

## Dataset Information

The dataset used in this project consists of anonymized student counseling dialogues (transcripts), diaries, and forum posts. The primary dataset contains 449 dialogues for the pilot study, with plans to expand to 5,000+ samples for the full implementation.

**Dataset Source**: The data is derived from a combination of:
- University counseling center anonymized session transcripts
- Student diary entries (with consent)
- Public student forum discussions related to mental health
- Simulated counseling conversations for development purposes

All personal information has been removed, and the data has been preprocessed to ensure privacy compliance and ethical use.

## Installation Requirements

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Required Libraries
The project relies on the following Python libraries:

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
nltk>=3.6.0
textblob>=0.15.3
vaderSentiment>=3.3.2
xgboost>=1.4.0
transformers>=4.5.0 (for BERT/RoBERTa implementation)
```

## Installation Guide

1. Clone this repository:
```bash
git clone https://github.com/Mikuu177/Data-Mining.git
cd Data-Mining
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download NLTK resources (required for text processing):
```bash
python download_nltk.py
```

## Project Structure

```
Data-Mining/
├── README.md                                 # Project documentation
├── system_architecture.html                  # System architecture diagram
├── methodology_flowchart.html                # CRISP-DM methodology flowchart
├── programme_and_methodology.md              # Detailed methodology document
│
├── student_data/                             # Raw data directory
│   └── mental_health_data.csv                # Primary dataset
│
├── student_data.xlsx                         # Excel format of the dataset
├── cleaned_student_data.csv                  # Preprocessed dataset
│
├── download_nltk.py                          # Script to download NLTK resources
├── advanced_data_cleaning.py                 # Data preprocessing pipeline
├── student_sentiment_analysis.py             # Basic sentiment analysis implementation
├── improved_sentiment_analysis.py            # Enhanced sentiment analysis models
├── create_gantt_chart.py                     # Project planning visualization
│
├── figures/                                  # Visualizations directory
│   ├── polarity_subjectivity_scatter.png
│   ├── textblob_polarity_histogram.png
│   └── textblob_sentiment_distribution.png
│
├── sentiment_results/                        # Analysis results
│   ├── sentiment_analysis_report.json
│   ├── student_sentiment_results.csv
│   └── [various visualization files]
│
└── model_results/                            # Trained models and evaluation
    ├── best_sentiment_model.pkl
    ├── feature_scaler.pkl
    ├── tfidf_vectorizer.pkl
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── model_comparison.png
```

## Usage Guide

### Data Preprocessing

To clean and prepare the raw data:

```bash
python advanced_data_cleaning.py
```

This script performs:
- Text normalization (lowercase, punctuation removal)
- Stopword filtering
- Special character removal
- Data validation and cleaning

### Running Sentiment Analysis

#### Basic Analysis

```bash
python student_sentiment_analysis.py
```

This script:
- Loads the cleaned data
- Extracts sentiment using TextBlob
- Generates initial visualizations
- Outputs basic sentiment scores to sentiment_results/

#### Advanced Analysis

```bash
python improved_sentiment_analysis.py
```

This script:
- Implements multiple sentiment analysis models (VADER, ML models)
- Performs cross-validation
- Generates comparative visualizations
- Outputs detailed analysis to model_results/

### Viewing Results

1. Sentiment scores and classifications are saved in:
   - `sentiment_results/student_sentiment_results.csv`
   - `sentiment_results/sentiment_analysis_report.json`

2. Visualizations are available in:
   - `figures/` directory
   - `sentiment_results/` directory
   - `model_results/` directory

3. To view the system architecture and methodology:
   - Open `system_architecture.html` in a web browser
   - Open `methodology_flowchart.html` in a web browser

## Model Information

The project implements and evaluates multiple sentiment analysis approaches:

1. **Lexicon-based Methods**:
   - TextBlob: Provides polarity (-1 to +1) and subjectivity (0-1) scores
   - VADER: Specialized for social media and short texts

2. **Machine Learning Models**:
   - Logistic Regression (baseline model)
   - Random Forest
   - Support Vector Machines (SVM)
   - XGBoost (achieved highest accuracy in pilot study)

3. **Deep Learning** (implementation in progress):
   - BERT fine-tuning
   - RoBERTa
   - Model ensembles

## Results Interpretation

The sentiment analysis results are categorized as:
- **Negative**: Polarity score < -0.2
- **Neutral**: Polarity score between -0.2 and +0.2
- **Positive**: Polarity score > +0.2

The pilot study found:
- Distribution: 54.8% neutral, 35.4% positive, 9.8% negative sessions
- Best performing model: Logistic Regression with hybrid features (91.11% accuracy)
- Most important features: Sentiment polarity and lexical diversity

## Future Work

- Scale to 5,000+ dialogues
- Implement deep learning models
- Develop web application prototype
- Create interactive dashboard
- Integrate with counseling workflow

## Citation

If you use this code or dataset for your research, please cite:

```
@article{student_mental_health_2023,
  title={Sentiment Analysis of Student Counseling Conversations for Mental Health Monitoring},
  author={[Your Name]},
  journal={Data Mining Research Project},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or collaboration opportunities, please contact [Your Contact Information].
