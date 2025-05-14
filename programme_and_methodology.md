# Programme and Methodology

This document outlines the detailed methodology and system architecture for the Student Mental Health Sentiment Analysis project.

## System Architecture

The system architecture for this project is designed to process student counseling dialogues through multiple stages of analysis, from data collection to deployment of actionable insights. The architecture follows a modular approach to ensure flexibility and scalability.

### Visualization

The system architecture diagram visually represents the five key layers of our analysis pipeline:

1. **Data Layer**: Sources and collection processes
2. **Processing Layer**: Text preprocessing and feature engineering
3. **Model Layer**: From baseline lexicon methods to advanced deep learning
4. **Output Layer**: Analytics dashboards and alert systems
5. **Deployment Layer**: Integration with counseling workflows

View the interactive system architecture diagram: [System Architecture](system_architecture.html)

### Components

1. **Data Sources and Collection Layer**
   - Counseling Transcripts
   - Student Diaries
   - Forum Posts
   - Online Interactions
   - GDPR-Compliant Anonymization
   - Secure Aggregation of 5,000+ Dialogues

2. **Text Processing Layer**
   - Advanced Text Preprocessing (noise removal, tokenization, lemmatization, POS tagging)
   - Comprehensive Feature Engineering (TF-IDF vectors, sentiment scores, linguistic markers)

3. **Model Layer**
   - Lexicon-based Baselines (VADER, TextBlob, LIWC Dictionary)
   - Statistical Machine Learning Models (Logistic Regression, Random Forest, XGBoost, SVM)
   - Advanced Neural Approaches (BERT, RoBERTa, Ensemble Methods)

4. **Output and Analysis Layer**
   - Interactive Sentiment Analysis Dashboard
   - Temporal Trend Visualization
   - High-Risk Detection System
   - Counselor Notification Framework

5. **Deployment and Integration Layer**
   - Secure Web Application
   - Privacy-preserving API & Service Interfaces
   - Continuous Feedback Loop
   - Model Improvement Pipeline

## Methodology Flowchart

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which provides a structured approach to planning and implementing data mining projects. Our implementation spans six months with clearly defined phases and milestones.

### Visualization

The interactive methodology flowchart visualizes the seven phases of our CRISP-DM approach, including timeline, milestones, and specific activities for each phase:

View the interactive methodology flowchart: [Methodology Flowchart](methodology_flowchart.html)

### Phases and Activities

1. **Business Understanding (2 weeks)**
   - Define success criteria (≥80% sentiment classification accuracy)
   - Identify stakeholder requirements
   - Address privacy and security constraints
   - Set performance metrics for high-risk detection

2. **Data Understanding (3 weeks)**
   - Collect 5,000+ anonymized dialogues
   - Conduct exploratory sentiment analysis
   - Identify common themes (academic stress, etc.)
   - Assess data quality and representativeness

3. **Data Preparation (4 weeks)**
   - Text cleaning (lowercasing, stopword removal)
   - Feature engineering (TF-IDF, sentiment scores)
   - Statistical feature extraction (lexical diversity)
   - Class balancing (SMOTE for minority class)

4. **Modeling (6 weeks)**
   - Implement baseline lexicon-based methods
   - Train and optimize machine learning models
   - Fine-tune deep learning models
   - Create ensemble methods for improved performance
   - Perform feature selection

5. **Evaluation (3 weeks)**
   - Calculate performance metrics (F1-score, negative-class recall)
   - Conduct cross-validation for generalizability
   - Have experts review misclassifications
   - Compare models and select the best approach

6. **Deployment (4 weeks)**
   - Develop a web app prototype for real-time analysis
   - Create interactive dashboards for visualization
   - Implement an alert system for high-risk cases
   - Integrate with counseling workflow

7. **Reporting and Dissemination (2 weeks)**
   - Prepare technical reports and documentation
   - Draft academic manuscripts
   - Conduct stakeholder workshops
   - Provide recommendations for future research

## Project Milestones

The project is structured around four key milestones, as visualized in the methodology flowchart:

1. **M1 (End of Month 1)**: Completion of business understanding
   - Stakeholders' requirements gathered
   - Initial data collection begun
   - Deliverable: Requirement analysis document, data access secured

2. **M2 (End of Month 3)**: Completion of data preprocessing and preliminary model
   - Cleaned dataset ready
   - Initial results from baseline models
   - Deliverable: Data description and exploratory analysis report, baseline model evaluation

3. **M3 (Mid Month 5)**: Full model evaluation and prototype system
   - Best model selected and evaluated
   - Simple prototype application functional for testing
   - Deliverable: Model comparison results, confusion matrices, prototype demo

4. **M4 (End of Month 6)**: Final report and system presentation
   - Project completion
   - Deliverable: Final project report, stakeholder presentation, code documentation

Throughout the project, we maintain close communication with stakeholders and domain experts to ensure that the outcomes are not only academically robust but also practically relevant for improving student mental health support. 