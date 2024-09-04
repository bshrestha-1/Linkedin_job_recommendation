# Job Recommendation System

This project implements a job recommendation system using natural language processing (NLP) and a Recurrent Neural Network (RNN). The system takes user inputs such as a job description, expected salary range, required skills, and work type to provide relevant job recommendations. It uses various techniques, including cosine similarity, weighted scoring, and model evaluation, to deliver accurate and personalized job recommendations.

## Features

- **Text Preprocessing**: Utilizes SpaCy to preprocess job descriptions by lemmatizing and removing stop words.
- **Cosine Similarity**: Measures the similarity between the dataset's user input description and job descriptions.
- **Weighted Scoring**: Combines factors (similarity, salary, skills, work type) into a relevance score.
- **Handling Imbalanced Data**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset, if applicable.
- **RNN-based Embeddings**: An RNN generates embeddings for deeper analysis and recommendation.
- **Model Evaluation**: Evaluates model performance using AUC-ROC and accuracy metrics.

## Usage

Upon running the script, you will be prompted to provide the following details:

  - Job Description: A brief description of the job you are interested in.
  - Salary Range: The minimum and maximum expected salary.
  - Skills Required: A comma-separated list of important skills for the job.
  - Work Type: The type of work (e.g., Full-time, Part-time).

The script will then process this information, filter relevant jobs, and generate recommendations. The recommendations will be saved in a file called recommendations.csv.

