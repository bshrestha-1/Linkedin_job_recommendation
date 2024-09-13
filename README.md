# Job Recommendation System

This project was my Capstone project for a Data Mining Course (CS5593) in Fall 2019 I took at the university. This used to be a simple recommendation program that used cosine similarity and clustering algorithms based on simpler features. I've recently revamped the project, applying new skills I picked up at work, like using TensorFlow to boost the system’s accuracy. The system is designed to feel intuitive and personal—taking in details like job descriptions, salary expectations, skills, and work preferences, it then delivers tailored job recommendations. I used techniques like cosine similarity, weighted scoring, and careful model evaluation to ensure the suggestions felt relevant and on point, making finding the right job easier and more personalized. The data for the code comes from web scrapping using [this GitHub project](https://github.com/bshrestha-1/linkedin_scraper). 

   >> This project is still under construction. I am fixing certain issues that I have been facing by incorporating TensorFlow. 

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

The script will then process this information, filter relevant jobs, and generate recommendations. The recommendations will be saved in a file called recommendations.csv.

