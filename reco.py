import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load SpaCy's English language model for text preprocessing
nlp = spacy.load('en_core_web_md')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

df = pd.read_csv('final_2000.csv')

# Missing data handling and preprocessing
df['description'] = df['description'].fillna('').apply(preprocess_text)
df['skill_abr'] = df['skill_abr'].fillna('')
df['industry_id'] = df['industry_id'].fillna('')
df['speciality'] = df['speciality'].fillna('')

df['skill_abr'] = df['skill_abr'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['industry_id'] = df['industry_id'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['speciality'] = df['speciality'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Weighted Scoring System with Cosine Similarity
def calculate_relevance_score(row, input_description, salary_range=None, input_skills=None, input_work_type=None):
    score = 0
    
    # Cosine similarity between user input description and job description
    user_vector = nlp(input_description).vector
    job_vector = nlp(row['description']).vector
    similarity_score = cosine_similarity([user_vector], [job_vector])[0][0]
    
    # Weighing cosine similarity heavily, e.g., 50% of the total score
    score += similarity_score * 0.5
    
    # Salary matching score, weighted at 20%
    if salary_range:
        min_salary, max_salary = salary_range
        if min_salary <= row['med_salary'] <= max_salary:
            score += 0.2
    
    # Skill matching score, weighted at 25%
    if input_skills:
        matching_skills = len(set(input_skills).intersection(set(row['skill_abr'])))
        skill_score = (matching_skills / len(input_skills)) * 0.25
        score += skill_score
    
    # Work type matching score, weighted at 5%
    if input_work_type and row['work_type'] == input_work_type:
        score += 0.05
    
    return score

def create_relevance_labels(df, input_description, salary_range=None, input_skills=None, input_work_type=None):
    df['relevance_score'] = df.apply(
        calculate_relevance_score,
        axis=1,
        input_description=input_description,
        salary_range=salary_range,
        input_skills=input_skills,
        input_work_type=input_work_type
    )
    # Filter jobs with a minimum relevance score, e.g., 0.5
    df = df[df['relevance_score'] >= 0.5]
    return df

# Function to handle class imbalance using SMOTE
def balance_dataset(train_padded, y_train):
    # Check for class imbalance
    class_distribution = Counter(y_train)
    print(f"Original class distribution: {class_distribution}")

    if class_distribution[0] == 0 or class_distribution[1] == 0:
        print("One of the classes has no instances, skipping SMOTE.")
        return train_padded, y_train

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    train_padded_resampled, y_train_resampled = smote.fit_resample(train_padded, y_train)

    # Check the new class distribution
    new_class_distribution = Counter(y_train_resampled)
    print(f"New class distribution after SMOTE: {new_class_distribution}")

    return train_padded_resampled, y_train_resampled

def recommend_jobs(input_description, salary_range=None, input_skills=None, input_work_type=None, test_embeddings=None, tokenizer=None, model=None, test_df=None):
    input_description = preprocess_text(input_description)
    input_sequence = tokenizer.texts_to_sequences([input_description])
    max_length = test_embeddings.shape[1]  # Ensure max_length is consistent
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')

    input_embedding = model.predict(input_padded)[0]

    test_df['description_similarity'] = cosine_similarity(test_embeddings, [input_embedding]).flatten()

    test_df['feature_score'] = 0

    # Salary range-based scoring
    if salary_range is not None:
        min_salary, max_salary = salary_range
        test_df['feature_score'] += test_df.apply(
            lambda row: 1 - np.abs(row['med_salary'] - (min_salary + max_salary) / 2) / (max_salary - min_salary)
            if row['med_salary'] >= min_salary and row['med_salary'] <= max_salary else 0,
            axis=1
        )

    if input_skills is not None:
        test_df['feature_score'] += test_df['skill_abr'].apply(lambda x: len(set(x).intersection(set(input_skills))) / len(input_skills) if len(input_skills) > 0 else 0)

    if input_work_type is not None:
        test_df['feature_score'] += test_df['work_type'].apply(lambda x: 1 if x == input_work_type else 0)

    test_df['total_score'] = test_df['description_similarity'] + test_df['feature_score']

    recommendations = test_df.sort_values(by='total_score', ascending=False).head(10)

    return recommendations[['job_id', 'company_name', 'title', 'description', 'med_salary', 'skill_abr', 'work_type']]

def evaluate_model(test_padded, y_test, model):
    # Predict on the test set
    y_pred_prob = model.predict(test_padded)

    # Check if both classes are present in y_test
    if len(np.unique(y_test)) > 1:
        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_test, y_pred_prob)
        print(f"AUC-ROC: {auc_roc:.2f}")
    else:
        print("AUC-ROC: Not defined (only one class present in y_test)")

    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

def get_input():
    print("Please provide the following details to get job recommendations:")
    input_description = input("Enter a brief job description: ")

    # Ask for salary range
    input_min_salary = input("Enter the minimum expected salary: ")
    input_max_salary = input("Enter the maximum expected salary: ")
    salary_range = (int(input_min_salary), int(input_max_salary)) if input_min_salary and input_max_salary else None

    input_skills = input("Enter skills required (comma-separated): ").split(',') if input("Enter skills required (comma-separated): ") else None
    input_work_type = input("Enter the work type (e.g., Full-time, Part-time): ")

    # Filter jobs using cosine similarity and weighted scoring
    filtered_df = create_relevance_labels(df, input_description, salary_range, input_skills, input_work_type)

    if filtered_df.empty:
        print("No relevant jobs found based on the input criteria.")
        return

    # Train-test split on filtered data
    train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42, shuffle=True)

    # Text Tokenization and Padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['description'])
    vocab_size = len(tokenizer.word_index) + 1

    max_length = 100  # You can adjust this value based on your data
    train_sequences = tokenizer.texts_to_sequences(train_df['description'])
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')

    test_sequences = tokenizer.texts_to_sequences(test_df['description'])
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    # Extract the target labels
    y_train = train_df['relevance_score'].apply(lambda x: 1 if x >= 0.5 else 0).values
    y_test = test_df['relevance_score'].apply(lambda x: 1 if x >= 0.5 else 0).values

    # Handle class imbalance using SMOTE
    train_padded_balanced, y_train_balanced = balance_dataset(train_padded, y_train)

    # Build the RNN model for deeper analysis
    embedding_dim = 64

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, trainable=False),
        tf.keras.layers.LSTM(256, return_sequences=False, dropout=0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model with early stopping
    model.fit(train_padded_balanced, y_train_balanced, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

    # Generate embeddings for the descriptions using the RNN model
    train_embeddings = model.predict(train_padded_balanced)
    test_embeddings = model.predict(test_padded)

    # Recommend jobs based on RNN embeddings
    recommendations = recommend_jobs(
        input_description=input_description,
        salary_range=salary_range,
        input_skills=input_skills,
        input_work_type=input_work_type,
        test_embeddings=test_embeddings,
        tokenizer=tokenizer,  # Pass tokenizer
        model=model,  # Pass model
        test_df=test_df  # Pass test_df
    )

    recommendations.to_csv("recommendations.csv", index=False)

    # Evaluate the model performance using AUC-ROC and Accuracy
    evaluate_model(test_padded, y_test, model)

if __name__ == "__main__":
    get_input()
