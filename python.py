import pandas as pd

# Load the CSV files
postings_df = pd.read_csv('postings.csv')
merged_companies_df = pd.read_csv('merged_companies.csv')
merged_jobs_df = pd.read_csv('merged_jobs.csv')

# Merge postings with merged_jobs on job_id
final_df = pd.merge(postings_df, merged_jobs_df, on='job_id', how='left')

# Merge the result with merged_companies on company_id
final_df = pd.merge(final_df, merged_companies_df, on='company_id', how='left')

# Rename and select the correct columns
final_df = final_df.rename(columns={
    'description_x': 'description',
    'max_salary_y': 'max_salary',
    'min_salary_y': 'min_salary',
    'med_salary_y': 'med_salary',
})

# Select the required columns
final_df = final_df[[
    'job_id',
    'company_name',
    'company_id',
    'title',
    'description',
    'location',
    'formatted_work_type',
    'remote_allowed',
    'job_posting_url',
    'application_type',
    'work_type',
    'max_salary',
    'min_salary',
    'med_salary',
    'industry_id',
    'skill_abr',
    'type',
    'company_size',
    'industry',
    'speciality'
]]

# Handle missing values
final_df['remote_allowed'].fillna(0, inplace=True)  # Set null remote_allowed to 0

# Save the final DataFrame to a new CSV file
final_df.to_csv('final_merged_postings.csv', index=False)

print("Final merged postings CSV file created successfully as 'final_merged_postings.csv'")

