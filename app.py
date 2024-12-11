from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import logging

# Flask application setup
app = Flask(__name__)
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define domain and weight mappings
domain_mapping = {
    'Civil Engineering': ['Structural Engineering', 'Environmental Engineering', 'Geotechnical Engineering', 'Water Resources Engineering'],
    'Mechanical Engineering': ['Thermal Engineering', 'HVAC Systems', 'Manufacturing Engineering', 'Automobile Engineering', 'Aerospace Engineering'],
    'Electrical Engineering': ['Power Systems', 'Control Systems', 'Electronics Engineering', 'Communication Systems', 'Signal Processing', 'Power Electronics'],
    'Electronics & Communication Engineering': ['Signal Processing', 'Wireless Communication', 'Antenna Design', 'Embedded Systems'],
    'Electronics Engineering': ['Wireless Communication', 'Embedded Systems', 'Control Systems', 'Telecommunication Engineering'],
    'Computer Science': ['Data Science', 'Artificial Intelligence', 'Machine Learning', 'Cybersecurity', 'Software Engineering', 'Information Technology'],
    'Instrumentation Engineering': ['Control Systems Engineering', 'Automation Engineering', 'Process Control', 'Measurement Engineering'],
    'Chemical Engineering': ['Process Engineering', 'Biochemical Engineering', 'Environmental Engineering', 'Chemical Process Design'],
    'Agriculture': ['Agronomy', 'Sustainable Agriculture', 'Soil Science', 'Horticulture'],
    'Environmental Science': ['Environmental Geology', 'Environmental Botany', 'Environmental Engineering', 'Environmental Ecology'],
    'Botany': ['Environmental Botany', 'Plant Ecology', 'Plant Genetics', 'Plant Pathology'],
    'Library Science': ['Information Management', 'Knowledge Management', 'Digital Library Systems', 'Archival Science'],
    'Chemistry': ['Biochemistry', 'Chemical Process Design', 'Material Science'],
    'Geology': ['Environmental Geology', 'Earth Science', 'Geotechnical Engineering'],
    'Automobile Engineering': ['Electric Vehicle Technologies', 'Vehicle Design', 'Thermal Systems'],
    'Structural Engineering': ['Civil Engineering', 'Geotechnical Engineering', 'Construction Engineering'],
    'Cybersecurity': ['Computer Science', 'Information Security', 'Cryptography'],
    'Artificial Intelligence': ['Computer Science', 'Data Science', 'Machine Learning'],
    'Data Science': ['Artificial Intelligence', 'Machine Learning', 'Computer Science']
}
weight_mapping = {
    'Diploma': 1,
    'Bachelors Degree': 1.5,
    'Masters Degree': 2,
    'PhD Field': 3,
    'Years of Experience': 2,
    'Current Department of Work': 2,
}

def validate_file(file):
    # Check if the uploaded file is valid.
    if not file or file.filename == '':
        return "No file selected."
    if not file.filename.endswith('.xlsx'):
        return "Invalid file format. Only .xlsx files are allowed."
    return None

def save_file(file):
    #Save the uploaded file to the upload folder
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logger.info(f"File saved at {filepath}")
    return filepath

def get_field_similarity(field1, field2):
    #Calculate similarity using cosine similarity for text fields with domain mapping
    if field1 in domain_mapping and field2 in domain_mapping[field1]:
        return 0.8  # High similarity for related fields
    if field1 == field2:
        return 1.0  # Full similarity for exact match
    embeddings = model.encode([field1, field2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def calculate_weighted_score(candidate_row, professor_row):
    # Calculate weighted score based on candidate and professor profiles
    total_weight = sum(weight_mapping.values())
    weighted_score_sum = 0

    for field, weight in weight_mapping.items():
        candidate_field = candidate_row.get(field, '')
        professor_field = professor_row.get(field, '')

        # Calculate similarity for text fields or numeric distance
        if isinstance(candidate_field, str) and isinstance(professor_field, str):
            similarity_score = get_field_similarity(candidate_field, professor_field)
        elif field == 'Years of Experience':
            try:
                candidate_experience = int(candidate_field or 0)
                professor_experience = int(professor_field or 0)
                similarity_score = 1 - abs(candidate_experience - professor_experience) / 30.0
            except ValueError:
                similarity_score = 0
        else:
            similarity_score = 0

        weighted_score_sum += similarity_score * weight

    return weighted_score_sum / total_weight

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_interviewers', methods=['POST'])
def get_interviewers():
    file = request.files.get('file')
    error_message = validate_file(file)
    if error_message:
        return jsonify({"error": error_message}), 400

    try:
        filepath = save_file(file)
        candidates = pd.read_excel(filepath, sheet_name='Candidates')
        professors = pd.read_excel(filepath, sheet_name='Professors')

        candidate_name = request.form['candidate_name']
        panel_members_count = int(request.form['panel_count'])

        candidate_row = candidates[candidates['Name'] == candidate_name]
        if candidate_row.empty:
            return jsonify({"error": "Candidate not found in the uploaded file."}), 404

        candidate_row = candidate_row.iloc[0]
        relevance_scores_list = []

        for _, professor_row in professors.iterrows():
            score = calculate_weighted_score(candidate_row, professor_row)
            relevance_scores_list.append({
                'Professor': professor_row['Name'],
                'Relevance Score': score,
                'Years of Experience': professor_row.get('Years of Experience', 'N/A')
            })

        relevance_scores = pd.DataFrame(relevance_scores_list)
        top_professors = relevance_scores.sort_values(by=['Relevance Score', 'Years of Experience'], ascending=[False, False]).head(panel_members_count)
        return jsonify(top_professors.to_dict(orient='records'))

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({"error": "An error occurred while processing the file."}), 500

if __name__ == '__main__':
    # Ensure the uploads folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)









