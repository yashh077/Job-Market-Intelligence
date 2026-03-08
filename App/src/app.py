from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Compatibility fix for scikit-learn version mismatch
# This fixes the '_fill_dtype' attribute error when loading models trained with older versions
def _patch_simple_imputer():
    """Patch SimpleImputer to add missing _fill_dtype attribute for compatibility"""
    if not hasattr(SimpleImputer, '_fill_dtype'):
        # Store original __getattribute__
        original_getattribute = SimpleImputer.__getattribute__
        
        def patched_getattribute(self, name):
            if name == '_fill_dtype':
                # Return dtype based on statistics_ if available, else default to float64
                if hasattr(self, 'statistics_') and self.statistics_ is not None:
                    return np.asarray(self.statistics_).dtype
                return np.float64
            # For all other attributes, use original behavior
            return original_getattribute(self, name)
        
        # Replace __getattribute__ with our patched version
        SimpleImputer.__getattribute__ = patched_getattribute

# Apply the patch before loading models
_patch_simple_imputer()

# Get the project root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "fake_classifier.pkl"
DATASET_PATH = ROOT_DIR / "Datasets" / "finalDataset.csv"
SALARY_DATASET_PATH = ROOT_DIR / "Datasets" / "Salary_Data_Based_country_and_race.csv"

# Load the model
try:
    fake_classifier = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    fake_classifier = None

# Load datasets for statistics
def load_datasets():
    datasets = {}
    try:
        if DATASET_PATH.exists():
            datasets['final'] = pd.read_csv(DATASET_PATH)
            print(f"[OK] Loaded finalDataset.csv: {len(datasets['final'])} rows")
    except Exception as e:
        print(f"[ERROR] Error loading finalDataset: {e}")
    
    try:
        if SALARY_DATASET_PATH.exists():
            datasets['salary'] = pd.read_csv(SALARY_DATASET_PATH)
            print(f"[OK] Loaded Salary_Data_Based_country_and_race.csv: {len(datasets['salary'])} rows")
    except Exception as e:
        print(f"[ERROR] Error loading salary dataset: {e}")
    
    return datasets

datasets = load_datasets()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': fake_classifier is not None,
        'datasets_loaded': len(datasets) > 0
    })

@app.route('/api/detect-fake', methods=['POST'])
def detect_fake():
    """Detect if a job posting is fake based on provided features"""
    if fake_classifier is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure fake_classifier.pkl exists.'
        }), 500
    
    try:
        data = request.json
        
        # Extract features from request
        # IMPORTANT: Features must be in the exact order expected by the model
        # Numeric features first, then categorical (matching train_models.py)
        feature_cols_num = [
            "Age",
            "Education_Level_Code",
            "Years of Experience",
            "Senior",
            "Salary_PPP_Adjusted",
            "PPP_Index",
        ]
        feature_cols_cat = [
            "Gender",
            "Education_Level",
            "Job Title",
            "Country",
            "Race",
        ]
        
        # Build features dictionary in correct order
        features = {}
        
        # Add numeric features first
        features['Age'] = float(data.get('age', 30))
        features['Education_Level_Code'] = float(data.get('education_level_code', 1.0))
        features['Years of Experience'] = float(data.get('years_of_experience', 5.0))
        features['Senior'] = float(data.get('senior', 0.0))
        features['Salary_PPP_Adjusted'] = float(data.get('salary_ppp_adjusted', 80000.0))
        features['PPP_Index'] = float(data.get('ppp_index', 1.0))
        
        # Add categorical features
        features['Gender'] = data.get('gender', 'Male')
        features['Education_Level'] = data.get('education_level', 'Bachelors')
        features['Job Title'] = data.get('job_title', 'Software Engineer')
        features['Country'] = data.get('country', 'USA')
        features['Race'] = data.get('race', 'White')
        
        # Create DataFrame with single row, ensuring correct column order
        # Order: numeric features first, then categorical (as in training)
        column_order = feature_cols_num + feature_cols_cat
        df = pd.DataFrame([features])
        # Reorder columns to match training data structure
        df = df[column_order]
        
        # Replace any infinite values
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        
        # Make prediction
        prediction = fake_classifier.predict(df)[0]
        probabilities = fake_classifier.predict_proba(df)[0]
        
        # Get probability of being fake (class 1)
        fake_probability = float(probabilities[1])
        real_probability = float(probabilities[0])
        
        result = {
            'is_fake': bool(prediction == 1),
            'confidence': {
                'fake': round(fake_probability * 100, 2),
                'real': round(real_probability * 100, 2)
            },
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'risk_level': 'High' if fake_probability > 0.7 else 'Medium' if fake_probability > 0.4 else 'Low',
            'features_used': features
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 400

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Get statistics about the datasets"""
    stats = {}
    
    if 'final' in datasets:
        df = datasets['final']
        stats['final'] = {
            'total_records': len(df),
            'fake_jobs': int(df['Fake_Job_Risk'].sum()) if 'Fake_Job_Risk' in df.columns else 0,
            'real_jobs': int((df['Fake_Job_Risk'] == 0).sum()) if 'Fake_Job_Risk' in df.columns else 0,
            'fake_percentage': round((df['Fake_Job_Risk'].sum() / len(df)) * 100, 2) if 'Fake_Job_Risk' in df.columns else 0,
            'columns': list(df.columns),
            'countries': df['Country'].unique().tolist() if 'Country' in df.columns else [],
            'job_titles': df['Job Title'].unique().tolist()[:20] if 'Job Title' in df.columns else []  # First 20
        }
    
    if 'salary' in datasets:
        df = datasets['salary']
        stats['salary'] = {
            'total_records': len(df),
            'columns': list(df.columns),
            'countries': df['Country'].unique().tolist() if 'Country' in df.columns else []
        }
    
    return jsonify(stats)

@app.route('/api/dataset/sample', methods=['GET'])
def get_dataset_sample():
    """Get a sample of records from the dataset"""
    dataset_type = request.args.get('type', 'final')
    limit = int(request.args.get('limit', 10))
    
    if dataset_type not in datasets:
        return jsonify({'error': f'Dataset type {dataset_type} not found'}), 404
    
    df = datasets[dataset_type]
    sample = df.head(limit)
    
    # Convert to JSON-serializable format
    result = sample.to_dict('records')
    
    # Convert numpy types to native Python types
    for record in result:
        for key, value in record.items():
            if isinstance(value, (np.integer, np.int64)):
                record[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                record[key] = float(value)
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif pd.isna(value):
                record[key] = None
    
    return jsonify({
        'dataset': dataset_type,
        'count': len(result),
        'data': result
    })


@app.route('/api/dataset/filter', methods=['GET'])
def get_dataset_filtered():
    """Get filtered records from the main dataset for frontend display."""
    if 'final' not in datasets:
        return jsonify({'error': 'Final dataset not loaded'}), 500

    df = datasets['final']

    # Query params
    country = request.args.get('country')
    job_title = request.args.get('job_title')
    fake_only = request.args.get('fake_only', 'false').lower() == 'true'
    real_only = request.args.get('real_only', 'false').lower() == 'true'
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    filtered = df

    # Apply filters safely only if columns exist
    if country and 'Country' in filtered.columns:
        filtered = filtered[filtered['Country'] == country]

    if job_title and 'Job Title' in filtered.columns:
        filtered = filtered[filtered['Job Title'] == job_title]

    if 'Fake_Job_Risk' in filtered.columns:
        if fake_only and not real_only:
            filtered = filtered[filtered['Fake_Job_Risk'] == 1]
        elif real_only and not fake_only:
            filtered = filtered[filtered['Fake_Job_Risk'] == 0]

    total_matches = int(len(filtered))

    # Pagination
    filtered = filtered.iloc[offset:offset + limit]

    records = filtered.to_dict('records')

    # Convert numpy / NaN types for JSON
    for record in records:
        for key, value in record.items():
            if isinstance(value, (np.integer, np.int64)):
                record[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                record[key] = float(value)
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif pd.isna(value):
                record[key] = None

    return jsonify({
        'dataset': 'final',
        'total_matches': total_matches,
        'count': len(records),
        'offset': offset,
        'data': records
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Flask Backend Server")
    print("="*50)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {MODEL_PATH.exists()}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Dataset exists: {DATASET_PATH.exists()}")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')

