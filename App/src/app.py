import re
from pathlib import Path
from urllib.parse import urlparse
import sys

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Compatibility fix for scikit-learn version mismatch
def _patch_simple_imputer():
    if not hasattr(SimpleImputer, '_fill_dtype'):
        original_getattribute = SimpleImputer.__getattribute__

        def patched_getattribute(self, name):
            if name == '_fill_dtype':
                if hasattr(self, 'statistics_') and self.statistics_ is not None:
                    return np.asarray(self.statistics_).dtype
                return np.float64
            return original_getattribute(self, name)

        SimpleImputer.__getattribute__ = patched_getattribute

_patch_simple_imputer()

app = Flask(__name__)
CORS(app)

REQUEST_TIMEOUT = 15
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error. Check backend logs.'}), 500

# Paths - ROOT_DIR is Major project 2 folder (2 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]
# Ensure project root is importable (needed for unpickling custom helpers)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODEL_PATH_DEMOGRAPHIC = ROOT_DIR / "fake_classifier.pkl"
MODEL_PATH_TEXT = ROOT_DIR / "fake_job_classifier.pkl"
MODEL_PATH_SALARY = ROOT_DIR / "salary_predictor.pkl"
MODEL_PATH_SALARY_INDIA = ROOT_DIR / "salary_predictor_india.pkl"
DATASET_PATH = ROOT_DIR / "Datasets" / "finalDataset.csv"
SALARY_DATASET_PATH = ROOT_DIR / "Datasets" / "Salary_Data_Based_country_and_race.csv"
FAKE_JOBS_DATASET_PATH = ROOT_DIR / "Datasets" / "fake_job_postings.csv"

# Load demographic model (Major project)
fake_classifier = None
try:
    fake_classifier = joblib.load(MODEL_PATH_DEMOGRAPHIC)
    print(f"[OK] Demographic model loaded from {MODEL_PATH_DEMOGRAPHIC}")
except Exception as e:
    print(f"[WARN] Demographic model not loaded: {e}")

# Load text/NLP model (FakeJobDetector) - BiLSTM or TF-IDF+LogReg
text_pipeline = None
MODEL_PATH_BILSTM = ROOT_DIR / "fake_job_bilstm.keras"
try:
    if MODEL_PATH_BILSTM.exists():
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        from bilstm_predictor import load_bilstm_predictor
        text_pipeline = load_bilstm_predictor()
        print(f"[OK] BiLSTM text model loaded from {MODEL_PATH_BILSTM}")
    else:
        text_pipeline = joblib.load(MODEL_PATH_TEXT)
        print(f"[OK] Text model loaded from {MODEL_PATH_TEXT}")
except Exception as e:
    try:
        text_pipeline = joblib.load(MODEL_PATH_TEXT)
        print(f"[OK] Text model loaded from {MODEL_PATH_TEXT}")
    except Exception as e2:
        print(f"[WARN] Text model not loaded: {e}")

# Load salary prediction model (regression)
salary_model = None
try:
    salary_model = joblib.load(MODEL_PATH_SALARY)
    print(f"[OK] Salary model loaded from {MODEL_PATH_SALARY}")
except Exception as e:
    print(f"[WARN] Salary model not loaded: {e}")

# Load India salary model (preferred, role+skills+location+experience)
salary_model_india = None
try:
    salary_model_india = joblib.load(MODEL_PATH_SALARY_INDIA)
    print(f"[OK] India salary model loaded from {MODEL_PATH_SALARY_INDIA}")
except Exception as e:
    print(f"[WARN] India salary model not loaded: {e}")

# Load datasets
datasets = {}
def load_datasets():
    global datasets
    try:
        if DATASET_PATH.exists():
            datasets['final'] = pd.read_csv(DATASET_PATH)
            print(f"[OK] Loaded finalDataset.csv: {len(datasets['final'])} rows")
    except Exception as e:
        print(f"[ERROR] finalDataset: {e}")
    try:
        if SALARY_DATASET_PATH.exists():
            datasets['salary'] = pd.read_csv(SALARY_DATASET_PATH)
            print(f"[OK] Loaded Salary dataset: {len(datasets['salary'])} rows")
    except Exception as e:
        print(f"[ERROR] Salary dataset: {e}")
    try:
        if FAKE_JOBS_DATASET_PATH.exists():
            datasets['jobs'] = pd.read_csv(FAKE_JOBS_DATASET_PATH)
            print(f"[OK] Loaded fake_job_postings.csv: {len(datasets['jobs'])} rows")
    except Exception as e:
        print(f"[ERROR] fake_job_postings: {e}")

load_datasets()

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def combine_job_text(title="", company_profile="", description="", requirements=""):
    parts = [clean_text(title), clean_text(company_profile), clean_text(description), clean_text(requirements)]
    combined = ' '.join(p for p in parts if p)
    return combined if combined else "unknown"

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False

def fetch_and_extract_text(url):
    headers = {'User-Agent': USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Could not connect to the URL.")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 'unknown'
        raise ValueError(f"HTTP error {code}.")
    except Exception as e:
        raise ValueError(f"Failed to fetch: {str(e)}")

    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    parts = []
    title_tag = soup.find('title')
    if title_tag:
        parts.append(title_tag.get_text())
    meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    if meta and meta.get('content'):
        parts.append(meta['content'])

    job_selectors = [
        '[class*="job-description"]', '[class*="job-content"]', '[class*="job-detail"]',
        '[class*="job-body"]', '[class*="job-info"]', 'article', 'main', '[role="main"]',
        '.content', '.post-content', '.job-posting'
    ]
    main_text = ""
    for sel in job_selectors:
        try:
            els = soup.select(sel)
            for el in els[:3]:
                t = el.get_text(separator=' ', strip=True)
                if len(t) > 100:
                    main_text += t + " "
        except Exception:
            pass
    if len(main_text) < 200:
        body = soup.find('body')
        if body:
            main_text = body.get_text(separator=' ', strip=True)
    parts.append(main_text)
    combined = ' '.join(p for p in parts if p)
    return combined[:50000]

def _to_json_safe(records):
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, (np.integer, np.int64)):
                r[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                r[k] = float(v)
            elif isinstance(v, np.ndarray):
                r[k] = v.tolist()
            elif pd.isna(v):
                r[k] = None
    return records

# ---- Endpoints ----

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'demographic_model_loaded': fake_classifier is not None,
        'text_model_loaded': text_pipeline is not None,
        'salary_model_loaded': salary_model is not None,
        'salary_model_india_loaded': salary_model_india is not None,
        'datasets_loaded': len(datasets) > 0
    })

# Demographic-based detection (Major project)
@app.route('/api/detect-fake', methods=['POST'])
def detect_fake_demographic():
    if fake_classifier is None:
        return jsonify({'error': 'Demographic model not loaded.'}), 500
    try:
        data = request.json
        feature_cols_num = ["Age", "Education_Level_Code", "Years of Experience", "Senior", "Salary_PPP_Adjusted", "PPP_Index"]
        feature_cols_cat = ["Gender", "Education_Level", "Job Title", "Country", "Race"]
        features = {
            'Age': float(data.get('age', 30)),
            'Education_Level_Code': float(data.get('education_level_code', 1.0)),
            'Years of Experience': float(data.get('years_of_experience', 5.0)),
            'Senior': float(data.get('senior', 0.0)),
            'Salary_PPP_Adjusted': float(data.get('salary_ppp_adjusted', 80000.0)),
            'PPP_Index': float(data.get('ppp_index', 1.0)),
            'Gender': data.get('gender', 'Male'),
            'Education_Level': data.get('education_level', 'Bachelors'),
            'Job Title': data.get('job_title', 'Software Engineer'),
            'Country': data.get('country', 'USA'),
            'Race': data.get('race', 'White')
        }
        column_order = feature_cols_num + feature_cols_cat
        df = pd.DataFrame([features])[column_order]
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        prediction = fake_classifier.predict(df)[0]
        probs = fake_classifier.predict_proba(df)[0]
        fake_prob = float(probs[1])
        risk = 'High' if fake_prob > 0.7 else 'Medium' if fake_prob > 0.4 else 'Low'
        return jsonify({
            'is_fake': bool(prediction == 1),
            'confidence': {'fake': round(fake_prob * 100, 2), 'real': round(float(probs[0]) * 100, 2)},
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'risk_level': risk,
            'features_used': features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Salary prediction (regression)
@app.route('/api/predict-salary', methods=['POST'])
def predict_salary():
    if salary_model_india is None and salary_model is None:
        return jsonify({'error': 'No salary model loaded. Run train_salary_model_india.py (preferred) or train_salary_model.py.'}), 500
    try:
        data = request.json or {}
        # Preferred: India model (annual INR salary)
        if salary_model_india is not None:
            title = (data.get("title") or data.get("job_title") or "").strip()
            skills = (data.get("skills") or data.get("tagsAndSkills") or "").strip()
            location = (data.get("location") or "").strip()

            # Validation/clamping
            def _clamp_float(v, lo, hi, default):
                try:
                    v = float(v)
                except Exception:
                    return float(default)
                if np.isnan(v) or np.isinf(v):
                    return float(default)
                return float(min(hi, max(lo, v)))

            min_exp = _clamp_float(data.get("minimumExperience", data.get("min_experience", 0)), 0, 40, 0)
            max_exp = _clamp_float(data.get("maximumExperience", data.get("max_experience", min_exp)), 0, 40, min_exp)
            if max_exp < min_exp:
                max_exp = min_exp

            reviews = _clamp_float(data.get("ReviewsCount", 0), 0, 5_000_000, 0)
            rating = _clamp_float(data.get("AggregateRating", 0), 0, 5, 0)

            if not title:
                return jsonify({"error": "Please provide a job title."}), 400
            if not skills:
                return jsonify({"error": "Please provide skills/tags (comma-separated)."}), 400
            if not location:
                location = "India"

            features = {
                "title": title,
                "tagsAndSkills": skills,
                "location": location,
                "minimumExperience": min_exp,
                "maximumExperience": max_exp,
                "ReviewsCount": reviews,
                "AggregateRating": rating,
            }
            df = pd.DataFrame([features])

            annual_inr = float(salary_model_india.predict(df)[0])
            annual_inr = max(0.0, annual_inr)
            low = annual_inr * 0.9
            high = annual_inr * 1.1
            monthly_inr = annual_inr / 12.0

            return jsonify({
                "predicted_annual_inr": round(annual_inr, 2),
                "predicted_lpa": round(annual_inr / 100000.0, 2),
                "predicted_monthly_inr": round(monthly_inr, 2),
                "range_annual_inr": {"low": round(low, 2), "high": round(high, 2)},
                "range_monthly_inr": {"low": round(low / 12.0, 2), "high": round(high / 12.0, 2)},
                "currency_note": "India salary model trained on job title + skills + location + experience. Output is INR per annum (and derived INR/month, LPA).",
                "features_used": features,
                "model": "india"
            })

        # Fallback: old PPP-adjusted model (kept for compatibility)
        feature_cols_num = ["Age", "Education_Level_Code", "Years of Experience", "Senior", "PPP_Index"]
        feature_cols_cat = ["Gender", "Education_Level", "Job Title", "Country", "Race"]
        features = {
            "Age": float(data.get("age", 30)),
            "Education_Level_Code": float(data.get("education_level_code", 1.0)),
            "Years of Experience": float(data.get("years_of_experience", 3.0)),
            "Senior": float(data.get("senior", 0.0)),
            "PPP_Index": float(data.get("ppp_index", 1.0)),
            "Gender": data.get("gender", "Male"),
            "Education_Level": data.get("education_level", "Bachelors"),
            "Job Title": data.get("job_title", "Software Engineer"),
            "Country": data.get("country", "USA"),
            "Race": data.get("race", "White"),
        }
        df = pd.DataFrame([features])[feature_cols_num + feature_cols_cat]
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        pred = float(salary_model.predict(df)[0])
        low = max(0.0, pred * 0.9)
        high = pred * 1.1
        return jsonify({
            "predicted_salary_ppp_adjusted": round(pred, 2),
            "range": {"low": round(low, 2), "high": round(high, 2)},
            "currency_note": "PPP-adjusted annual salary estimate (fallback model).",
            "features_used": features,
            "model": "ppp_fallback"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Text-based detection (FakeJobDetector)
@app.route('/api/detect', methods=['POST'])
def detect_fake_text():
    if text_pipeline is None:
        return jsonify({'error': 'Text model not loaded. Run train_model.py first.'}), 500
    try:
        data = request.json or {}
        combined = combine_job_text(
            data.get('title', ''),
            data.get('company_profile', ''),
            data.get('description', ''),
            data.get('requirements', '')
        )
        if len(combined) < 10:
            return jsonify({'error': 'Please provide at least a job title and description.'}), 400
        prediction = text_pipeline.predict([combined])[0]
        proba = text_pipeline.predict_proba([combined])[0]
        fake_prob = float(proba[1])
        risk = 'High' if fake_prob > 0.7 else 'Medium' if fake_prob > 0.4 else 'Low'
        return jsonify({
            'is_fake': bool(prediction == 1),
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': {'fake': round(fake_prob * 100, 2), 'real': round(float(proba[0]) * 100, 2)},
            'risk_level': risk,
            'message': f"This job posting appears to be {'FAKE - High scam risk!' if prediction == 1 else 'likely REAL.'} ({risk} risk)"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# URL-based detection (FakeJobDetector)
@app.route('/api/detect-url', methods=['POST'])
def detect_from_url():
    if text_pipeline is None:
        return jsonify({'error': 'Text model not loaded.'}), 500
    try:
        data = request.json or {}
        url = (data.get('url') or '').strip()
        if not url:
            return jsonify({'error': 'Please provide a URL.'}), 400
        if not is_valid_url(url):
            return jsonify({'error': 'Invalid URL. Use http:// or https://'}), 400
        extracted = fetch_and_extract_text(url)
        if len(extracted) < 50:
            return jsonify({'error': 'Could not extract enough text from the page.'}), 400
        combined = clean_text(extracted) or extracted[:5000]
        prediction = text_pipeline.predict([combined])[0]
        proba = text_pipeline.predict_proba([combined])[0]
        fake_prob = float(proba[1])
        risk = 'High' if fake_prob > 0.7 else 'Medium' if fake_prob > 0.4 else 'Low'
        return jsonify({
            'is_fake': bool(prediction == 1),
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': {'fake': round(fake_prob * 100, 2), 'real': round(float(proba[0]) * 100, 2)},
            'risk_level': risk,
            'message': f"This job posting appears to be {'FAKE!' if prediction == 1 else 'likely REAL.'} ({risk} risk)",
            'url_analyzed': url,
            'text_extracted': True
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Dataset stats - unified for both projects
@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
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
            'job_titles': df['Job Title'].unique().tolist()[:20] if 'Job Title' in df.columns else []
        }
    if 'salary' in datasets:
        df = datasets['salary']
        stats['salary'] = {
            'total_records': len(df),
            'columns': list(df.columns),
            'countries': df['Country'].unique().tolist() if 'Country' in df.columns else []
        }
    if 'jobs' in datasets:
        df = datasets['jobs']
        fake_c = int(df['fraudulent'].sum()) if 'fraudulent' in df.columns else 0
        stats['jobs'] = {
            'total_records': len(df),
            'fake_jobs': fake_c,
            'real_jobs': len(df) - fake_c,
            'fake_percentage': round(fake_c / len(df) * 100, 2) if len(df) > 0 else 0
        }
    return jsonify(stats)

@app.route('/api/dataset/sample', methods=['GET'])
def get_dataset_sample():
    dataset_type = request.args.get('type', 'final')
    limit = min(int(request.args.get('limit', 10)), 50)
    fake_only = request.args.get('fake_only', 'false').lower() == 'true'

    if dataset_type == 'jobs' and 'jobs' in datasets:
        df = datasets['jobs']
        if fake_only and 'fraudulent' in df.columns:
            df = df[df['fraudulent'] == 1]
        sample = df.head(limit)
        records = sample.to_dict('records')
        _to_json_safe(records)
        return jsonify({'data': records, 'count': len(records), 'dataset': 'jobs'})

    if dataset_type not in datasets:
        return jsonify({'error': f'Dataset {dataset_type} not found'}), 404
    df = datasets[dataset_type]
    sample = df.head(limit)
    records = sample.to_dict('records')
    _to_json_safe(records)
    return jsonify({'dataset': dataset_type, 'count': len(records), 'data': records})

@app.route('/api/dataset/filter', methods=['GET'])
def get_dataset_filtered():
    if 'final' not in datasets:
        return jsonify({'error': 'Final dataset not loaded'}), 500
    df = datasets['final']
    country = request.args.get('country')
    job_title = request.args.get('job_title')
    fake_only = request.args.get('fake_only', 'false').lower() == 'true'
    real_only = request.args.get('real_only', 'false').lower() == 'true'
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    filtered = df
    if country and 'Country' in filtered.columns:
        filtered = filtered[filtered['Country'] == country]
    if job_title and 'Job Title' in filtered.columns:
        filtered = filtered[filtered['Job Title'] == job_title]
    if 'Fake_Job_Risk' in filtered.columns:
        if fake_only and not real_only:
            filtered = filtered[filtered['Fake_Job_Risk'] == 1]
        elif real_only and not fake_only:
            filtered = filtered[filtered['Fake_Job_Risk'] == 0]
    total = int(len(filtered))
    filtered = filtered.iloc[offset:offset + limit]
    records = filtered.to_dict('records')
    _to_json_safe(records)
    return jsonify({'dataset': 'final', 'total_matches': total, 'count': len(records), 'offset': offset, 'data': records})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Major Project 2 - Integrated Fake Job Detection API")
    print("="*50)
    print(f"Demographic model: {MODEL_PATH_DEMOGRAPHIC.exists()}")
    print(f"Text model: {MODEL_PATH_TEXT.exists()}")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
