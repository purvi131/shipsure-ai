after cloning run this in bash 
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib
python backend/data/generate_data.py
python backend/model/train_model.py
uvicorn backend.api.main:app --reload
