"""Local Python deployment. Run python app.py and visit http://127.0.0.1:5000."""
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from inference import Recommender

ROOT=Path(__file__).resolve().parent
app=Flask(__name__,static_folder=str(ROOT/'web'),static_url_path='')
app.config['MAX_CONTENT_LENGTH']=64*1024
model=Recommender()

@app.get('/')
def home():return send_from_directory(ROOT/'web','index.html')

@app.get('/model.json')
def artifact():return send_from_directory(ROOT/'artifacts','model.json')

@app.get('/api/health')
def health():return jsonify(status='ok',model=model.model['primary_model'],movies=model.model['eligible_movies'])

@app.post('/api/recommend')
def recommend():
    try:return jsonify(model.recommend(request.get_json(silent=True)))
    except (ValueError,TypeError) as exc:return jsonify(error=str(exc)),400

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
