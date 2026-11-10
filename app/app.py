"""Local Flask interface for the trained SMS spam classifier."""
from pathlib import Path
import joblib
from flask import Flask, render_template, request
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'models' / 'spam_classifier.joblib'
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    message = ''
    label = None
    error = None
    if request.method == 'POST':
        message = request.form.get('message','').strip()
        if not message:
            error='Introduce un mensaje.'
        elif len(message)>5000:
            error='El mensaje supera los 5000 caracteres.'
        elif not MODEL.exists():
            error='Primero ejecuta python train.py para generar el modelo.'
        else:
            classifier=joblib.load(MODEL)  # trusted local artifact only
            label='SPAM' if int(classifier.predict([message])[0])==1 else 'HAM'
    return render_template('index.html',message=message,label=label,error=error)

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
