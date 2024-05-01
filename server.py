
from __future__ import print_function # In python 2.7
import numpy as np
import pickle
import os
from flask import Flask,redirect,url_for,render_template,request,jsonify
from flask import Response
from flask import send_from_directory
import time
import os
# !python -m spacy download en_core_web_md
import spacy
nlp = spacy.load("en_core_web_sm")
import cv2
import subprocess
from deepface import DeepFace
import pandas as pd
import json
import sys
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf
import speech_recognition as speechrecognizer
import speech_recognition as sr

import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tfidf_vectorizer = TfidfVectorizer()

from flask import Flask
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import os
from flask import Flask
import secrets
import string
from flask import flash
import mysql.connector
from flask import Flask, render_template, session, redirect, url_for
from flask import redirect, url_for, render_template
from flask_login import current_user
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
from flask_login import current_user, login_required
from flask import Flask
from flask_login import LoginManager, UserMixin
from flask import Flask, render_template
from flask_mysqldb import MySQL
from flask_login import LoginManager, current_user
from speech_emotion_recognition import *

 # Import your function to fetch user name
app = Flask(__name__)

# Configure MySQL database URI
# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Varungazala@16'
app.config['MYSQL_DB'] = 'Intel'  # Database name
app.config['MYSQL_PORT'] = 3306  # Change this port if needed
mysql = MySQL(app)

login_manager = LoginManager(app)

# Set the secret key from an environment variable

def generate_secret_key(length=24):
    alphabet = string.ascii_letters + string.digits + '!@#$%^&*()-_=+'
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# Set the secret key
app.secret_key = generate_secret_key()

app.secret_key = "super secret key"

# @app.route('/')
# def home():
#     return render_template('login.html')
# Login route
# Secret key for session management
@app.route('/')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM register WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        cur.close()
        
        if user:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')  # Flash success message
            time.sleep(2)  # Wait for 2 seconds before redirecting
            return render_template('Main_page.html')  # Redirect to main_page
        else:
            flash('Invalid login credentials. Please try again.', 'error')  # Flash error message
            return render_template('login.html')  # Redirect back to login page

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO register (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
        mysql.connection.commit()
        cur.close()

        # Now, let's initialize the session for the newly registered user
        session['logged_in'] = True
        session['username'] = username

        # Redirect to home page after successful registration
        return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return render_template('login.html')

@login_manager.user_loader
def load_user(user_id):
    # Implement this function to load a user from the database based on user_id
    # Example: return User.query.get(int(user_id))
    pass


class User(UserMixin):
    pass



class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    try:
        user_id = int(user_id)  # Ensure user_id is an integer
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM register WHERE id = %s", (user_id,))
        user_data = cur.fetchone()
        cur.close()
        if user_data:
            user = User()
            user.id = user_data[0]  # Assuming id is the first column in your register table
            user.username = user_data[1]  # Assuming username is the second column
            # Add more attributes as needed
            return user
        else:
            print("User not found for id:", user_id)
            return None
    except Exception as e:
        print("Error loading user:", e)
        return None

@app.route('/account')
def account():
    username = session.get('username')  # Retrieve username from session
    return render_template('account.html', username=username) 

# Other routes...


@app.route('/')
def main_page():
    """
    Render the main page.
    """
    global similarities
    global Questions_Arr
    global Correct_Answer_Arr
    global User_Answers
    global All_Text_Details

    similarities = []
    Questions_Arr = []
    Correct_Answer_Arr = []
    User_Answers = []
    All_Text_Details = []

    global emotion_counts
    global All_Video_Details


    All_Video_Details = []
    emotion_counts = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0,
        'no_face': 0,
    } 
    return render_template('Main_page.html')



df = pd.read_csv('data/normalized_dataset.csv')
df2 = pd.read_csv('data/normalized_dataset_ds.csv')
similarities = []
Questions_Arr = []
Correct_Answer_Arr = []
User_Answers = []
All_Video_Details = []
All_Text_Details = []

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ogg', 'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


camera = cv2.VideoCapture(0)
emotion_counts = {
    'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sad': 0,
    'surprise': 0,
    'neutral': 0,
    'no_face': 0,
}
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_text(text):
    tokens = [token.text.lower() for token in nlp(text) if token.text.isalpha()]
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

def sentences_similarity(sentence1, sentence2):
    preprocessed_sentence1 = preprocess_text(sentence1)
    preprocessed_sentence2 = preprocess_text(sentence2)

    corpus = [preprocessed_sentence1, preprocessed_sentence2]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similarity = similarity_matrix[0][1]
    return similarity
        

@app.route('/')
def index():
    """
    Render the main page.
    """
    global similarities
    global Questions_Arr
    global Correct_Answer_Arr
    global User_Answers
    global All_Text_Details

    similarities = []
    Questions_Arr = []
    Correct_Answer_Arr = []
    User_Answers = []
    All_Text_Details = []

    global emotion_counts
    global All_Video_Details


    All_Video_Details = []
    emotion_counts = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0,
        'no_face': 0,
    } 
    return render_template('login.html')

#contact

# Initialize SQLAlchemy


# Create Database Tables




@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    if request.method == 'POST':
        # Extract form data
        name = request.form['name']
        email = request.form['email']
        telephone = request.form['telephone']
        subject = request.form['subject']
        message = request.form['message']
        
        # Store form data in the database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO contact (name, email, telephone, subject, message) VALUES (%s, %s, %s, %s, %s)",
                    (name, email, telephone, subject, message))  # Note: 'contact' table name
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('home'))    

@app.route('/contact', methods=['GET', 'POST'])
def contact():
        if request.method == 'POST':
        # Extract form data
            name = request.form['name']
            email = request.form['email']
            telephone = request.form['telephone']
            subject = request.form['subject']
            message = request.form['message']
       
            # Process the form data here (e.g., send email, save to database)
            return render_template('Main_page.html')  # Redirect to index after form submission
        else:
            return render_template('contact.html')
  
        
        



    


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        message = request.form['f_message']
        # Process the form data here (e.g., send email, save to database)
        return redirect(url_for('home.html'))  # Redirect to index after form submission
    else:
        return render_template('feedback.html')


    ########
@app.route('/home')
def home():
    global similarities
    global Questions_Arr
    global Correct_Answer_Arr
    global User_Answers
    global All_Text_Details

    similarities = []
    Questions_Arr = []
    Correct_Answer_Arr = []
    User_Answers = []
    All_Text_Details = []

    global emotion_counts
    global All_Video_Details


    All_Video_Details = []
    emotion_counts = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0,
        'no_face': 0,
    } 
    return render_template('Main_page.html')

@app.route('/about')
def about():
    """
    Render the about page.
    """
    return render_template('Main_page.html')

@app.route('/text_test_instructions')
def text_test_instructions():
    """
    Render the instructions for the text test.
    """
    return render_template('Instructions_text.html')

@app.route('/video_test_instructions')
def video_test_instructions():
    """
    Render the instructions for the video test.
    """
    return render_template('Instructions_video.html')

@app.route('/audio_test_instructions')
def audio_test_instructions():
    """
    Render the instructions for the text test.
    """
    return render_template('Instructions_audio.html')

@app.route('/new')
def new():
    """
    Render the new page.
    """
    return render_template('Domain.html')

@app.route('/domain_video')
def domain_video():
    """
    Render the new page.
    """
    return render_template('Domain_video.html')

@app.route('/Text_Test')
def Text_Test():
    """
    Render the text test page.
    """
    return render_template('Text_Test.html')

@app.route('/Text_Test2')
def Text_Test2():
    """
    Render the text test page.
    """
    return render_template('Text_Test2.html')


@app.route('/Text_Test_Results')
def Text_Test_Results():
    """
    Render the text test results page.
    """
    return render_template('Text_Test_Results.html')

@app.route('/Video_Test')
def Video_Test():
    """
    Render the video test page.
    """
    return render_template('Video_Test.html')

@app.route('/Video_Test2')
def Video_Test2():
    """
    Render the video test page.
    """
    return render_template('Video_Test2.html')

@app.route('/Video_Test_Results')
def Video_Test_Results():
    """
    Render the video test results page.
    """
    return render_template('Video_Test_Results.html')
# Resume-Builder Part
#@app.route('/resume_0')
#def home2():
    #return render_template("index.html")
@app.route('/new_audio')
def new_audio():
    """
    Render the new page.
    """
    return render_template('Domain_audio.html')
   

@app.route('/audio_ds')
def audio_ds():
    
     # Flash message
    flash("After pressing the button above, you will have 59 sec to answer the question.")
    
    return render_template('audio_ds.html', display_button=False)   
    

@app.route('/resume_1')
def resume_1():
    return render_template("resume_1.html")

@app.route('/resume_2')
def resume_2():
    return render_template("resume_2.html")

@app.route('/resume_template')
def resume_template():
    return render_template("resume_template.html")

    
@app.route('/favicon.ico')
def favicon():
    """
    Serve the favicon.
    """
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/Questions')
def Text_Questions():
    """
    Get random questions from the dataset.
    """
    global Questions_Arr
    global Correct_Answer_Arr
    random_rows = df.sample(n=10)
    Questions_Arr = random_rows['Questions'].tolist()
    Correct_Answer_Arr = random_rows['Answers'].tolist()
    return Questions_Arr

@app.route('/Text_Answers/<int:Qindex>', methods=['POST'])
def text_answers(Qindex):
    """
    Receive and process user answers for text questions.
    """
    global All_Text_Details
    global Questions_Arr
    global Correct_Answer_Arr

    answer = request.data.decode('utf-8')
    temp_list = []
    temp_list.append(Questions_Arr[Qindex])
    temp_list.append(Correct_Answer_Arr[Qindex])
    temp_list.append(answer)
    temp_list.append(sentences_similarity(str(answer),str(Correct_Answer_Arr[Qindex])))
    All_Text_Details.append(temp_list)
    return jsonify(answer)

@app.route('/Questions2')
def Text_Questions2():
    """
    Get random questions from the dataset.
    """
    global Questions_Arr
    global Correct_Answer_Arr
    random_rows = df2.sample(n=10)
    Questions_Arr = random_rows['Questions'].tolist()
    Correct_Answer_Arr = random_rows['Answers'].tolist()
    return Questions_Arr

@app.route('/Text_Answers/<int:Qindex>', methods=['POST'])
def text_answers2(Qindex):
    """
    Receive and process user answers for text questions.
    """
    global All_Text_Details
    global Questions_Arr
    global Correct_Answer_Arr

    answer = request.data.decode('utf-8')
    temp_list = []
    temp_list.append(Questions_Arr[Qindex])
    temp_list.append(Correct_Answer_Arr[Qindex])
    temp_list.append(answer)
    temp_list.append(sentences_similarity(str(answer),str(Correct_Answer_Arr[Qindex])))
    All_Text_Details.append(temp_list)
    return jsonify(answer)

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

# Audio Index
@app.route('/audio_index', methods=("POST", "GET"))
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 59 sec to answer the question.")
    
    return render_template('audio.html', display_button=False)


# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 59 # in sec
    rec_sub_dir = os.path.join('uploads','voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True)


# Audio Emotion Analysis
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():
    
    emotions = ["Angry", "Disgust", "Fear",  "Happy", "Sad", "Surprise", "Neutral"]
    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models copy', 'MODEL_CNN_LSTM.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join('uploads','voice_recording.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/script/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/script/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/script/db','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static/script/db", "audio_emotions_other.txt"), sep=",")

    # Get most common emotion during the interview for other candidates
    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static/script/db','audio_emotions_dist_other.txt'), sep=',')

    # Sleep
    time.sleep(0.5)

    return render_template('audio_dash.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotion_dist, prob_other=emotion_dist_other)



@app.route('/video_results')
def video_results():
    temp_Emotion_Counts = emotion_counts
    temp_Results = All_Video_Details

    temp_Emotion_Counts['angry'] = emotion_counts['angry'] * 0.2
    temp_Emotion_Counts['disgust'] = emotion_counts['disgust'] * 0.2
    temp_Emotion_Counts['fear'] = emotion_counts['fear'] * 0.2
    temp_Emotion_Counts['happy'] = emotion_counts['happy'] * 1.3
    temp_Emotion_Counts['sad'] = emotion_counts['sad'] * 0.2
    temp_Emotion_Counts['neutral'] = emotion_counts['neutral'] * 1
    temp_Emotion_Counts['no_face'] = 0

    new_temp = [] 
    new_temp.append(temp_Results)
    new_temp.append(list(temp_Emotion_Counts.values()))
    return jsonify(new_temp)


@app.route('/text_results')
def text_results():
    # All_Text_Details_temp = []
    All_Text_Details_temp = All_Text_Details    
    return jsonify(All_Text_Details_temp)

def generate_frames():
    global camera
    while True:
        if camera is not None:
            success, frame = camera.read()
            if not success:
                break
        else:
            break
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_copy = frame
        frame = buffer.tobytes()
 
        # convert to grayscale
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

        # detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # loop over faces
        for (x, y, w, h) in faces:
            # extract face
            face = frame_copy[y:y+h, x:x+w]
            # recognize emotion if a face is detected
            if len(face) > 0:
                try:
                    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=True)
                    if result[0]['dominant_emotion'] is not None:
                        emotion_counts[result[0]['dominant_emotion']] += 1
                    else:
                        emotion_counts['no_emotion'] += 1
                except ValueError as err:
                    emotion_counts['no_face'] += 1
            # update the no_face count if no face is detected
            else:
                emotion_counts['no_face'] += 1

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/videofeed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return 'Camera started'

@app.route('/start_again')
def start_again():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    while not camera.isOpened():  # Wait until the camera is ready
        pass
    return 'Camera started'

@app.route('/stop')
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        camera = cv2.VideoCapture(0)
    return 'Camera stopped'

if __name__=='__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')


