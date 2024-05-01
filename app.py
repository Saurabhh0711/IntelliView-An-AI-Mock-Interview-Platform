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
import cv2
import subprocess
from deepface import DeepFace
from werkzeug.utils import secure_filename
import pandas as pd


app = Flask(__name__)
df = pd.read_csv('data/normalized_dataset.csv')
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


if __name__=='__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')