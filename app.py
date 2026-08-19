from flask import Flask, request, render_template, flash
import numpy as np
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import time, re, threading, sqlite3 , random , math , os
from flask import redirect, url_for, jsonify
import requests, json 
import random, cv2

#print(cv2.__version__)
#print(hasattr(cv2, 'CascadeClassifier'))
#########################
import kagglehub

# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)


####################
app = Flask(__name__, template_folder='.')
app.secret_key = 'javidshahjavidshahjavidshah'

def extract_faces(pixels):
    cascade_file = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_file):
        return jsonify({'status': 'error', 'message': 'فایل مدل تشخیص چهره در پوشه پروژه یافت نشد!'}), 500
        
    face_cascade = cv2.CascadeClassifier(cascade_file)
    cropped_faces = []

    for frame in pixels:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            cropped_faces.append(face_resized)
        else:
            cropped_faces.append(cv2.resize(frame, (224, 224)))

    return np.array(cropped_faces)


def pixelization():
    video_path = os.path.join('uploads', 'user_video.webm')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    pixels = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % fps == 0:
            pixels.append(frame)
            
        frame_count += 1

    cap.release()
    return np.array(pixels)
def SaveVideo(video):
    os.makedirs('uploads', exist_ok=True)
    save_path = os.path.join('uploads', 'user_video.webm')
    video.save(save_path)
@app.route("/", methods=["GET", "POST"])   
def index():
    if request.method == "POST":
        print("shaloom")
        video = request.files.get('video')

        if not video:
            return jsonify({'status': 'error', 'message': 'ridim haji!'}), 400

        SaveVideo(video)
        pixs=pixelization()
        pixs=extract_faces(pixs)
        print(pixs.shape)
        cv2.imwrite('test_face.jpg', pixs[0])
        return jsonify({
            'status': 'success',
            'ai_engine': {
                'analysis': 'Calm & High Focus Detected',
                'advice': 'Signal processing completed. Facial frame matrix successfully generated.'
            }
        })
        #return redirect(url_for('index'))
    return render_template("templates/index.html")


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)