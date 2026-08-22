from flask import Flask, request, render_template, flash
import numpy as np
import os
from flask import redirect, url_for, jsonify
import random, cv2
import AI_emotions, Calculations, Prompt
cv2.setNumThreads(1)
app = Flask(__name__, template_folder='.')
app.secret_key = 'javidshahjavidshahjavidshah'

    
def Responds(emotions_list, intensity, stability, wavelength, burstiness):
    return Prompt.give_advice(emotions_list, intensity, stability, wavelength, burstiness)

def calcualtions(Emotions):
    features=[]
    features.append(Calculations.calculate_intensity(Emotions))
    features.append(Calculations.calculate_stability(Emotions))
    features.append(Calculations.calculate_wavelength(Emotions))
    features.append(Calculations.calculate_burstiness(Emotions))
    return features
    
def extract_emotions(faces):
    emotions=[]
    for face in faces:
        #print("*")
        emotions.append(AI_emotions.tell_emotion(face))
    return np.array(emotions)


def extract_faces(pixels):
    cascade_file = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_file):
        return jsonify({'status': 'error', 'message': 'there is not any files!'}), 500
        
    face_cascade = cv2.CascadeClassifier(cascade_file)
    cropped_faces = []

    for frame in pixels:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            #print("+")
            cropped_faces.append(face_resized)
        else:
            #print("+")
            cropped_faces.append(cv2.resize(frame, (224, 224)))

    return np.array(cropped_faces)


def pixelization():
    video_path = os.path.join('uploads', 'user_video.webm')
    cap = cv2.VideoCapture(video_path)
    fps = 30
    pixels = []
    frame_count = 0

    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
            
        if frame_count % fps == 0:
            ret, frame = cap.retrieve()
            if ret:
                small_frame = cv2.resize(frame, (320, 240))
                pixels.append(small_frame)
                del frame     
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
        
        video = request.files.get('video')
        if not video:
            return jsonify({'status': 'error', 'message': 'there is no file!'}), 400
        
        SaveVideo(video)
        pixs = pixelization()
        pixs = extract_faces(pixs)
        emotions = extract_emotions(pixs)
        features = calcualtions(emotions)

        intensity = float(features[0])
        stability = float(features[1])
        wavelength = float(features[2])
        burstiness = float(features[3])
        
        responds = Responds(emotions, intensity, stability, wavelength, burstiness)
        explain = responds[0]
        advice = responds[1]

        if isinstance(emotions, np.ndarray):
            emotions_list = emotions.tolist()
        else:
            emotions_list = list(emotions)
            
        emotions_str = ", ".join(emotions_list)

        cv2.imwrite('test_face.jpg', pixs[0])
        
        return jsonify({
            'status': 'success',
            'explain': str(explain),
            'advice': str(advice),
            'emotions_str': emotions_str,
            'intensity': intensity,
            'stability': stability,
            'wavelength': wavelength,
            'burstiness': burstiness
        })
    
    return render_template("templates/index.html")
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
