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
    print("helllo???", flush=True)
    return Prompt.give_advice(emotions_list, intensity, stability, wavelength, burstiness)

def calcualtions(Emotions):
    features=[]
    features.append(Calculations.calculate_intensity(Emotions))
    features.append(Calculations.calculate_stability(Emotions))
    features.append(Calculations.calculate_wavelength(Emotions))
    features.append(Calculations.calculate_burstiness(Emotions))
    print("calcul", flush=True)
    return features
    
def extract_emotions(faces):
    emotions=[]
    print("emotion", flush=True)
    for face in faces:
        emotions.append(AI_emotions.tell_emotion(face))
    print("after emotions", flush=True)
    return np.array(emotions)

def extract_faces(pixels):
    cascade_file = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_file):
        # اصلاح مهم: به جای شیء وب (jsonify)، اینجا None برمی‌گردانیم تا حلقه بعدی کرش نکند
        return None 
    print("ligger1", flush=True)    
    face_cascade = cv2.CascadeClassifier(cascade_file)
    cropped_faces = []
    print("nai nia", flush=True)
    for frame in pixels:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
        print("ligger3", flush=True)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            cropped_faces.append(face_resized)
        else:
            cropped_faces.append(cv2.resize(frame, (224, 224)))
    print("ligger4", flush=True)
    return np.array(cropped_faces)

def pixelization():
    video_path = os.path.join('uploads', 'user_video.webm')
    cap = cv2.VideoCapture(video_path)
    fps = 30
    pixels = []
    frame_count = 0
    print("zigga1", flush=True)
    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
        print("zigga2", flush=True) 
        if frame_count % fps == 0:
            ret, frame = cap.retrieve()
            if ret:
                small_frame = cv2.resize(frame, (320, 240))
                pixels.append(small_frame)
                del frame     
        frame_count += 1
    print("zigga3", flush=True)
    cap.release()
    return np.array(pixels)

def SaveVideo(video):
    os.makedirs('uploads', exist_ok=True)
    save_path = os.path.join('uploads', 'user_video.webm')
    print("saveali", flush=True)
    video.save(save_path)

@app.route("/", methods=["GET", "POST"])   
def index():
    if request.method == "POST":
        print("lonly lonly lonnly only only only", flush=True)
        video = request.files.get('video')
        print("video getar", flush=True)
        if not video:
            return jsonify({'status': 'error', 'message': 'there is no file!'}), 400
        
        # اضافه شدن بلاک try-except برای گرفتن ارورها به صورت JSON به جای کرش کردن کل سرور
        try:
            SaveVideo(video)
            
            pixs = pixelization()
            if len(pixs) == 0:
                return jsonify({'status': 'error', 'message': 'Video processing failed: No frames extracted. Check Linux codecs.'}), 500
                
            pixs = extract_faces(pixs)
            if pixs is None:
                return jsonify({'status': 'error', 'message': 'haarcascade file is missing on server!'}), 500
                
            emotions = extract_emotions(pixs)
            features = calcualtions(emotions)

            intensity = float(features[0])
            stability = float(features[1])
            wavelength = float(features[2])
            burstiness = float(features[3])
            
            responds = Responds(emotions, intensity, stability, wavelength, burstiness)
            explain = responds[0]
            advice = responds[1]
            print("aaayyyyoooo", flush=True)
            if isinstance(emotions, np.ndarray):
                emotions_list = emotions.tolist()
            else:
                emotions_list = list(emotions)
                
            emotions_str = ", ".join(emotions_list)

            # بررسی وجود فریم قبل از ذخیره عکس برای جلوگیری از خطای IndexError
            if len(pixs) > 0:
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
            
        except Exception as e:
            print(f"Backend Error: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Python Exception: {str(e)}'}), 500
    
    return render_template("templates/index.html")

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
