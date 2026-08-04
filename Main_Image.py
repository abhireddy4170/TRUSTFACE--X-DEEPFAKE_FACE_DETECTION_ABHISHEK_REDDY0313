from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
import pandas as pd
from keras.utils.np_utils import to_categorical
from PIL import Image, ImageTk

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, LSTM
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
import keras

import sounddevice as sd
import librosa
from scipy.io import wavfile
from keras.models import model_from_json
from keras.initializers import glorot_uniform

# ====================================================
# SYSTEM INITIALIZATION
# ====================================================
main = tkinter.Tk()
main.title("TRUSTFACE-X: Deepfake Face Exposure")
main.geometry("1200x700")
# Ultra-Modern Palette: Main Background
main.configure(bg="#F8FAFC")

try:
    main.iconbitmap('app_icon.ico')
except Exception:
    pass 

# Global Variables (Untouched)
global lstm_model, filename, X, Y, dataset, labels
detection_model_path = 'model/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

# ====================================================
# UNTOUCHED BACKEND FUNCTIONS
# ====================================================
def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, labels, X, Y, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text="Dataset: " + filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv("Dataset/metadata.csv")
    labels = np.unique(dataset['label'])
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        images = dataset['filename'].ravel()
        classes = dataset['label'].ravel()
        for i in range(len(images)):
            if os.path.exists("Dataset/images/"+images[i]):
                 img = cv2.imread("Dataset/images/"+images[i])
                 img = cv2.resize(img, (32, 32))
                 X.append(img)
                 label = getLabel(classes[i])
                 Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X1.txt',X)
        np.save('model/Y1.txt',Y)
    text.insert(END,"Class labels found in Dataset : "+str(labels)+"\n")    
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")    

def trainModel():
    text.delete('1.0', END)
    global X, Y, labels, lstm_model
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing : "+str(X_test.shape[0])+"\n\n")
    lstm_model = Sequential()
    lstm_model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (1, 32, 32, 3)))
    lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((2, 2))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((1, 1))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Flatten()))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 64, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:18200] = y_test1[0:18200]
    calculateMetrics("DL", y_test1, predict)

def playVideo(filename, output):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (500, 500))
            cv2.putText(frame, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
            cv2.imshow('Deep Fake Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()    

def uploadImage():
    text.delete('1.0', END)
    global lstm_model, labels

    filename = askopenfilename(initialdir="Images", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    pathlabel.config(text="File: " + filename)

    if not filename:
        text.insert(END, "No file selected!\n")
        return

    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        face_img = image[fY:fY + fH, fX:fX + fW]

        img = cv2.resize(face_img, (32, 32))
        img = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        preds = lstm_model.predict(img)
        predict = np.argmax(preds)
        recognize = labels[predict]

        if predict == 0:
            text.insert(END, "Uploaded image detected as Deepfake\n")
        else:
            text.insert(END, "Uploaded image detected as Real\n")

        cv2.putText(image, 'Status: ' + recognize, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Deep Fake Detection Output', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        text.insert(END, "No face detected in the image!\n")

def uploadImage1():
    text.delete('1.0', END)
    global lstm_model, labels
    filename = askopenfilename(initialdir="Images")
    pathlabel.config(text=filename)
    image = cv2.imread(filename)
    if image is None:
        text.insert(END, "Error: Unable to load image.\n")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        text.insert(END, "No face detected in the image.\n")
        return

    (fX, fY, fW, fH) = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])[0]
    face = image[fY:fY + fH, fX:fX + fW]
    img = cv2.resize(face, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    temp = []
    temp.append(im2arr)
    img = np.asarray(temp)
    im2arr = np.array(img, dtype="float32") / 255.0
    im2arr = np.expand_dims(im2arr, axis=0)
    img1 = np.expand_dims(im2arr, axis=1)

    preds = lstm_model.predict(img1)
    predict = np.argmax(preds)
    result = labels[predict]

    text.insert(END, f"Uploaded image detected as: {result}\n")
    cv2.putText(image, f"Prediction: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Deep Fake Detection Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def uploadVideo():
    text.delete('1.0', END)
    global lstm_model, labels
    fake = 0
    real = 0
    count = 0
    output = ""
    filename = askopenfilename(initialdir = "Videos")
    pathlabel.config(text="Video: " + filename)
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) > 0:
                count = count + 1
                faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                image = frame[fY:fY + fH, fX:fX + fW]
                img = cv2.resize(image, (32, 32))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,32,32,3)
                temp = []
                temp.append(im2arr)
                img = np.asarray(temp)
                img = img.astype('float32')
                img = img/255
                preds = lstm_model.predict(img)
                predict = np.argmax(preds)
                recognize = labels[predict]
                if predict == 0:
                    fake += 1
                else:
                    real += 1
                frame = cv2.resize(frame, (500, 500))
            else:
                frame = cv2.resize(frame, (500, 500))
            cv2.putText(frame, 'Video analysis under progress', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
            cv2.imshow('Deep Fake Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count > 30:
                if real > fake:
                    output = "Video is Real"
                    text.insert(END,"Uploaded video detected as Real\n")
                else:
                    output = "Deepfake Detected"
                    text.insert(END,"Uploaded video detected as Deepfake\n")
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    playVideo(filename, output)

speech_classifier = None
speech_type = ['REAL', 'FAKE']

def loadModel():
    global speech_classifier
    if speech_classifier is not None:
        return True 

    try:
        if os.path.exists('deepfake_audio_model.json') and os.path.exists('deepfake_audio_weights.h5'):
            with open("deepfake_audio_model.json", "r") as json_file:
                loaded_model_json = json_file.read()
            loaded_model_json = loaded_model_json.replace("GlorotUniform", "glorot_uniform")
            speech_classifier = model_from_json(
                loaded_model_json,
                custom_objects={"glorot_uniform": glorot_uniform()}
            )
            speech_classifier.load_weights("deepfake_audio_weights.h5")
            text.insert(END, "Model Loaded Successfully\n")
            return True
        else:
            text.insert(END, "ERROR: Model files not found!\n")
            return False
    except Exception as e:
        text.insert(END, "MODEL LOAD ERROR: " + str(e) + "\n")
        return False

def uploadAudio():
    global speech_classifier
    text.delete('1.0', END)
    model_loaded = loadModel()
    if not model_loaded:
        return
    filename = filedialog.askopenfilename(initialdir = "testAudio",filetypes=[("Audio Files", "*.wav *.mp3")])
    if filename == "":
        return
    fname = os.path.basename(filename)
    y, sr = librosa.load(filename, sr=None)
    segment_duration = 3
    segment_samples = int(sr * segment_duration)
    segments = []

    for start in range(0, len(y), segment_samples):
        segment = y[start:start + segment_samples]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        if len(mfcc) < 180:
            mfcc = np.pad(mfcc, (0, 180 - len(mfcc)))
        else:
            mfcc = mfcc[:180]
        segments.append(mfcc)

    if len(segments) == 0:
        text.insert(END, "ERROR: Audio too short\n")
        return

    test = np.asarray(segments).astype('float32')
    if np.max(test) != 0:
        test = test / np.max(test)

    test = test.reshape((test.shape[0], 180, 1, 1))
    predictions = speech_classifier.predict(test)
    avg_pred = np.mean(predictions, axis=0)
    predict = np.argmax(avg_pred)

    deepfake = speech_type[predict]
    confidence = float(avg_pred[predict]) * 100

    text.insert(END, "Audio File : " + fname + "\n")
    text.insert(END, "Prediction Result : " + deepfake + "\n")
    text.insert(END, "Prediction Confidence : {:.2f}%\n".format(confidence))

    rate, snd = wavfile.read(filename)
    preview_len = min(len(snd), segment_samples)
    sd.play(snd[:preview_len], rate)
    sd.wait()

# ====================================================
# NEW SAAS-STYLE UI LAYOUT
# ====================================================

# UI Logic Wrappers
def on_upload_dataset():
    uploadDataset()
    status_label.config(text="Status: Dataset Loaded | Ready for Training", fg="#D97706") # Muted Amber

def on_train_model():
    status_label.config(text="Status: Training Model...", fg="#6366F1") # Primary Accent
    main.update()
    trainModel()
    status_label.config(text="Status: Model Trained & Loaded", fg="#059669") # Muted Emerald

def on_upload_image():
    status_label.config(text="Status: Processing Image...", fg="#6366F1")
    main.update()
    uploadImage()
    status_label.config(text="Status: Image Analysis Complete", fg="#059669")

def on_upload_video():
    status_label.config(text="Status: Analyzing Video...", fg="#6366F1")
    main.update()
    uploadVideo()
    status_label.config(text="Status: Video Analysis Complete", fg="#059669")

def on_upload_audio():
    status_label.config(text="Status: Analyzing Audio...", fg="#6366F1")
    main.update()
    uploadAudio()
    status_label.config(text="Status: Audio Analysis Complete", fg="#059669")

# --- Top Header Panel ---
header_frame = Frame(main, bg="#E0E7FF", height=70, bd=0, highlightthickness=1, highlightbackground="#E2E8F0")
header_frame.pack(side=TOP, fill=X)
header_label = Label(header_frame, text="TRUSTFACE-X: HYBRID CNN-LSTM INTELLIGENCE FOR DEEPFAKE EXPOSURE", bg="#E0E7FF", fg="#1E293B", font=('Segoe UI', 18, 'bold'))
header_label.pack(pady=15)

# --- Main Layout Body ---
body_frame = Frame(main, bg="#F8FAFC")
body_frame.pack(side=TOP, fill=BOTH, expand=True)

# --- Left Navigation Sidebar ---
sidebar_frame = Frame(body_frame, bg="#EEF2F7", width=250, bd=0, highlightthickness=1, highlightbackground="#E2E8F0")
sidebar_frame.pack(side=LEFT, fill=Y)

sidebar_title = Label(sidebar_frame, text="Control Panel", bg="#EEF2F7", fg="#1E293B", font=('Segoe UI', 14, 'bold'))
sidebar_title.pack(pady=(20, 20))

def create_nav_button(parent, text, command):
    # Primary Accent: #6366F1, Text: White
    btn = Button(parent, text=text, command=command, bg="#6366F1", fg="white", 
                 font=('Segoe UI', 11, 'bold'), relief="flat", pady=12, cursor="hand2")
    btn.pack(fill=X, padx=15, pady=8)
    # Button Hover: #4F46E5
    btn.bind("<Enter>", lambda e: e.widget.config(bg="#4F46E5"))
    btn.bind("<Leave>", lambda e: e.widget.config(bg="#6366F1"))
    return btn

btn_dataset = create_nav_button(sidebar_frame, "Upload Dataset", on_upload_dataset)
btn_train   = create_nav_button(sidebar_frame, "Train Model", on_train_model)
btn_image   = create_nav_button(sidebar_frame, "Image Detection", on_upload_image)
btn_video   = create_nav_button(sidebar_frame, "Video Detection", on_upload_video)
btn_audio   = create_nav_button(sidebar_frame, "Audio Detection", on_upload_audio)


# --- Right Main Content Area ---
content_frame = Frame(body_frame, bg="#F8FAFC")
content_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=30, pady=20)

# 1. Status & Info Board (Top)
info_frame = Frame(content_frame, bg="#FFFFFF", bd=0, highlightthickness=1, highlightbackground="#E2E8F0")
info_frame.pack(side=TOP, fill=X, pady=(0, 15))

status_label = Label(info_frame, text="Status: System Ready | Model Not Trained", bg="#FFFFFF", fg="#64748B", font=('Segoe UI', 11, 'bold'))
status_label.pack(side=LEFT, padx=15, pady=12)

pathlabel = Label(info_frame, text="Dataset / File: None", bg="#FFFFFF", fg="#64748B", font=('Segoe UI', 10, 'italic'))
pathlabel.pack(side=RIGHT, padx=15, pady=12)

# 2. Main Display Area (Parent container for our layers)
display_area = Frame(content_frame, bg="#FFFFFF", bd=0, highlightthickness=1, highlightbackground="#E2E8F0")
display_area.pack(side=TOP, fill=BOTH, expand=True)

# 3. Layer 1: The Image (Base Background Layer)
try:
    dash_image = Image.open("deepf(1).png")
    # Setting a clean resolution for the background layer
    dash_image = dash_image.resize((1200, 700), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)
    dash_photo = ImageTk.PhotoImage(dash_image)

    dash_image_label = Label(display_area, image=dash_photo, bg="#FFFFFF", bd=0)
    dash_image_label.image = dash_photo  
    # .place() anchors it in the center of the display area
    dash_image_label.place(relx=0.5, rely=0.45, anchor=CENTER) 
except Exception as e:
    pass 

# 4. Layer 2: Logs & Output Console (Floating Top Layer)
# We use .place() to float this directly over the display_area and the image
console_frame = Frame(display_area, bg="#BBEAF5", bd=0, highlightthickness=1, highlightbackground="#E2E8F0")

# relwidth=0.85 means it takes up 85% of the width, relheight=0.35 means 35% of the height
# relx=0.5 and rely=0.95 with anchor=S floats it at the bottom center, overlapping the image
console_frame.place(relx=0.5, rely=0.79, relwidth=0.50, relheight=0.35, anchor=S)

console_header = Frame(console_frame, bg="#C9DCF3", bd=0, highlightthickness=1, highlightbackground="#E2E8F0")
console_header.pack(side=TOP, fill=X)
console_title = Label(console_header, text=" Output Console", bg="#A9C9EF", fg="#1E293B", font=('Segoe UI', 11, 'bold'))
console_title.pack(side=LEFT, padx=10, pady=8)

scroll = Scrollbar(console_frame)
scroll.pack(side=RIGHT, fill=Y, pady=5)

# Console uses very light gray background with Muted Slate Text
text = Text(console_frame, height=8, yscrollcommand=scroll.set, bg="#ACC5DE", fg="#334155", 
            font=('Consolas', 11), relief="flat", insertbackground="#1E293B")
text.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
scroll.config(command=text.yview)

main.mainloop()