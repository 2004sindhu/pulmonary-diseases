import os
import time
import shutil
import re
import numpy as np
import pandas as pd
from io import BytesIO
import librosa.display
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io.wavfile import write
from datetime import datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Function to get list of audio files from a folder
def get_audio_files(folder_path):
    audio_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp3', '.wav')):
            audio_files.append(os.path.join(folder_path, filename))
    return audio_files

def save_uploaded_file(uploaded_file, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'audio.wav'), "wb") as f:
        f.write(uploaded_file.getbuffer())

def Denoise(raw_audio, sample_rate=4000, filter_order=5, filter_lowcut=50, filter_highcut=1800, btype="bandpass"):
    b, a = 0,0
    if btype == "bandpass":
        b, a = signal.butter(filter_order, [filter_lowcut/(sample_rate/2), filter_highcut/(sample_rate/2)], btype=btype)

    if btype == "highpass":
        b, a = signal.butter(filter_order, filter_lowcut, btype=btype, fs=sample_rate)

    audio = signal.lfilter(b, a, raw_audio)

    return audio

def build_mfcc(file_path):
    plt.interactive(False)
    file_audio_series,sr = librosa.load(file_path,sr=None)    
    spec_image = plt.figure(figsize=[1,1])
    ax = spec_image.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectogram = librosa.feature.melspectrogram(y=file_audio_series, sr=sr)
    logmel=librosa.power_to_db(spectogram, ref=np.max)
    librosa.display.specshow(librosa.feature.mfcc(S=logmel, n_mfcc=30))
    
    image_name  = 'image/mfccfile.png'
    plt.savefig(image_name, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    spec_image.clf()
    plt.close(spec_image)
    plt.close('all')

def model_predict():
    model = load_model('mfcc30.h5')
    labels = np.load('labels.npy')
    # Load the VGG16 model without the top classification layer
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Load and preprocess the test image
    test_image_path = 'image/mfccfile.png'
    img = load_img(test_image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Extract features from the test image using VGG16
    features = vgg16.predict(img_array)
    # Make prediction using the trained LSTM model
    prediction = model.predict(features.reshape((features.shape[0], -1, features.shape[-1])))
    # Convert prediction probabilities to class labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

    return [predicted_class[0]]

from datetime import date

# Function to generate a PDF with additional details
st.markdown(
    """
    <style>
        .prescription-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ddd;
            font-family: Arial, sans-serif;
            margin-bottom: 20px;
        }
        .prescription-title {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            color: #333;
            margin-bottom: 15px;
        }
        .prescription-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .prescription-table th, .prescription-table td {
            border: 1px solid #ccc;
            padding: 8px;
            text-align: left;
        }
        .prescription-table th {
            background-color: #f2f2f2;
        }
        .download-btn {
            margin-top: 20px;
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to generate a PDF with styled details
def generate_pdf(name, age, gender, disease, hospital_name, doctor_name, specialization, diagnosis_date):
    buffer = BytesIO()  # Create a memory buffer for the PDF
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header with hospital name
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 40, "{hospital_name}")

    # Doctor details section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 80, f"Doctor Name: {doctor_name}")
    c.drawString(400, height - 80, f"Specialization: {specialization}")

    # Patient Details section
    y_position = height - 140
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y_position, "Patient Details:")
    c.setFont("Helvetica", 10)
    details = [
        f"Name: {name}",
        f"Age: {age}",
        f"Gender: {gender}",
        f"Disease: {disease}",
        f"Date of Diagnosis: {diagnosis_date}",
    ]
    for detail in details:
        y_position -= 20
        c.drawString(30, y_position, detail)

    # Add a footer (optional)
    y_position -= 40
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(30, y_position, "Generated by Pulmonary Disease Detection System")

    
    c.showPage()
    c.save()

    buffer.seek(0)  # Go to the beginning of the buffer
    return buffer

# Update the download_pdf function to handle the PDF as a binary stream
def download_pdf(name, age, gender, disease, hospital_name, doctor_name, specialization, diagnosis_date):
    pdf_buffer = generate_pdf(name, age, gender, disease, hospital_name, doctor_name, specialization, diagnosis_date)
    pdf_filename = f"{name}_patient_info.pdf"

   

    # Use Streamlit's download button with the PDF buffer
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Updated Streamlit app
st.title("Pulmonary Disease Detection")

# Folder path where audio files are stored
folder_path = 'audio'  # Replace 'path_to_your_folder' with the actual folder path

# Get the list of audio files from the folder
audio_files = get_audio_files(folder_path)
hospital_name = st.text_input("Enter Hospital Name")
doctor_name = st.text_input("Enter Doctor Name")
specialization = st.text_input("Enter Doctor Specialization")
diagnosis_date = date.today().strftime("%Y-%m-%d")

# Existing fields for patient details
name = st.text_input("Enter Your Name")
age = st.text_input("Enter Your Age")
gender = st.selectbox("Select Your Gender", ["---", "Male", "Female", "Other"])
mobile_number = st.text_input("Enter Your Mobile Number")

option = st.radio("Select an option", ("Upload Manually", "Browse List"))
c = 0
# If user chooses to upload manually
if option == "Upload Manually":
    upfile = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if upfile:
        save_uploaded_file(upfile, "uploaded_audio")
        uploaded_file = 'uploaded_audio/audio.wav'
        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            st.success('Audio File Uploaded')
            c = 1
elif option == "Browse List":
    select_file = st.selectbox("Select an audio file", audio_files)
    if select_file:
        shutil.copy(select_file, "uploaded_audio/audio.wav")
        uploaded_file = 'uploaded_audio/audio.wav'
        st.audio(uploaded_file, format='audio/*')
        st.success('Audio File Uploaded')
        c = 1

if c == 1:
    if st.button("Generate"):
        raw_audio, sample_rate = librosa.load(uploaded_file, sr=4000)
        adfile = Denoise(raw_audio)
        path = 'denaudio'
        save_path = os.path.join(path, 'denoise.wav')
        write(save_path, sample_rate, adfile)
        file = 'denaudio/denoise.wav'
        build_mfcc(file)
        model_pred = model_predict()
        disease = model_pred[0]
        patient_info = {
            "name": name,
            "age": age,
            "gender": gender,
            "mobile": mobile_number,
            "hospital_name": hospital_name,
            "doctor_name": doctor_name,
            "specialization": specialization,
            "diagnosis_date": diagnosis_date,
        }
        details = ["Name", "Age", "Gender", "Disease", "Hospital Name", "Doctor Name", "Specialization", "Date of Diagnosis"]
        vals = [name, age, gender, disease, hospital_name, doctor_name, specialization, diagnosis_date]
        d = {"Details": details, "Values": vals}
        df = pd.DataFrame(d)
        st.write(df.to_html(index=False, escape=False), unsafe_allow_html=True)
        
        # Provide the PDF for download
        st.write("Download the PDF with your details below:")
        download_pdf(name, age, gender, disease, hospital_name, doctor_name, specialization, diagnosis_date)
else:
    st.success("Enter correct details")

