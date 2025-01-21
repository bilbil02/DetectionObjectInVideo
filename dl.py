import streamlit as st
import os
import cv2
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# URL file yolov3.weights yang diunggah ke penyimpanan cloud
url = 'https://drive.google.com/file/d/1Ifv2TgD0KpVYG7cKHB0GVPcJF4XhCWgR/view?usp=sharing'  # Ganti dengan URL file Anda

# Cek dan unduh file yolov3.weights
if not os.path.exists('yolov3.weights'):
    st.write("Mengunduh file yolov3.weights...")
    r = requests.get(url)
    with open('yolov3.weights', 'wb') as f:
        f.write(r.content)
    
# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]
    return net, output_layers

# Load class names
def load_class_names():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Extract features using YOLO and prepare sequences for LSTM
def extract_features_for_lstm(video_path, net, output_layers, class_names, time_steps):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    sequences = []
    labels = []
    sequence_buffer = []

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Extract features (confidence scores)
        features = np.zeros(len(class_names))
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    features[class_id] += confidence

        sequence_buffer.append(features)
        if len(sequence_buffer) == time_steps:
            sequences.append(sequence_buffer.copy())
            labels.append(np.argmax(features))  # Example: Assign label based on dominant feature
            sequence_buffer.pop(0)

    cap.release()
    return np.array(sequences), np.array(labels)

# Build LSTM model
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train LSTM model
def train_lstm_model(sequences, labels, class_names):
    labels = to_categorical(labels, num_classes=len(class_names))
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape, len(class_names))

    lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    return lstm_model

# Process video and save output
def process_and_save_video(video_path, output_path, net, output_layers, class_names, lstm_model, time_steps):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

    sequence_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        features = np.zeros(len(class_names))
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    features[class_id] += confidence
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_names[class_id]}: {int(confidence * 100)}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        sequence_buffer.append(features)
        if len(sequence_buffer) == time_steps:
            prediction = lstm_model.predict(np.expand_dims(sequence_buffer, axis=0))
            predicted_class = np.argmax(prediction)
            cv2.putText(frame, f"Sequence Prediction: {class_names[predicted_class]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            sequence_buffer.pop(0)

        out.write(frame)

    cap.release()
    out.release()

# Streamlit UI
st.markdown(
            "<h1 style='text-align: center;'>Object Detection In Video using YOLO and LSTM</h1>",
            unsafe_allow_html=True
        )


uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi"])
time_steps = 10

if st.button("Process Video"):
    if uploaded_video:
        # Save uploaded video
        video_path = "uploaded_video.mp4"
        output_path = "output_video.avi"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Load YOLO model and class names
        net, output_layers = load_yolo_model()
        class_names = load_class_names()

        # Extract features and train LSTM
        st.write("Extracting features and preparing data...")
        sequences, labels = extract_features_for_lstm(video_path, net, output_layers, class_names, time_steps)

        st.write("Training LSTM model...")
        lstm_model = train_lstm_model(sequences, labels, class_names)

        st.write("Processing video for object detection and LSTM predictions...")
        process_and_save_video(video_path, output_path, net, output_layers, class_names, lstm_model, time_steps)

        st.success("Processing complete! Download the processed video below.")
        with open(output_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.avi")

    else:
        st.error("Please upload a video file.")
