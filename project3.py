import cv2
import torch
import speech_recognition as sr
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

# Load the model and tokenizer
model_path = 'bert-base-uncased'  # You can replace this with your local model path
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Dummy data and dataset for fraud detection (you should replace this with actual data)
class FraudDetectionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        return encoding, label

# Function to use CMU Sphinx (offline) for speech-to-text conversion
def sphinx_speech_to_text(recognizer, mic):
    print("Listening for your speech...")

    # Capture the speech from the microphone
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=1000)  # Add timeout for better efficiency

    try:
        # Use CMU Sphinx for offline conversion
        print("Converting speech to text...")
        text = recognizer.recognize_sphinx(audio)
        print(f"Converted Text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sphinx could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Sphinx error: {e}")
        return None

# Function to process the text with the BERT model
def process_text_with_model(text):
    # Create dataset and dataloader
    texts = [text]
    labels = [1 if 'fraudulent' in text else 0 for text in texts]  # Simplified label logic

    dataset = FraudDetectionDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            input_ids = inputs['input_ids'].squeeze(1)
            attention_mask = inputs['attention_mask'].squeeze(1)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(outputs.logits)  # Outputs the raw logits for fraud detection

# Capture audio and process it for fraud detection in a separate thread
def capture_and_process_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        user_input = input("Press Enter to start speech recognition or 'q' to quit: ")
        
        if user_input.lower() == 'q':
            break

        text = sphinx_speech_to_text(recognizer, mic)  # Using CMU Sphinx for speech recognition
        
        if text:
            process_text_with_model(text)
        else:
            print("No text to process.")

# Function to capture video frames
def capture_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected_faces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()

        key = input("Press Enter to capture another frame or 'q' to quit: ")
        if key.lower() == 'q':
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to start both audio and video capture in separate threads
def main():
    # Create threads for audio and video capture
    audio_thread = threading.Thread(target=capture_and_process_audio)
    video_thread = threading.Thread(target=capture_video)

    # Start both threads
    audio_thread.start()
    video_thread.start()

    # Wait for both threads to finish
    audio_thread.join()
    video_thread.join()

if __name__ == "__main__":
    main() 
