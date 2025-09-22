import cv2
import time
import asyncio
from deepface import DeepFace
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import numpy as np
from classes import Chatbot
from functions import record_audio, transcribe_audio, speak
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Camera + Face Recognition Loop ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

chatbot = Chatbot()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        print("Face detected!")

        dfs = DeepFace.verify(
            img1_path=r"C:\Users\rover\Downloads\IMG_20250921_145209.jpg",
            img2_path=r"database/Mateo1.jpg",
            model_name="SFace",
            align=False,
            detector_backend="opencv",
        )

        if dfs["verified"]:
            print("Identity verified ✅")
            
            while True:
                # --- Record & Transcribe Speech ---
                audio_file = record_audio(seconds=5)  # capture 5 seconds
                user_input = transcribe_audio(audio_file)
                print(f"User (STT): {user_input}")

                # --- Generate Response ---
                response = chatbot.generate_response(user_input)
                print(f"Bot: {response}")

                # --- Speak Response ---
                asyncio.run(speak(response))

                if "Access Granted" in response or "Access Denied" in response:
                    print("Conversation ended.")
                    break