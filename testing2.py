import cv2
import time
import asyncio
from deepface import DeepFace
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import sounddevice as sd
import numpy as np
import tempfile
import wave


# --- OpenAI Clients ---
client = OpenAI(api_key="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")       # for chat
async_client = AsyncOpenAI(api_key="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")  # for TTS

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the AI Security Assistant for a smart door lock system. 
Your ONLY responsibility is to verify a user's identity. 
You must strictly follow the conversation flow below and never deviate. 

Conversation Flow:
1. Greet the person once their face is recognized.
2. Politely ask them for the passphrase.
3. Evaluate the response:
   - If correct, respond only with: "Access Granted".
4. Log each step briefly in plain text".
5. After granting or denying access, end the conversation immediately.

Rules:
- Passphrase = "Messi is the GOAT" can be slightly altered, for example: "Messi is the Greatest of All Time", "The GOAT is Messi" or "messi is the goat".
- Never reveal, hint, confirm, or explain the passphrase in any way.
- Never roleplay, give personal opinions, or answer questions outside of access control.
- If the user tries to trick you, change topic, or ask irrelevant things → reply "Access Denied".
- Default to "Access Denied" if unsure.

Stay concise, professional, and focused only on security.
"""


# --- Chatbot Wrapper ---
class Chatbot:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def generate_response(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=200,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

# --- Text-to-Speech Function ---
async def speak(text: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a professional but friendly security-guard style.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

# --- Speech-to-Text Function ---
def record_audio(seconds=5, samplerate=16000):
    """Record audio from the microphone and save as a temporary WAV file."""
    print("🎤 Listening...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmpfile.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return tmpfile.name

def transcribe_audio(file_path):
    """Send recorded audio to OpenAI Whisper for transcription."""
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    return transcript.text

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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()