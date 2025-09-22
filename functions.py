from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import sounddevice as sd
import tempfile
import wave

async def speak(text: str, async_client: AsyncOpenAI):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a professional but friendly security-guard style.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

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

def transcribe_audio(file_path, client: OpenAI):
    """Send recorded audio to OpenAI Whisper for transcription."""
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    return transcript.text