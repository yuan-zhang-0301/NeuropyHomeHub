import os
import sounddevice as sd
import soundfile as sf
import tempfile
import streamlit as st
import openai
import simpleaudio as sa
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------
def play_audio(file_path):
    """Play a pre-recorded .wav audio file."""
    if os.path.exists(file_path):
        try:
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            st.error(f"Error playing audio: {e}")
    else:
        st.error(f"Audio file not found: {file_path}")

# ------------------------------------------------
def record_audio(filename, duration=10, fs=16000):
    """Record audio for a short duration and save to a WAV file."""
    #st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    sf.write(filename, recording, fs)
    #st.success("Recording complete.")
    return filename

# ------------------------------------------------
def transcribe_with_whisper(filename):
    """Transcribe a recorded audio file using the Whisper API."""
    #st.info("Transcribing audio with Whisper API...")
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

# ------------------------------------------------
def analyze_sentiment_with_chatgpt(text):
    prompt = f"""
    Analyze the sentiment of the following text based on the circumplex model of emotions. Identify the primary and secondary (if any) emotion (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and its intensity (Low, Medium, High):\n\n{text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

# ------------------------------------------------
def extract_entities_with_emotions(text):
    prompt = f"""
    Extract the following categories of entities from the text below. For each category, list the relevant details:

    1. **People**: Names of people mentioned in the text.
    2. **Locations**: Specific locations mentioned (e.g., park, apartment, city, bookstore).
    3. **Events**: Key actions or activities described in the text (e.g., walking, kissing, watching a movie).
    4. **Environment Conditions**: The surroundings or environment (e.g., rainy, noisy, cold).
    5. **Emotions**: Identify emotions based on the circumplex model of emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (Low, Medium, High).
    6. **Associations**: For each emotion, provide:
       - People
       - Locations
       - Events
       - Environment Conditions

    Text:
    {text}

    Please provide the output in a structured format.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

# ------------------------------------------------
def process_transcription_with_chatgpt(transcription):
    st.write("### Transcription")
    st.write(transcription)

    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        st.write("### Sentiment Analysis")
        st.write(sentiment)
    except Exception as e:
        st.error(f"Sentiment Analysis Error: {e}")

    try:
        entities = extract_entities_with_emotions(transcription)
        st.write("### Extracted Entities and Emotions")
        st.write(entities)
    except Exception as e:
        st.error(f"Entity Extraction Error: {e}")

# ------------------------------------------------
def continuous_transcription():
    play_audio("tell me about your day.wav")
    st.info("Listening... (say 'that's it' to end)")
    full_transcription = ""

    while True:
        # Create a temporary file to store the current audio chunk.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio_file = tmp.name

        # Record a chunk (now 10 seconds long)
        record_audio(temp_audio_file, duration=10)

        # Transcribe the recorded chunk.
        transcript = transcribe_with_whisper(temp_audio_file)
        st.write("Partial Transcript: " + transcript)
        full_transcription += " " + transcript

        # Check if the termination phrase is in the transcript (and transcript is not empty).
        if transcript and "that's it" in transcript.lower():
            st.success("Termination phrase detected. Stopping recording...")
            play_audio("thanks for sharing.wav")
            break

    process_transcription_with_chatgpt(full_transcription)

# ------------------------------------------------
def main():
    st.title("Neuropy HomeHub")
    st.write("Tell me about your day!")
    
    if st.button("Start Conversation"):
        try:
            continuous_transcription()
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()