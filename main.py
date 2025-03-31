from TTS.api import TTS
import os

# Load the story from file (English)
with open("story_en.txt", "r", encoding="utf-8") as f:
    story = f.read()

# Create output dir
os.makedirs("outputs", exist_ok=True)

# Load a pre trained TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)

# Output path
output_path = "outputs/english.wav"

# Generate speech
tts.tts_to_file(text=story, file_path=output_path)

print(f"English narration saved to: {output_path}")