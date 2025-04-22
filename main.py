import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS
import torch

STORY_PATH = "story_en.txt"
OUTPUT_DIR = "outputs"
SPANISH_OUTPUT_TEXT = "story_es.txt"
SPANISH_OUTPUT_AUDIO = os.path.join(OUTPUT_DIR, "spanish.wav")
ENGLISH_OUTPUT_AUDIO = os.path.join(OUTPUT_DIR, "english.wav")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load English Story
with open(STORY_PATH, "r", encoding="utf-8") as f:
    english_story = f.read()

# Load multilingual T5 model on Apple Silicon (MPS if available)
print("Loading FLAN-T5 Large model (multilingual)...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

def rewrite_story(original, language, tone):
    prompt = (
        f"Translate the following fairytale into {language}. "
        f"Use a {tone}, storytelling style for kids:\n\n{original}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Generate Spanish Version
print("Rewriting story in Spanish...")
spanish_story = rewrite_story(english_story, "Spanish", "warm Latin American")

# Save the Spanish text
with open(SPANISH_OUTPUT_TEXT, "w", encoding="utf-8") as f:
    f.write(spanish_story)

print(f"Spanish version saved to {SPANISH_OUTPUT_TEXT}")

# Narrate in English
print("Narrating English version...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
tts.tts_to_file(english_story, file_path=ENGLISH_OUTPUT_AUDIO)
print(f"English narration saved to {ENGLISH_OUTPUT_AUDIO}")

# Narrate in Spanish
print("Narrating Spanish version...")
tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=True, gpu=False)
tts.tts_to_file(spanish_story, file_path=SPANISH_OUTPUT_AUDIO)
print(f"Spanish narration saved to {SPANISH_OUTPUT_AUDIO}")