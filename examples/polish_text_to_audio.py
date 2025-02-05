import numpy as np

import queue

from whisperspeech.pipeline import Pipeline

pipe = Pipeline()

# Initialize queue
audio_queue = queue.Queue()

def generate_and_play(pipe, text):
    # Generate audio and put it in the queue
    audio_tensor = pipe.generate(text, lang="pl", cps=14)
    audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
    if len(audio_np.shape) == 1:
        audio_np = np.expand_dims(audio_np, axis=0)
    else:
        audio_np = audio_np.T
    audio_queue.put(audio_np)

    # Play audio from the queue
    def play_audio():
        while True:
            audio_np = audio_queue.get()
            if audio_np is None:
                break
            sd.play(audio_np, samplerate=24000)
            sd.wait()

    playback_thread = threading.Thread(target=play_audio)
    playback_thread.start()

# Polish text samples
texts = [
    "Dzień dobry, jak się masz?",
    "To jest próbny tekst do syntezy mowy.",
    "WhisperSpeech", " działa świetnie po polsku!"
]


for text in texts:
    print(f"Generating: {text}")
    generate_and_play(pipe, text)

print("Script complete. Check 'output.wav' for the generated audio.")
