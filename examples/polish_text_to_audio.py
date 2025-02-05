import torch
import numpy as np

import queue
import sounddevice as sd
import threading
import time

from whisperspeech.pipeline import Pipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
pipe = Pipeline(device=device)


# Initialize queue
audio_queue = queue.Queue()


def generate_and_play(pipe, text, playback_event):
    # Generate audio and put it in the queue
    audio_tensor = pipe.generate(text, lang="pl", cps=13)
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
            playback_event.wait()  # Wait for the previous playback to finish
            playback_event.clear()  # Reset the event
            sd.play(audio_np, samplerate=24000)
            sd.wait()
            time.sleep(0.05)  # Introduce a 50ms delay
            playback_event.set()  # Signal that this playback has finished

    playback_thread = threading.Thread(target=play_audio, daemon=True)
    playback_thread.start()
    return playback_thread

# Polish text samples
texts = [
    "Dzień dobry, jak się masz?",
    "To jest próbny tekst do syntezy mowy.",
    "WhisperSpeech", " działa świetnie po polsku!"
]


# Create an event to synchronize playback
playback_event = threading.Event()
playback_event.set()  # Initial state: playback can start

# Keep track of all playback threads
playback_threads = []

for text in texts:
    print(f"Generating: {text}")
    thread = generate_and_play(pipe, text, playback_event)
    playback_threads.append(thread)

# Signal all playback threads to exit
for _ in playback_threads:
    audio_queue.put(None)


print("Script complete. Check 'output.wav' for the generated audio.")
