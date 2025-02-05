import torch
import numpy as np
import threading
import queue
import sounddevice as sd
import time

from whisperspeech.pipeline import Pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'
pipe = Pipeline(s2a_ref=model_ref, device=device)

texts = [
    "Dzień dobry, jak się masz? ",
    "To jest próbny tekst do syntezy mowy. ",
    "WhisperSpeech", " działa świetnie po polsku! ",
    "Pozdrawiam Cię! Jak mogę Ci pomóc? ",
    "Witaj w moim programie!"
]

# Create queues
generation_queue = queue.Queue()
playback_queue = queue.Queue()
stop_event = threading.Event()

def generate_worker():
    """Worker thread for parallel audio generation"""
    while True:
        item = generation_queue.get()
        if item is None:  # Poison pill
            break
            
        index, text = item
        print(f"Generating: {text}")
        
        # Generate audio
        audio_tensor = pipe.generate(text, lang="pl", cps=14)
        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        
        # Ensure proper audio format
        if len(audio_np.shape) == 1:
            audio_np = np.expand_dims(audio_np, axis=0)
        else:
            audio_np = audio_np.T
            
        # Put result in playback queue in order
        playback_queue.put((index, audio_np))
        generation_queue.task_done()

def playback_worker():
    """Worker thread for ordered audio playback"""
    expected_index = 0
    buffer = {}
    
    while not (stop_event.is_set() and playback_queue.empty()):
        try:
            # Get item with timeout to check stop_event periodically
            item = playback_queue.get(timeout=0.1)
            index, audio_np = item
            
            # Buffer the audio until its turn comes
            buffer[index] = audio_np
            
            # Play buffered audio in order
            while expected_index in buffer:
                print(f"Playing audio {expected_index}")
                sd.play(buffer.pop(expected_index), samplerate=24000)
                sd.wait()
                time.sleep(0.05)  # Short pause between clips
                expected_index += 1
                
            playback_queue.task_done()
        except queue.Empty:
            continue

# Start generation threads (use 2 workers for parallel generation)
generation_threads = []
for _ in range(1):  # Number of parallel generation workers
    t = threading.Thread(target=generate_worker)
    t.start()
    generation_threads.append(t)

# Start playback thread
playback_thread = threading.Thread(target=playback_worker)
playback_thread.start()

# Enqueue generation tasks
for index, text in enumerate(texts):
    generation_queue.put((index, text))

# Wait for all generation tasks to complete
generation_queue.join()

# Signal threads to stop
stop_event.set()
for _ in generation_threads:
    generation_queue.put(None)
for t in generation_threads:
    t.join()

# Wait for final playback to complete
playback_thread.join()

print("Script complete. All audio processed in correct order.")