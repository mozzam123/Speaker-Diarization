import time
import os
import librosa
import soundfile as sf
from pyannote.audio import Pipeline

# Function to save audio segment to .wav file
def save_audio_segment(segment, audio_path, output_dir, speaker_label):
    start, end = segment.start, segment.end
    print("start: ", start)
    print("end: ", end)
    y, sr = librosa.load(audio_path, sr=None, offset=start, duration=end-start)
    output_path = os.path.join(output_dir, f"{speaker_label}_{start:.2f}_{end:.2f}.wav")
    sf.write(output_path, y, sr)

# Path to your audio file
audio_path = 'audio_files/voice1.wav'

# Load the audio file using librosa
y, sr = librosa.load(audio_path, sr=None)

# Obtain your Hugging Face token and make sure you've accepted the conditions on the model page
use_auth_token = "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"

# Load the speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=use_auth_token)

# Perform diarization on the whole audio file
start_time = time.time()
diarization_result = pipeline(audio_path)
end_time = time.time()

# Print the result and the time taken
print(diarization_result)
# print(f"Time taken for diarization: {end_time - start_time} seconds")

# Create a directory to save speaker audio segments
output_dir = 'speaker_segments'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each speaker segment and save to .wav file
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    # print("turn: ", turn)
    # print("_: ", _)
    # print("speaker: ", speaker)
    save_audio_segment(turn, audio_path, output_dir, speaker)

# print("Speaker diarization completed and segments saved successfully.")
