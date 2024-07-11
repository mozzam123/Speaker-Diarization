import librosa
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio import Audio

# Path to your audio file
audio_path = 'voice_data.wav'

# Load the audio file using librosa
y, sr = librosa.load(audio_path, sr=None)

# Obtain your Hugging Face token and make sure you've accepted the conditions on the model page
# Replace 'YOUR_AUTH_TOKEN' with your actual token
use_auth_token = "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"

# Load the speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=use_auth_token)

# Perform diarization on the whole audio file
diarization_result = pipeline(audio_path)
print(diarization_result)

# Perform diarization on an excerpt of the audio
excerpt = Segment(start=2.0, end=5.0)
waveform, sample_rate = Audio().crop(audio_path, excerpt)
excerpt_diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
print(excerpt_diarization_result)
