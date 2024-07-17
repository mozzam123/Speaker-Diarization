import time
import os
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from google.cloud import speech

# Function to save audio segment to .wav file
def save_audio_segment(segment, audio_path, output_dir, speaker_label):
    start, end = segment.start, segment.end
    print("start: ", start)
    print("end: ", end)
    y, sr = librosa.load(audio_path, sr=None, offset=start, duration=end-start)
    output_path = os.path.join(output_dir, f"{speaker_label}_{start:.2f}_{end:.2f}.wav")
    sf.write(output_path, y, sr)
    return output_path

# Function to transcribe audio file
def transcribe_file(audio_file, language="hi-IN"):
    """Transcribe the given audio file asynchronously."""
    client = speech.SpeechClient.from_service_account_json("smooth-splicer-351412-f10e00def7fd.json")
    data = []
    start_time = time.time()
    print(f"started at {time.asctime(time.localtime(start_time))}")
    with open(audio_file, "rb") as ch:
        content = ch.read()
    """
     Note that transcription is limited to a 60 seconds audio file.
     Use a GCS file for audio longer than 1 minute.
    """
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language,
        enable_separate_recognition_per_channel=True,
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    for result in response.results:
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
        chunk_data = {
            "transcript": result.alternatives[0].transcript,
            "confidence": result.alternatives[0].confidence,
        }
        data.append(chunk_data)
    end_time = time.time()
    print(f"started at: {time.asctime(time.localtime(start_time))}")
    print(f"detected at: {time.asctime(time.localtime(end_time))}")
    print(f"Total Time required: {end_time - start_time}s")
    return data[0]

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
print(f"Time taken for diarization: {end_time - start_time} seconds")

# Create a directory to save speaker audio segments
output_dir = 'speaker_segments'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each speaker segment, save to .wav file, and transcribe
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    segment_path = save_audio_segment(turn, audio_path, output_dir, speaker)
    print(f"Transcribing segment: {segment_path}")
    transcription = transcribe_file(segment_path)
    print(f"Transcription: {transcription['transcript']}, Confidence: {transcription['confidence']}")

print("Speaker diarization and transcription completed successfully.")
