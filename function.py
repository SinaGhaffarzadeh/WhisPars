
import wave
import pyaudio
from faster_whisper import WhisperModel

def record_chunk(p, stream, file_path, chunk_length=1):
    '''
    The `record_chunk` function is responsible for recording a short audio segment (or "chunk") from a 
    microphone and saving it as a `.wav` audio file. It uses the PyAudio and wave libraries to handle
    audio input and file writing.
    It takes raw audio data from a stream (microphone), stores the recorded data in memory,
    and then writes it to a `.wav` file with the appropriate format settings 
    (mono channel, 16-bit depth, 16000 Hz sample rate (2 times of Nyquist)). The length of the recording 
    is defined by `chunk_length` in seconds.

    Input parameters;
    - p: PyAudio class from pyaudio lab
    - stream: the open audio microphone for recording
    - file_path: path to save the recorded WAV file
    - chunk_length: duration (in seconds) of the recording (default is 1 second)

    Tips: 1- Human speech is mostly concentrated in the 0–8 kHz range. According to the Nyquist theorem,
             to capture audio accurately, you need a sample rate at least twice the highest frequency you 
             want to preserve.
          2- Buffer size is the number of samples readed at once from the audio microphone. 
             It aligns well with memory and CPU/GPU architecture.
    '''

    frames = [] # Empty list to store audio data chunks

    for _ in range(0, int(16000 / 1024 * chunk_length)): # Loop to record audio in 1024-sample blocks for the desired duration.
                                                         # Formula: (sample_rate / buffer_size) * duration_seconds
        data = stream.read(1024) # Reads 1024 samples of audio data from the stream
        frames.append(data)

    wf = wave.open(file_path, 'wb') # Opens a new WAV file for writing in binary mode.
    wf.setnchannels(1) # Sets number of audio channels to 1 (mono) / 2 refers to Stereo.
                       # Because we record the audio from one mic and also speech recognition models expect mono input.
                       # Stereo adds complexity and noise.

    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16)) # Sets the sample width based on 16-bit integer audio format (2 bytes per sample)

    wf.setframerate(16000) # Sets the sample rate to 16000 samples per second (Hz)

    wf.writeframes(b''.join(frames)) # Joins all recorded frames into a single bytes object and writes them to the file

    wf.close() # Closes the WAV file to ensure data is saved properly




def transcribe_chunk(model, file_path, lang):
    '''
    The`transcribe_chunk`function, takes an audio file and uses a speech-to-text model to transcribe
    the spoken content into text.
    It is used to transcribe **short audio chunks** into readable text. 
    This is particularly useful in real-time or streaming speech recognition systems,
    where audio is recorded and transcribed in small segments instead of processing an entire large
    audio file at once.

    Inputs of function;
    model: the speech-to-text model used for transcription (such as `WhisperModel` from `faster-whisper`).
    file_path: the path to the audio file to be transcribed.
    lang: the language code for the transcription (e.g., 'fa' for Persian, 'en' for English).

    Tip: Imagine the model has to generate a sentence word by word. At each step, 
        it doesn't just pick the most likely next word — instead, 
        it keeps track of the top N most likely sequences, where N = beam_size.

        For example:

        At step 1, model considers the top 5 possible first words.

        At step 2, for each of those 5 first words, it considers 5 possible next words → 25 combinations.

        It keeps only the 5 most likely sentence fragments (based on their cumulative probability).

        This continues until the sentence ends.

        | Beam Size | Accuracy        | Speed    | Memory Use |
        | --------- | --------------- | -------- | ---------- |
        | 1         | Lowest (greedy) | Fastest  | Very Low   |
        | 5         | Good balance    | Moderate | Moderate   |
        | 10+       | High accuracy   | Slower   | High       |

        Use beam_size=1 if you want real-time transcription.
        Use beam_size=5 or higher for better transcription quality, 
        especially in offline processing or noisy audio.
    '''

    segments, _ = model.transcribe(
    file_path,
    language= lang, # fa = Persian , en = English
    beam_size=1, # It helps the model generate more accurate text output by exploring multiple 
                 # possibilities at each prediction step.
    temperature=0.2, # Lower temperature for more deterministic output
    vad_filter=True  # applies Voice Activity Detection to remove silence and non-speech segments.
    ) 

    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip() # removes any leading or trailing whitespace.