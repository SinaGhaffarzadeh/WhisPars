
'''
Describtion:
This Python script implements a real-time speech-to-text pipeline for transcribing audio input from 
a microphone using OpenAI's WhisperModel via the faster-whisper library. 
It supports multilingual transcription, with specific handling for Persian language formatting. 
Audio is recorded using the pyaudio library, segmented into 1-second chunks, 
and transcribed incrementally. The results are streamed to the console and saved in a cumulative log.

arabic-reshaper==3.0.0
faster-whisper==1.1.1
python-bidi==0.6.6   
torch==2.7.0+cu118
torchaudio==2.7.0+cu118
torchvision==0.22.0+cu118

Future works: 1- run it on GPU and test the accuracy of persian language model
              2- adding a language model 
'''

import torch 
import pyaudio
import wave
import os
from faster_whisper import WhisperModel
from persian_language_converter import persian_lang_converter
from function import record_chunk, transcribe_chunk

# Checking availibity of Cuda on system
print('Cuda is available!', torch.cuda.is_available())  # Should return True
print("The version of Cuda is:",torch.version.cuda)

# Ilustrating number of active mic on system with their indexes
# p = pyaudio.PyAudio()
# print("Available audio input devices:\n")
# for i in range(p.get_device_count()):
#     info = p.get_device_info_by_index(i)
#     if info["maxInputChannels"] > 0:
#         print(f"Index {i}: {info['name']} ({int(info['maxInputChannels'])} channels)")
# p.terminate()


NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def main2():

    model_size = "tiny"  # tiny/small/medium models are availabe to use
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Turning on the mic
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, # Sets the format based on 16-bit integer audio format (2 bytes per sample)
                channels=1, # refer to recording system which is mono (1 chanel) or stereo (2 chanel)
                rate=16000, # refer to sample rate
                input=True, 
                frames_per_buffer=1024, # refer to the samples in each frame
                input_device_index=1 # index of used mic (To extract this index you can use commaneded lines above)
                )


    accumulated_transcription = ""

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            language = "fa"
            record_chunk(p, stream, chunk_file) #  Recording and saving voice as a .wav file
            transcription = transcribe_chunk(model, chunk_file, lang=language) # Transcripting saved file
            if language == "fa": # To avoide the problems of persian language we used persian_lang_converter function as a helper
                print(NEON_GREEN + persian_lang_converter(transcription) + RESET_COLOR)
            else:
                print(NEON_GREEN + transcription + RESET_COLOR)

            os.remove(chunk_file) 
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")
        with open("log.txt", "w", encoding="utf-8") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        print("LOG:" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main2()
