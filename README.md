# WhisPars 🎙️🇮🇷

**WhisPars** is a real-time multilingual speech-to-text Python pipeline that utilizes [OpenAI's Whisper model](https://github.com/openai/whisper) via the [faster-whisper](https://github.com/guillaumekln/faster-whisper) library. The system is optimized for Persian (Farsi) speech transcription, with a dedicated post-processing module to properly render the Persian script using `arabic-reshaper` and `python-bidi`.

## 🔍 Description

- Captures real-time audio input from the microphone in 1-second chunks.
- Transcribes audio on the fly using the Whisper model.
- Supports multilingual transcription with a focus on **Persian**.
- Fixes RTL (Right-To-Left) and letter-joining issues in Persian text.
- Streams transcription to console and logs it to a text file.
- Designed to be lightweight and usable on CPU (GPU optional).

---

## 📦 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 How It Works

1. Initializes microphone and sets recording parameters (mono, 16kHz).
2. Records audio in short chunks (1 second by default).
3. Transcribes each chunk in real-time using `WhisperModel`.
4. Converts Persian text to correct RTL format.
5. Prints and logs transcription.

---

## 🧠 Language Support

- **Primary**: Persian (`fa`)
- **Others**: Supported by Whisper (e.g., English `en`, etc.)
- **Persian Fixes**: RTL rendering and letter joining using:
  - `arabic_reshaper`
  - `python-bidi`

---

## 📁 File Structure

```bash
WhisPars/
├── main.py                     # Main script that runs the live transcription
├── function.py                 # Helper functions: record_chunk, transcribe_chunk
├── persian_language_converter.py # Handles RTL and reshaping for Persian
├── requirements.txt            # List of dependencies
├── log.txt                     # Final transcription output
```

---

## 🛠️ Usage

```bash
python main.py
```

> Press `Ctrl + C` to stop recording. All transcriptions will be saved to `log.txt`.

---

## 🔊 Microphone Selection

To list available microphones:

```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"Index {i}: {info['name']}")
p.terminate()
```

Set the `input_device_index` in `main.py` to your preferred microphone.

---

## ⚙️ Model Options

Available Whisper model sizes:
- `tiny` (default)
- `small`
- `medium`

> For better accuracy (at the cost of speed), use a larger model.

---

## 📌 Future Work

- 🗂️ Add support for **long-form transcription** and diarization.
- 🌐 Develop a simple **GUI** or **web interface**.

---

## 🧠 Background Concepts

- **Nyquist Theorem**: Human speech mostly lies under 8kHz. Thus, a 16kHz sampling rate is used to preserve intelligibility.
- **Beam Search**:
  - `beam_size=1`: Fastest, real-time (greedy)
  - `beam_size=5+`: Better accuracy, slower


---

## ❤️ Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper by Guillaume Klein](https://github.com/guillaumekln/faster-whisper)
- Persian RTL support: `arabic-reshaper`, `python-bidi`

---
