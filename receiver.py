import numpy as np
import pyaudio
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 44100
F0 = 18500
F1 = 19500
BIT_DURATION = 0.1
FREQ_TOLERANCE = 300

CHUNK = int(SAMPLE_RATE * BIT_DURATION)

def dominant_freq(chunk: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(len(chunk))
    spectrum = np.abs(rfft(chunk * window))
    freqs = rfftfreq(len(chunk), 1 / sample_rate)
    return freqs[np.argmax(spectrum)]

def bits_to_text(bits: list[int]) -> str:
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        if 32 <= byte <= 126:
            chars.append(chr(byte))
    return "".join(chars)

def receive(num_bits: int):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1,
                    rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)

    print(f"Listening for {num_bits} bits...")
    bits = []

    for i in range(num_bits):
        raw = stream.read(CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.float32)
        freq = dominant_freq(chunk, SAMPLE_RATE)

        if abs(freq - F1) < FREQ_TOLERANCE:
            bits.append(1)
        elif abs(freq - F0) < FREQ_TOLERANCE:
            bits.append(0)
        else:
            bits.append(0)
            print(f"  bit {i}: ambiguous freq={freq:.0f} Hz")

    stream.stop_stream()
    stream.close()
    p.terminate()

    text = bits_to_text(bits)
    print(f"Received: {text!r}")
    print(f"Raw bits: {''.join(map(str, bits))}")
    return text

if __name__ == "__main__":
    message = "hello world"
    num_bits = len(message) * 8
    receive(num_bits)