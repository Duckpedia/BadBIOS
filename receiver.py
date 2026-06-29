import numpy as np
import pyaudio
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 44100
F0 = 18500
F1 = 19500
BIT_DURATION = 0.1
FREQ_TOLERANCE = 300

PREAMBLE = [1, 0] * 8

CHUNK = int(SAMPLE_RATE * BIT_DURATION)


def dominant_freq(chunk: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(len(chunk))
    spectrum = np.abs(rfft(chunk * window))
    freqs = rfftfreq(len(chunk), 1 / sample_rate)
    return freqs[np.argmax(spectrum)]


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        chars.append(chr(byte))
    return "".join(chars)


def detect_bit(freq):
    if abs(freq - F1) < FREQ_TOLERANCE:
        return 1
    elif abs(freq - F0) < FREQ_TOLERANCE:
        return 0
    return None


def receive(num_bits):
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("Waiting for preamble...")

    bits = []

    while True:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.float32)

        bit = detect_bit(dominant_freq(chunk, SAMPLE_RATE))
        if bit is None:
            continue

        bits.append(bit)

        if len(bits) > len(PREAMBLE):
            bits.pop(0)

        if bits == PREAMBLE:
            print("Preamble detected!")
            break

    print("Receiving message...")

    message_bits = []

    while len(message_bits) < num_bits:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.float32)

        bit = detect_bit(dominant_freq(chunk, SAMPLE_RATE))

        if bit is not None:
            message_bits.append(bit)

    stream.stop_stream()
    stream.close()
    p.terminate()

    text = bits_to_text(message_bits)

    print(f"Received: {text!r}")
    print(f"Bits: {''.join(map(str, message_bits))}")

    return text


if __name__ == "__main__":
    message = "hello world"
    receive(len(message) * 8)
