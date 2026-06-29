import numpy as np
import pyaudio
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 44100
F0 = 18500
F1 = 19500
BIT_DURATION = 0.1
FREQ_TOLERANCE = 300

PREAMBLE = [
    1, 1, 1, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1, 1, 0,
    1, 0, 1, 1, 1, 0, 0, 0,
]

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


def detect_bit(freq: float) -> int:
    if abs(freq - F1) < FREQ_TOLERANCE:
        return 1
    elif abs(freq - F0) < FREQ_TOLERANCE:
        return 0
    return 0


def read_bit(stream, index: int | None = None) -> int:
    raw = stream.read(CHUNK, exception_on_overflow=False)
    chunk = np.frombuffer(raw, dtype=np.float32)
    freq = dominant_freq(chunk, SAMPLE_RATE)
    bit = detect_bit(freq)

    if index is not None and abs(freq - F0) >= FREQ_TOLERANCE and abs(freq - F1) >= FREQ_TOLERANCE:
        print(f"  bit {index}: ambiguous freq={freq:.0f} Hz")

    return bit


def receive(num_bits: int):
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("Waiting for preamble...")
    recent_bits = []

    while True:
        recent_bits.append(read_bit(stream))

        if len(recent_bits) > len(PREAMBLE):
            recent_bits.pop(0)

        if recent_bits == PREAMBLE:
            print("Preamble detected!")
            break

    print(f"Listening for {num_bits} bits...")

    message_bits = []
    for i in range(num_bits):
        message_bits.append(read_bit(stream, i))

    stream.stop_stream()
    stream.close()
    p.terminate()

    text = bits_to_text(message_bits)

    print(f"Received: {text!r}")
    print(f"Raw bits: {''.join(map(str, message_bits))}")

    return text


if __name__ == "__main__":
    message = "hello world"
    receive(len(message) * 8)
