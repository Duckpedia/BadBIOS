import numpy as np
import pyaudio

SAMPLE_RATE = 44100
F0 = 18500
F1 = 19500
BIT_DURATION = 0.1
AMPLITUDE = 0.5

def text_to_bits(text: str) -> list[int]:
    bits = []
    for char in text:
        byte = ord(char)
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def generate_tone(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    window = np.hanning(len(t))
    return (AMPLITUDE * np.sin(2 * np.pi * freq * t) * window).astype(np.float32)

def transmit(message: str):
    bits = text_to_bits(message)
    print(f"Transmitting {len(bits)} bits: {message!r}")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1,
                    rate=SAMPLE_RATE, output=True)

    for bit in bits:
        freq = F1 if bit else F0
        tone = generate_tone(freq, BIT_DURATION, SAMPLE_RATE)
        stream.write(tone.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Done")

if __name__ == "__main__":
    transmit("hello world")