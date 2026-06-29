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
READ_CHUNK = CHUNK // 4
PREAMBLE_THRESHOLD = 0.95


def dominant_freq(chunk: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(len(chunk))
    spectrum = np.abs(rfft(chunk * window))
    freqs = rfftfreq(len(chunk), 1 / sample_rate)
    return freqs[np.argmax(spectrum)]


def classify_bit(chunk: np.ndarray) -> tuple[int | None, float]:
    window = np.hanning(len(chunk))
    spectrum = np.abs(rfft(chunk * window))
    freqs = rfftfreq(len(chunk), 1 / SAMPLE_RATE)

    f0_energy = spectrum[np.argmin(np.abs(freqs - F0))]
    f1_energy = spectrum[np.argmin(np.abs(freqs - F1))]

    if max(f0_energy, f1_energy) == 0:
        return None, 0.0

    bit = 1 if f1_energy > f0_energy else 0
    confidence = abs(f1_energy - f0_energy) / max(f0_energy, f1_energy)

    if confidence < 0.25:
        return None, confidence

    return bit, confidence


def preamble_match_score(samples: np.ndarray) -> float:
    matches = 0

    for index, expected_bit in enumerate(PREAMBLE):
        start = index * CHUNK
        end = start + CHUNK
        bit, _ = classify_bit(samples[start:end])

        if bit == expected_bit:
            matches += 1

    return matches / len(PREAMBLE)


def find_preamble(samples: np.ndarray) -> int | None:
    preamble_samples = len(PREAMBLE) * CHUNK

    if len(samples) < preamble_samples:
        return None

    best_offset = None
    best_score = 0.0
    max_offset = len(samples) - preamble_samples

    for offset in range(0, max_offset + 1, READ_CHUNK // 4):
        score = preamble_match_score(samples[offset : offset + preamble_samples])

        if score > best_score:
            best_score = score
            best_offset = offset

    if best_score >= PREAMBLE_THRESHOLD:
        return best_offset

    return None


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
        frames_per_buffer=READ_CHUNK,
    )

    print("Waiting for preamble...")

    preamble_samples = len(PREAMBLE) * CHUNK
    max_buffer_samples = preamble_samples + CHUNK
    buffer = np.array([], dtype=np.float32)
    pending = np.array([], dtype=np.float32)

    while True:
        raw = stream.read(READ_CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.float32)
        buffer = np.concatenate((buffer, chunk))

        if len(buffer) > max_buffer_samples:
            buffer = buffer[-max_buffer_samples:]

        preamble_offset = find_preamble(buffer)
        if preamble_offset is not None:
            payload_start = preamble_offset + preamble_samples
            pending = buffer[payload_start:]
            print("Preamble detected!")
            break

    print("Receiving message...")

    message_bits = []

    while len(message_bits) < num_bits:
        while len(pending) < CHUNK:
            raw = stream.read(READ_CHUNK, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.float32)
            pending = np.concatenate((pending, chunk))

        bit_chunk = pending[:CHUNK]
        pending = pending[CHUNK:]

        bit, _ = classify_bit(bit_chunk)

        if bit is None:
            bit = detect_bit(dominant_freq(bit_chunk, SAMPLE_RATE))

        message_bits.append(0 if bit is None else bit)

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
