from scipy.signal import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import math
from google.colab import files
import librosa
import random


#uploaded = files.upload()



def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


"""
Tools for Townsend's ADT
Replace's abbey road's version
Also includes prototypes for synths and other similar things.
"""

def pad(arr,d):
    base=[0]*d
    base.extend(arr)
    base.extend([0]*d)
    return base

def unpad(arr):
    i = 0
    while i < len(arr) and arr[i]==0:
        i+=1
    k = len(arr)-1
    while k >= 0 and arr[k]==0:
        k-=1
    return arr[i:k+1]

def adt(arr, d):
    # Pad with d zeros on both ends
    newArr = pad(arr, d)
    # Apply ADT transform: sum with delayed version
    transformed = [newArr[i] + newArr[i - d] for i in range(d, len(newArr))]
    return unpad(transformed)
import numpy as np

def stack_with_delay(base_track, overlay_track, delay_sec, fs=44100):
    delay_samples = int(delay_sec * fs)
    
    # Length of the output track = max length needed to fit both
    total_length = max(len(base_track), delay_samples + len(overlay_track))
    
    # Initialize output array with zeros
    combined_track = np.zeros(total_length, dtype=np.float32)
    
    # Add base track from the start
    combined_track[:len(base_track)] += base_track
    
    # Add overlay track starting at delay_samples
    combined_track[delay_samples:delay_samples + len(overlay_track)] += overlay_track
    
    return combined_track

def shift_and_adt(arr, d, ammount,fs=44100):
    newArr = pad(arr, d)
    length = len(newArr)

    # Extract delayed portion (as float32 for librosa)
    delayed = np.array(newArr[:length - d], dtype=np.float32)

    # Random pitch shift in semitones between -0.1 and 0.1
    random_semitone = random.uniform(-ammount, ammount)

    # Use librosa pitch_shift (preserves length)
    shifted_delayed = librosa.effects.pitch_shift(delayed, sr=fs, n_steps=random_semitone)

    # Sum current samples with pitch-shifted delayed samples
    transformed = []
    for i in range(d, length):
        current_sample = newArr[i]
        delayed_sample = shifted_delayed[i - d]
        transformed.append(current_sample + delayed_sample)

    return unpad(transformed)

def chorus_effect(arr,d,steps):
  for i in range(steps):
    arr=shift_and_adt(arr,d,0.1/steps)
  return arr

def rev_adt(arr, d):
    # Pad with d zeros on both ends (same as adt)
    padded_output = pad(arr, d)
    recovered = [0]*len(padded_output)
    for i in range(d, len(padded_output)):
        recovered[i] = padded_output[i] - recovered[i - d]
    # Remove padding
    return unpad(recovered)

import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
def cubic_spline(p0, p1, p2, p3, t):
    a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
    b = p0 - 2.5*p1 + 2*p2 - 0.5*p3
    c = -0.5*p0 + 0.5*p2
    d = p1
    
    return ((a * t + b) * t + c) * t + d

def get_to_speed(input_signal, fs=44100, start_speed=0, end_speed=1.0, ramp_time=1.0):
    input_signal=np.array(input_signal)
    duration = len(input_signal) / fs
    output_duration = ramp_time / ((start_speed + end_speed) / 2) + (duration - ramp_time) / end_speed
    output_len = int(output_duration * fs)
    output_signal = np.zeros(output_len)
    
    for i in range(output_len):
        t = i / fs
        
        if t < ramp_time:
            input_t = start_speed * t + ((end_speed - start_speed) / (2 * ramp_time)) * t**2
        else:
            pos_at_ramp = start_speed * ramp_time + ((end_speed - start_speed) / (2 * ramp_time)) * ramp_time**2
            input_t = pos_at_ramp + end_speed * (t - ramp_time)
        
        input_index = input_t * fs
        if input_index >= input_signal.size - 1:
            input_index = input_signal.size - 1.000001
        
        idx_floor = int(np.floor(input_index))
        idx_ceil = idx_floor + 1
        
        alpha = input_index - idx_floor
        output_signal[i] = (1 - alpha) * input_signal[idx_floor] + alpha * input_signal[idx_ceil]
    
    return output_signal

def hi_hat_hit(duration=0.4, fs=44100):
    # Generate white noise
    noise = np.random.normal(0, 1, int(fs * duration))
    
    # High-pass filter to simulate bright metallic sound (e.g., above 5000 Hz)
    def highpass_filter(data, cutoff=5000, fs=44100, order=6):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)
    
    filtered_noise = highpass_filter(noise, cutoff=6000, fs=fs)
    
    # Very fast amplitude envelope (quick attack and very quick decay)
    envelope = np.exp(-50 * np.linspace(0, duration, int(fs * duration)))
    
    hi_hat = filtered_noise * envelope
    
    # Normalize
    hi_hat /= np.max(np.abs(hi_hat)) if np.max(np.abs(hi_hat)) > 0 else 1
    
    return hi_hat.astype(np.float32)

fs = 44100  # sample rate
duration = 0.5  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Frequency drops from high_freq to low_freq exponentially
high_freq = 150  # starting frequency (Hz)
low_freq = 50    # ending frequency (Hz)

# Exponential frequency drop over time
freq = high_freq * (low_freq / high_freq) ** (t / duration)

# Instantaneous phase for frequency modulation
phase = 2 * np.pi * np.cumsum(freq) / fs

# Generate sine wave with pitch drop
kick = np.sin(phase)

# Amplitude envelope: fast attack, quick decay
attack_time = 0.01  # 10 ms attack
decay_time = 0.3    # 300 ms decay
envelope = np.concatenate((
    np.linspace(0, 1, int(fs * attack_time)),  # attack
    np.linspace(1, 0, int(fs * decay_time)),   # decay
))
# Pad envelope if shorter than signal
if len(envelope) < len(kick):
    envelope = np.pad(envelope, (0, len(kick) - len(envelope)), 'constant')

kick = kick * envelope

# Normalize to -1..1
kick /= np.max(np.abs(kick))

duration = 0.5  # seconds

# Generate white noise
noise = np.random.normal(0, 1, int(fs * duration))

def bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

filtered_noise = bandpass_filter(noise, 1000, 8000, fs)

# Create an amplitude envelope (fast attack, exponential decay)
envelope = np.exp(-5 * np.linspace(0, duration, int(fs * duration)))
snare_sound = filtered_noise * envelope

# Normalize
snare_sound /= np.max(np.abs(snare_sound))

# Convert to float32
kick_f32 = kick.astype(np.float32)
snare_f32 = snare_sound.astype(np.float32)

# Normalize just in case
kick_f32 /= np.max(np.abs(kick_f32))
snare_f32 /= np.max(np.abs(snare_f32))

# Concatenate
combined = np.concatenate((kick_f32, snare_f32))


def squareWave(freq, sampleRate, leng):
    ret = []
    amplitude = 2000
    samples = int(leng * sampleRate)
    period_samples = sampleRate / freq
    half_period = period_samples / 2

    curr = amplitude
    time_since_toggle = 0

    for i in range(samples):
        ret.append(curr)
        time_since_toggle += 1
        if time_since_toggle >= half_period:
            curr = -curr
            time_since_toggle = 0

    return ret
def sawWave(freq, sampleRate, leng):
  ret=[]
  curr=(sampleRate//freq)//2
  for i in range(int(sampleRate*leng)):
    curr-=1
    if curr==-(sampleRate//freq)//2:
      curr=(sampleRate//freq)//2
    ret.append(curr)
  return ret

def sineWave(freq, sampleRate, leng, amplitude=1000):
    ret = []
    for i in range(int(sampleRate * leng)):
        phase = 2 * math.pi * freq * (i / sampleRate)
        ret.append(math.sin(phase) * amplitude)
    return ret

def piano(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = 0.6 * np.sin(2 * np.pi * freq * t) + \
           0.3 * np.sin(2 * np.pi * freq * 2 * t) + \
           0.1 * np.sin(2 * np.pi * freq * 3 * t)

    return taper(wave,0,fs//10)

def custom_synth(freq, sampleRate, leng, harmonics):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = np.zeros_like(t)

    for multiplier, amplitude in harmonics:
        wave += amplitude * np.sin(2 * np.pi * freq * multiplier * t)

    return taper(wave, 0, sampleRate // 10)
kick_f32 = [i*60 for i in kick_f32]
def drums(beat_pattern, shuffle=0.0):
    beat_wave = []
    beat_spacing = 0.4  # 400 ms between beats
    tot = 0.0  # running time in beats

    for i in range(len(beat_pattern)):
        # Compute delay with optional shuffle
        if shuffle > 0 and i % 2 == 1:
            delay = tot * beat_spacing + (shuffle * beat_spacing * 0.5)
        else:
            delay = tot * beat_spacing

        # Add kick/snare/hi-hat
        if beat_pattern[i] == "kick":
            beat_wave = stack_with_delay(beat_wave, kick_f32, delay)
        elif beat_pattern[i] == "snare":
            beat_wave = stack_with_delay(beat_wave, snare_f32, delay)
        
        # Add hi-hat always
        beat_wave = stack_with_delay(beat_wave, hi_hat_hit(), delay)

        # Add silent beat for None
        if beat_pattern[i] is None:
            pass  # just skip, no need to add silence explicitly here

        tot += 1

    return beat_wave


def applyReverb(audio, delay_ms=50, decay=0.5, repeats=5, fs=44100):
    delay_samples = int(fs * delay_ms / 1000)
    reverb = np.copy(audio).astype(np.float32)
    
    for i in range(1, repeats + 1):
        delayed = np.pad(audio, (delay_samples * i, 0))[:len(audio)]
        reverb += decay**i * delayed
    
    # Normalize to avoid clipping
    reverb /= np.max(np.abs(reverb))
    return reverb

def taper(arr, start, end):
    for i in range(start):
        arr[i] *= (i / start)
    for i in range(end):
        arr[-(i + 1)] *= (i / end)
    return arr

def toSemiTones(note):
    if len(note) == 2:
        note_part = note[0]
        octave = int(note[1])
    else:
        note_part = note[:2]
        octave = int(note[2])

    chromat = {
        "C": 0,
        "C#": 1, "Db": 1,
        "D": 2,
        "D#": 3, "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6, "Gb": 6,
        "G": 7,
        "G#": 8, "Ab": 8,
        "A": 9,
        "A#": 10, "Bb": 10,
        "B": 11
    }

    semitones_from_C4 = (octave - 4) * 12 + chromat[note_part]
    semitones_from_A4 = semitones_from_C4 - 9
    return semitones_from_A4

def noteToFreq(note):
    semi = toSemiTones(note)
    return 440 * 2 ** (semi / 12)

def bass_synth(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)

    # Basic waveform: fundamental + 2nd and 3rd harmonics with decreasing amplitude
    wave = (1.0 * np.sin(2 * np.pi * freq * t) +
            0.5 * np.sin(2 * np.pi * freq * 2 * t) +
            0.25 * np.sin(2 * np.pi * freq * 3 * t))

    # Apply clipping distortion to add growl (soft clipping)
    wave = np.tanh(wave * 3)  # increase 3 for more distortion

    # Envelope: slow attack and medium decay for bass fullness
    attack = np.clip(t * 5, 0, 1)
    decay = np.exp(-t * 3)
    envelope = attack * decay
    wave *= envelope

    # Optional low-pass filter to mellow sound
    def lowpass_filter(data, cutoff=400, fs=sampleRate, order=4):
        b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        return lfilter(b, a, data)
    
    wave = lowpass_filter(wave)

    # Normalize
    wave /= np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1

    return wave

def harmonica(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = np.zeros_like(t)

    # Add odd harmonics with decreasing amplitude
    for i in range(1, 20, 2):  # odd harmonics only
        wave += (1.0 / i**0.8) * np.sin(2 * np.pi * freq * i * t)

    # Add filtered noise to simulate reed noise
    noise = np.random.normal(0, 1, len(t))
    filtered_noise = bandpass_filter(noise, 1000, 3000, sampleRate)
    wave += 0.05 * filtered_noise  # subtle noise

    # Tremolo (slow amplitude modulation)
    tremolo = 1 + 0.15 * np.sin(2 * np.pi * 5 * t)
    wave *= tremolo

    # Envelope: fast attack, moderate decay
    attack = np.clip(t * 20, 0, 1)
    decay = np.exp(-t * 2)
    envelope = attack * decay
    wave *= envelope

    # Smooth taper at start and end
    return taper(wave, start=sampleRate//50, end=sampleRate//50)




def violin(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = np.zeros_like(t)

    # Bright sawtooth-like harmonics, but shaped over time
    for i in range(1, 12):
        envelope = np.clip(t * 4, 0, 1)
        amp = (1.0 / i) * envelope
        wave += amp * np.sin(2 * np.pi * freq * i * t)

    env = np.clip(t * 3, 0, 1) * np.exp(-t * 1.5)
    wave *= env

    ret = wave  # short taper
    return ret[::-1]


def flute(freq, sampleRate, leng):  
    fluteHarmonics = [
        (1, 1.0),
        (2, 0.1),
        (3, 0.05)
    ]
    return taper(custom_synth(freq, sampleRate, leng, fluteHarmonics),fs//10,0)
# Build the full waveform

chorus = [
    ("G4", 0.4), 
    ("G4", 0.4), 
    ("A4", 0.8), 
    ("G4", 0.8),
    ("C5", 0.8), 
    ("B4", 1.2),
    ("G4", 0.4), 
    ("G4", 0.4), 
    ("A4", 0.8), 
    ("G4", 0.8), 
    ("D5", 0.8), 
    ("C5", 1.2),
    ("G4", 0.4), 
    ("G4", 0.4), 
    ("G5", 0.8), 
    ("E5", 0.8), 
    ("C5", 0.8), 
    ("B4", 0.8), 
    ("A4", 1.2),
    ("F5", 0.4), 
    ("F5", 0.4), 
    ("E5", 0.8), 
    ("C5", 0.8), 
    ("D5", 0.8), 
    ("C5", 1.2),
]

verse = [
    ("E4", 0.4), 
    ("F4", 0.4), 
    ("G4", 0.8), 
    ("E4", 0.8),
    ("A4", 0.8), 
    ("G4", 1.2),

    ("C4", 0.4), 
    ("D4", 0.4), 
    ("E4", 0.8), 
    ("G4", 0.8), 
    ("F4", 0.8), 
    ("E4", 1.2),

    ("A4", 0.4), 
    ("G4", 0.4), 
    ("C5", 0.8), 
    ("B4", 0.8), 
    ("A4", 0.8), 
    ("F4", 0.8), 
    ("E4", 1.2),

    ("D4", 0.4), 
    ("E4", 0.4), 
    ("F4", 0.8), 
    ("G4", 0.8), 
    ("F4", 0.8), 
    ("G4", 2),
]

canon_melody_in_c = [
    ("C4", 0.2),
    ("E4", 0.4),
    ("G4", 0.4),
    ("A4", 0.8),
    ("G4", 0.4),
    ("F4", 0.4),
    ("E4", 0.8),

    ("D4", 0.4),
    ("F4", 0.4),
    ("G4", 0.8),
    ("F4", 0.4),
    ("E4", 0.4),
    ("D4", 0.8),

    ("C4", 0.4),
    ("E4", 0.4),
    ("F4", 0.8),
    ("E4", 0.4),
    ("D4", 0.4),
    ("C4", 0.4),
    ("D4", 0.4),
    ("C4", 0.4),
    ("D4", 0.4),
    ("C4", 1.2),
]

def playSong(song,instrument):
  full_wave = []
  for note, duration in song:
      freq = noteToFreq(note)
      wave = instrument(freq, fs, duration)
      full_wave.extend(wave)
  return full_wave
music = playSong(chorus, piano)
rev_music = rev_adt(music, 2 * fs)
music = [music[i] * 6 + rev_music[i] for i in range(len(music))]

verse = adt(playSong(verse, piano), fs * 2)
verse = [i * 6 for i in verse]
finalVerse = verse

verse += music
repLen = len(verse)

if isinstance(verse, np.ndarray):
    verse = verse.tolist()
bridge=playSong(canon_melody_in_c, flute)
bridge=[sample_point*10 for sample_point in bridge]

drum_loop = drums(["kick", "kick", "snare", None], 0)
drum_loop = [i * 2 for i in drum_loop]

verse = [i * 3 for i in verse]
verse = verse * 3

start = repLen // fs
uptoBridge=len(verse)
verse.extend(bridge)

while (start + 0.4 * 4) < uptoBridge//fs+4:
    verse = stack_with_delay(verse, drum_loop, start)
    start += 0.4 * 4

verse = get_to_speed(verse)
if isinstance(verse, np.ndarray):
    verse = verse.tolist()

if isinstance(finalVerse, np.ndarray):
    finalVerse = finalVerse.tolist()

verse.extend(finalVerse)

verse = verse[::-1]

verse = get_to_speed(verse)
verse = verse[::-1]

Audio(verse, rate=fs)
