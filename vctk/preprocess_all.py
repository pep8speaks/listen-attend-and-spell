from collections import Counter
import librosa
import numpy as np
from joblib import Parallel, delayed

from utils import calculate_mfcc_op


VOCAB_FILENAME = './data/vocab.txt'
SAMPLE_RATE = 16000
N_MFCC = 13
WINDOW = int(.02 * SAMPLE_RATE)
STEP = WINDOW // 2
N_MELS = 40
ALPHA = 0.9

vocabulary = Counter()
running_mean = np.zeros(1, N_MFCC)
running_std = np.zeros(1, N_MFCC)


def read_audio_and_text(inputs):
    audio_path = inputs['file_path']
    text = inputs['text']
    text = ' '.join(text.split())
    for p in ',.:;?!-_':
        text = text.replace(p, '')
    text = text.lower().split()
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return {
        'waveform': audio,
        'text': text
    }


def build_features_and_vocabulary_fn(inputs):
    global running_mean, running_std
    waveform = inputs['waveform']
    text = inputs['text']
    mfcc = calculate_mfcc_op(SAMPLE_RATE, N_MFCC, WINDOW, STEP, N_MELS)(waveform)
    vocabulary.update(text)
    running_mean = ALPHA * running_mean + (1 - ALPHA) * np.mean(mfcc, axis=0)
    running_std = ALPHA * running_std + (1 - ALPHA) * np.std(mfcc, axis=0)
    return {
        'mfcc': mfcc,
        'text': text
    }


if __name__ == "__main__":
    print('Processing audio dataset.')
    fake_data = [
        {
            'file_path': 'data/VCTK-Corpus/wav48/p340/p340_047.wav',
            'text': 'I felt under a lot of pressure.'
        },
        {
            'file_path': 'data/VCTK-Corpus/wav48/p340/p340_210.wav',
            'text': 'I had just taken it off.'
        }
    ]
    a = read_audio_and_text(fake_data[0])
    b = build_features_and_vocabulary_fn(a)
    print(b)
