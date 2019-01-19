import tensorflow as tf
import tensorflow_transform as tft
import librosa

from utils import calculate_mfcc_op


VOCAB_FILENAME = './data/vocab.txt'
SAMPLE_RATE = 16000
N_MFCC = 13
WINDOW = int(.02 * SAMPLE_RATE)
STEP = WINDOW // 2
N_MELS = 40


def read_audio_and_text(inputs):
    audio_path = inputs['file_path']
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return {
        'waveform': audio
    }


def build_features_and_vocabulary_fn(inputs):
    waveform = inputs['waveform']
    text = inputs['text']
    mfcc = calculate_mfcc_op(SAMPLE_RATE, N_MFCC, WINDOW, STEP, N_MELS)(waveform)
    mfcc_normalized = tft.scale_to_z_score(mfcc, name='mfcc_normalization')
    tft.vocabulary(text, top_k=10000, vocab_filename=VOCAB_FILENAME, name='vocabulary')
    return {
        'mfcc': mfcc_normalized,
        'text': text
    }


if __name__ == "__main__":
    print('Processing audio dataset.')
