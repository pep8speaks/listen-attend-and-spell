from collections import Counter
import librosa
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from multiprocessing import Lock
from joblib import Parallel, delayed, dump
from argparse import ArgumentParser

from utils import calculate_mfcc_op


SAMPLE_RATE = 16000

vocabulary = Counter()
means = None
stds = None
total = 0
par_handle = None
session = tf.Session()
mutex = Lock()
waveform_place = tf.placeholder(tf.float32, [None, None])
mfcc_op = None


def make_example(input, label):
    feature_lists = tf.train.FeatureLists(feature_list={
        'labels': tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()]))
            for p in label
        ]),
        'inputs': tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=f))
            for f in input
        ])
    })

    return tf.train.SequenceExample(feature_lists=feature_lists)


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


def build_features_and_vocabulary_fn(args, inputs):
    global means, stds, total
    waveform = inputs['waveform']
    text = inputs['text']
    mfcc = session.run(mfcc_op, {waveform_place: waveform[np.newaxis, :]})[0, :, :]
    vocabulary.update(text)
    if means is None:
        means = np.mean(mfcc, axis=0)
        stds = np.std(mfcc, axis=0)
    else:
        means += np.mean(mfcc, axis=0)
        stds += np.std(mfcc, axis=0)
    total += 1
    return {
        'mfcc': mfcc,
        'text': text
    }


def write_tf_output(writer, inputs):
    with mutex:
        writer.write(make_example(inputs['mfcc'], inputs['text']).SerializeToString())
    par_handle.update()


def process_line(args, writer, line):
    filename, text = line.split(',')
    inputs = {
        'file_path': filename,
        'text': text.strip()
    }
    out = read_audio_and_text(inputs)
    out = build_features_and_vocabulary_fn(args, out)
    write_tf_output(writer, out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', help='File with audio paths and texts.', required=True)
    parser.add_argument('--output_file', help='Target TFRecord file name.', required=True)
    parser.add_argument('--norm_file', help='File name for normalization data.', default=None)
    parser.add_argument('--vocab_file', help='Vocabulary file name.', default=None)
    parser.add_argument('--top_k', help='Max size of vocabulary.', type=int, default=1000)
    parser.add_argument('--n_mfcc', help='Number of MFCC coeffs.', type=int, default=13)
    parser.add_argument('--n_mels', help='Number of mel-filters.', type=int, default=40)
    parser.add_argument('--window', help='Analysis window length in ms.', type=int, default=20)
    parser.add_argument('--step', help='Analysis window step in ms.', type=int, default=10)
    args = parser.parse_args()
    print('Processing audio dataset from file {}.'.format(args.input_file))
    window = int(SAMPLE_RATE * args.window / 1000.0)
    step = int(SAMPLE_RATE * args.step / 1000.0)
    mfcc_op = calculate_mfcc_op(SAMPLE_RATE, args.n_mfcc, window, step, args.n_mels)(waveform_place)
    lines = open(args.input_file, 'r').readlines()
    par_handle = tqdm(unit='sound')
    with tf.io.TFRecordWriter(args.output_file) as writer:
        Parallel(n_jobs=4, prefer="threads")(delayed(process_line)(args, writer, x) for x in lines)
    session.close()
    par_handle.close()
    if args.norm_file is not None:
        dump([means / total, stds / total], args.norm_file)
    if args.vocab_file is not None:
        with open(args.vocab_file, 'w') as f:
            for x, _ in vocabulary.most_common(args.top_k):
                f.write(x + '\n')
