import argparse
import numpy as np
import tensorflow as tf

import utils

from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Listen, Attend and Spell(LAS) implementation based on Tensorflow. '
                    'The model utilizes input pipeline and estimator API of Tensorflow, '
                    'which makes the training procedure truly end-to-end.')

    parser.add_argument('--data', type=str, required=True,
                        help='inference data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of imported model')
    parser.add_argument('--save', type=str, required=True,
                        help='path of saving inference results')

    parser.add_argument('--beam_width', type=int, default=0,
                        help='number of beams (default 0: using greedy decoding)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--delimiter', help='Symbols delimiter. Default: " "', type=str, default=' ')
    parser.add_argument('--output_ipa', action='store_true',
                        help='If model''s outputs are binary features, make the graph output phones')
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')
    parser.add_argument('--spe', help='Use only SPE features', action='store_true')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, norm_filename=None, num_channels=39, batch_size=8, num_epochs=1,
    binf2phone=None):
    binary_targets = binf2phone is not None
    labels_shape = [] if not binary_targets else len(binf2phone.index)
    labels_dtype = tf.string if not binary_targets else tf.float32
    dataset = utils.read_dataset(dataset_filename, num_channels, labels_shape=labels_shape,
        labels_dtype=labels_dtype)
    vocab_table = utils.create_vocab_table(vocab_filename)

    if norm_filename is not None:
        means, stds = utils.load_normalization(args.norm)
    else:
        means = stds = None

    sos = binf2phone[utils.SOS].values if binary_targets else utils.SOS
    eos = binf2phone[utils.EOS].values if binary_targets else utils.EOS

    dataset = utils.process_dataset(
        dataset, vocab_table, sos, eos, means, stds, batch_size, num_epochs,
        binary_targets=binary_targets, labels_shape=labels_shape)

    return dataset.take(10)


def main(args):
    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args)

    hparams.decoder.set_hparam('beam_width', args.beam_width)

    vocab_list = utils.load_vocab(args.vocab)
    binf2phone_np = None
    binf2phone = None
    if not hparams.decoder.binary_outputs:
        vocab_size = len(vocab_list)
    else:
        binf2phone = utils.load_binf2phone(args.binf_map, args.spe, vocab_list)
        vocab_size = len(binf2phone.index)
        vocab_list = binf2phone.columns
        if args.output_ipa:
            binf2phone_np = binf2phone.values

    def model_fn(features, labels,
        mode, config, params):
        return las_model_fn(features, labels, mode, config, params,
            binf2phone=binf2phone_np)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams)

    predictions = model.predict(
        input_fn=lambda: input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels, batch_size=args.batch_size, num_epochs=1,
            binf2phone=binf2phone),
        predict_keys='sample_ids')

    if args.beam_width > 0:
        predictions = [vocab_list[y['sample_ids'][:, 0]].tolist() + [utils.EOS]
                       for y in predictions]
    else:
        predictions = [[vocab_list[x] for x in y['sample_ids']] + [utils.EOS]
                       for y in predictions]

    predictions = [args.delimiter.join(y[:y.index(utils.EOS)]) for y in predictions]

    with open(args.save, 'w') as f:
        f.write('\n'.join(predictions))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
