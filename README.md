# listen-attend-and-spell

## Overview

**listen-attend-and-spell** contains the implementation of [*Listen, Attend and Spell*][las] (LAS) model based on Tensorflow. In this project, the LAS model is trained via input pipeline and estimator API of Tensorflow, which makes the whole procedure truly end-to-end.

## Usage

### Requirements
Run `pip install -r requirements.txt` to get the necessary version.
To do training on phone targets, you'll need `espeak-ng` installed.

### Data Preparing
Before running the training script, you should convert your data into TFRecord format, collect normalization data and prepare vocabulary.
To do that, collect your train and test data in separate CSV files like this:
```csv
filename1.wav,en,big brown fox
filename2.wav,en,another fox
```
Recipes for some datasets are available in `recipes` folder.
After that call data collection script: `process_all.py`.
```text
usage: preprocess_all.py [-h] --input_file INPUT_FILE --output_file
                         OUTPUT_FILE [--norm_file NORM_FILE]
                         [--vocab_file VOCAB_FILE] [--top_k TOP_K]
                         [--n_mfcc N_MFCC] [--n_mels N_MELS] [--window WINDOW]
                         [--step STEP] [--n_jobs N_JOBS]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        File with audio paths and texts.
  --output_file OUTPUT_FILE
                        Target TFRecord file name.
  --norm_file NORM_FILE
                        File name for normalization data.
  --vocab_file VOCAB_FILE
                        Vocabulary file name.
  --top_k TOP_K         Max size of vocabulary.
  --n_mfcc N_MFCC       Number of MFCC coeffs.
  --n_mels N_MELS       Number of mel-filters.
  --window WINDOW       Analysis window length in ms.
  --step STEP           Analysis window step in ms.
  --n_jobs N_JOBS       Number of parallel jobs.
  --targets {words,phones}
                        Determines targets type.
```

### Training and Evaluation
Simply run `python3 train.py --train TRAIN_TFRECORD --vocab VOCAB_TABLE --model_dir MODEL_DIR --norm NORM_FILE`.
You can also specify the validation data and some hyperparameters.
To find out more, please run `python3 train.py -h`.
```text
usage: train.py [-h] --train TRAIN [--valid VALID] --vocab VOCAB [--norm NORM]
                [--mapping MAPPING] --model_dir MODEL_DIR
                [--eval_secs EVAL_SECS] [--encoder_units ENCODER_UNITS]
                [--encoder_layers ENCODER_LAYERS] [--use_pyramidal]
                [--decoder_units DECODER_UNITS]
                [--decoder_layers DECODER_LAYERS]
                [--embedding_size EMBEDDING_SIZE]
                [--sampling_probability SAMPLING_PROBABILITY]
                [--attention_type {luong,bahdanau,custom}]
                [--attention_layer_size ATTENTION_LAYER_SIZE] [--bottom_only]
                [--pass_hidden_state] [--batch_size BATCH_SIZE]
                [--num_channels NUM_CHANNELS] [--num_epochs NUM_EPOCHS]
                [--learning_rate LEARNING_RATE] [--dropout DROPOUT]

Listen, Attend and Spell(LAS) implementation based on Tensorflow. The model
utilizes input pipeline and estimator API of Tensorflow, which makes the
training procedure truly end-to-end.

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         training data in TFRecord format
  --valid VALID         validation data in TFRecord format
  --vocab VOCAB         vocabulary table, listing vocabulary line by line
  --norm NORM           normalization params
  --mapping MAPPING     additional mapping when evaluation
  --model_dir MODEL_DIR
                        path of saving model
  --eval_secs EVAL_SECS
                        evaluation every N seconds, only happening when
                        `valid` is specified
  --encoder_units ENCODER_UNITS
                        rnn hidden units of encoder
  --encoder_layers ENCODER_LAYERS
                        rnn layers of encoder
  --use_pyramidal       whether to use pyramidal rnn
  --decoder_units DECODER_UNITS
                        rnn hidden units of decoder
  --decoder_layers DECODER_LAYERS
                        rnn layers of decoder
  --embedding_size EMBEDDING_SIZE
                        embedding size of target vocabulary, if 0, one hot
                        encoding is applied
  --sampling_probability SAMPLING_PROBABILITY
                        sampling probabilty of decoder during training
  --attention_type {luong,bahdanau,custom}
                        type of attention mechanism
  --attention_layer_size ATTENTION_LAYER_SIZE
                        size of attention layer, see
                        tensorflow.contrib.seq2seq.AttentionWrapperfor more
                        details
  --bottom_only         apply attention mechanism only at the bottommost rnn
                        cell
  --pass_hidden_state   whether to pass encoder state to decoder
  --batch_size BATCH_SIZE
                        batch size
  --num_channels NUM_CHANNELS
                        number of input channels
  --num_epochs NUM_EPOCHS
                        number of training epochs
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout DROPOUT     dropout rate of rnn cell
```

### Tensorboard
With the help of tensorflow estimator API, you can launch tensorboard by `tensorboard --logdir=MODEL_DIR`  to see the training procedure.

## Result
### Phones
### Words
#### TIMIT
The following figures show the results on TIMIT dataset (4620 training sentence and 1680 testing sentence). If you prepare the TFRecord files of TIMIT, you can run the model with `misc/hparams.json` (put it into your model directory) to produce the similar results. Note that, customarily, we map phones into 39 phone set while evaluating TIMIT dataset, and thus, the edit distance evaluated down below is based on 39 phone set.

![training curve](images/curve.png)

## References

- [WindQAQ implementation][original_implementation]
- [Listen, Attend and spell][las]
- [How to create TFRecord][sequence_example]
- [nabu's implementation][nabu]
- [Tensorflow official seq2seq code][nmt]

[original_implementation]: https://github.com/WindQAQ/listen-attend-and-spell
[nabu]: https://github.com/vrenkens/nabu
[nmt]: https://github.com/tensorflow/nmt
[las]: https://arxiv.org/pdf/1508.01211.pdf
[sequence_example]: https://github.com/tensorflow/magenta/blob/master/magenta/common/sequence_example_lib.py
