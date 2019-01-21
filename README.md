# listen-attend-and-spell

## Overview

**listen-attend-and-spell** contains the implementation of [*Listen, Attend and Spell*][las] (LAS) model based on Tensorflow. In this project, the LAS model is trained via input pipeline and estimator API of Tensorflow, which makes the whole procedure truly end-to-end.

## Usage

### Requirements

Tensorflow and Numpy are needed. Run `pip install -r requirements.txt` to get the lastest version.

### Data Preparing
Before running the training script, you should convert your data into TFRecord format. The code snippet down below may help you understand how to create it:
```python
def make_example(inputs, labels):
    feature_lists = tf.train.FeatureLists(feature_list={
        'labels': tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()]))
            for p in labels
        ]),
        'inputs': tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=f))
            for f in inputs
        ])
    })

    return tf.train.SequenceExample(feature_lists=feature_lists)

list_features = [np.random.rand(i * 10, 39) for i in range(10)]
list_labels = np.random.choice(string.ascii_uppercase, size=10)

with tf.python_io.TFRecordWriter('data.tfrecords') as writer:
    for inputs, labels in zip(list_features, list_labels):
        writer.write(make_example(inputs, labels).SerializeToString())
```
Note that it is no need to convert labels into index on your own! Just encode the string-like labels into bytes
More deatils are available at [sequence example][sequence_example].

Moreover, you should create a vocabulary table containing all symbols in your training data. For more details,  please refer to `misc/timit-phone.table`.

### Training and Evaluation
Simply run `python3 train.py --train TRAIN_TFRECORD --vocab VOCAB_TABLE --model_dir MODEL_DIR --norm NORM_FILE`.
You can also specify the validation data and some hyperparameters. To find out more, please run `python3 train.py -h`.
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
                [--use_tpu {yes,no,fake}] [--tpu_name TPU_NAME]
                [--tpu_num_shards TPU_NUM_SHARDS] [--train_steps TRAIN_STEPS]
                [--eval_steps EVAL_STEPS]
                [--tpu_steps_per_checkpoint TPU_STEPS_PER_CHECKPOINT]
                [--max_frames MAX_FRAMES] [--max_size MAX_SIZE]

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
  --use_tpu {yes,no,fake}
                        Use TPU for training?
  --tpu_name TPU_NAME   Name of TPU.
  --tpu_num_shards TPU_NUM_SHARDS
                        Number of TPU shards.
  --train_steps TRAIN_STEPS
                        Max steps for training (required for TPU usage).
  --eval_steps EVAL_STEPS
                        Evaluation steps (required for TPU usage).
  --tpu_steps_per_checkpoint TPU_STEPS_PER_CHECKPOINT
                        TPU step per checkpoint.
  --max_frames MAX_FRAMES
                        Maximum number of input frames.
  --max_size MAX_SIZE   Maximum number of output symbols.
```

### Tensorboard
With the help of tensorflow estimator API, you can launch tensorboard by `tensorboard --logdir=MODEL_DIR`  to see the training procedure.

## Result
### TIMIT
The following figures show the results on TIMIT dataset (4620 training sentence and 1680 testing sentence). If you prepare the TFRecord files of TIMIT, you can run the model with `misc/hparams.json` (put it into your model directory) to produce the similar results. Note that, customarily, we map phones into 39 phone set while evaluating TIMIT dataset, and thus, the edit distance evaluated down below is based on 39 phone set.

![training curve](images/curve.png)

### VCTK
[VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) is also used as benchmark. Before running `./run-vctk.sh` to process data and train the model, *please run `pip install speechpy` to install `speechpy` first*. Note that since VCTK has no train-test split, the first 90 speakers are chose as the training set, and remaining are the testing set. **For more details about how to extract audio features and create TFRecord files, please refer to `vctk/`**.

## References

- [Listen, Attend and spell][las]
- [How to create TFRecord][sequence_example]
- [nabu's implementation][nabu]
- [Tensorflow official seq2seq code][nmt]

## Contact

Issues and pull requests are welcomed. Feel free to [contact me](mailto:windqaq@gmail.com) if there's any problems.

[nabu]: https://github.com/vrenkens/nabu
[nmt]: https://github.com/tensorflow/nmt
[las]: https://arxiv.org/pdf/1508.01211.pdf
[sequence_example]: https://github.com/tensorflow/magenta/blob/master/magenta/common/sequence_example_lib.py
