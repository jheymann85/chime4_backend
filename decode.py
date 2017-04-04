import argparse
import inspect
import os

import numpy as np
import tensorflow as tf
import wavefile

import kaldi_ops
import tf_wrn

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

# Assests
KALDI_FILES = os.path.join(current_dir, 'assets', 'kaldi_files')
HCLG = os.path.join(KALDI_FILES, 'HCLG.fst')
WORD_SYMS = os.path.join(KALDI_FILES, 'words.txt')
MODEL = os.path.join(KALDI_FILES, 'final.mdl')

ID2W = kaldi_ops.read_word_table(WORD_SYMS)

MEL_OPTS = kaldi_ops.MelOpts(
    num_bins=80, low_freq=20, high_freq=0, vtln_low=100, vtln_high=-500
)

# Parse arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('wav_file', help='Wav audio file to decode')
params = arg_parser.parse_args()

# Read audio
with wavefile.WaveReader(params.wav_file) as wav_reader:
    channels = wav_reader.channels
    assert channels == 1
    assert wav_reader.samplerate == 16000

    samples = wav_reader.frames
    wav_data = np.empty((channels, samples), dtype=np.float32, order='F')
    wav_reader.read(wav_data)
    wav_data = np.squeeze(wav_data)

# Build backend
fbank = kaldi_ops.fbank(
    tf.constant(wav_data * tf.int16.max), mel_opts=MEL_OPTS
)
feature = kaldi_ops.add_deltas(fbank)
feature -= tf.reduce_mean(feature, axis=0, keep_dims=True)
probes = tf_wrn.build_resnet(feature)
decode = kaldi_ops.decode(
    probes['log_likelihoods'], HCLG, MODEL
)

# Decode
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    decode = sess.run(decode)

# Output
print(' '.join(ID2W.get(i, '<UNK>') for i in decode.decode_sequence))
