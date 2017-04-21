import argparse
import inspect
import os
import path

import numpy as np
import tensorflow as tf
import wavefile
from tqdm import tqdm

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


# Options
MEL_OPTS = kaldi_ops.MelOpts(
    num_bins=80, low_freq=20, high_freq=0, vtln_low=100, vtln_high=-500
)


# Parse arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('flist', help='Flist to decode. E.g. dt05_simu')
arg_parser.add_argument('chime_root_dir', help='Root directory for CHiME')
arg_parser.add_argument(
    '--output_path',
    type=str,
    help='Transcriptions and lattices will be stored in this path',
    default='results'
)
arg_parser.add_argument(
    '--verbose', action='store_true',
    help='Output transcription for each utterance'
)
params = arg_parser.parse_args()
RESULTS_PATH = path.Path(params.output_path)
RESULTS_PATH.mkdir_p()


# Read audio
def read_audio(file_name):
    with wavefile.WaveReader(file_name) as wav_reader:
        channels = wav_reader.channels
        assert channels == 1
        assert wav_reader.samplerate == 16000
        samples = wav_reader.frames
        wav_data = np.empty((channels, samples), dtype=np.float32, order='F')
        wav_reader.read(wav_data)
        wav_data = np.squeeze(wav_data)
        return wav_data


# Build backend
print('Building model... ', end='')
wav_data = tf.placeholder(tf.float32, shape=(None,), name='audio')
utt_id_tensor = tf.placeholder(tf.string, name='utt_id')
fbank = kaldi_ops.fbank(
    wav_data * tf.int16.max, mel_opts=MEL_OPTS
)
feature = kaldi_ops.add_deltas(fbank)
feature -= tf.reduce_mean(feature, axis=0, keep_dims=True)
probes = tf_wrn.build_resnet(feature)
decode_opts = kaldi_ops.DecodeOpts(
    lattice_ark_file=str(RESULTS_PATH / params.flist + '.ark')
)
decode_op = kaldi_ops.decode(
    probes['log_likelihoods'], HCLG, MODEL, decode_opts=decode_opts,
    utt_id=utt_id_tensor
)
print(' - done')


# Read filelist
def read_chime_flist(flist_name):
    chime_root = path.Path(params.chime_root_dir)
    flist_file = chime_root / 'annotations' / flist_name + '_1ch_track.list'
    print('Reading filelist {}'.format(flist_file), end='')
    flist = list()
    with open(flist_file) as fid:
        for line in fid:
            flist.append(
                chime_root / 'audio' / '16kHz' / 'isolated_1ch_track' /
                '.'.join(line.strip().split('.')[::2]) # Remove .CHx
            )
    print(' - found {} audio files'.format(len(flist)))
    return flist


# Decode
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    flist = read_chime_flist(params.flist)
    decodes = dict()
    print('Start decoding')
    for audio_file in tqdm(flist):
        audio_data = read_audio(str(audio_file))
        utt_id = str(audio_file.basename()).split('.')[0]
        decode = sess.run(
            decode_op,
            {
                wav_data: audio_data,
                utt_id_tensor: utt_id
            }
        )
        decode_txt = ' '.join(
            ID2W.get(i, '<UNK>') for i in decode.decode_sequence
        )
        decodes[utt_id] = decode_txt
        if params.verbose:
            print(utt_id, decode_txt)


# Write output
with open(str(RESULTS_PATH / params.flist + '.decode'), 'w') as fid:
    for utt_id, transcription in sorted(decodes.items()):
        fid.write(utt_id)
        fid.write('\t')
        fid.write(transcription)
        fid.write('\n')
