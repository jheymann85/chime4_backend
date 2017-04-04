import os
from collections import namedtuple

import tensorflow as tf
import inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

base_dir = os.path.join(current_dir, 'user_ops')
try:
    _kaldi_module = tf.load_op_library(base_dir + '/libkaldi.so')
except tf.errors.NotFoundError as e:
    try:
         _kaldi_module = tf.load_op_library(base_dir + '/libkaldi.dylib')
    except:
        raise e

Decode = namedtuple(
    'Decode', ('decode_sequence', 'alignment', 'num_frames',
               'likelihood', 'partial')
)
DecodeOptsBase = namedtuple(
    'DecodeOpts',
    ('acoustic_scale', 'beam', 'max_active', 'min_active', 'lattice_beam',
     'prune_interval', 'beam_delta', 'hash_ratio', 'prune_scale', 'allow_partial')
)
class DecodeOpts(DecodeOptsBase):
    """
    Args:
        acoustic_scale (float): Scaling factor for acoustic likelihoods
        beam (int32): Decoding beam. Larger->slower, more accurate.
        max_active (int32): Decoder max active states. Larger->slower.
        min_active (int32): Decoder minimum #active states
        lattice_beam (float): Lattice generation beam.
            Larger->slower and deeper lattice
        prune_interval (int32): Interval (in frames) at which to prune tokens
        beam_delta (float): Increment used in decoding -- this parameter is
            obscure and relates to a speedup in the way the max-active
            constraint is applied. Larger is more accurate.
        hash_ratio (float): Setting used in decoder to control hash behavior
        prune_scale (float): It affects the algorithm that prunes
            the tokens as we go.
        allow_partial (bool): If true, produce output even if end state
            was not reached
    """
    __slots__ = ()

default_DecodeOpts = DecodeOpts(
    acoustic_scale=0.1, beam=10, max_active=2147483647, min_active=200,
    lattice_beam=10, prune_interval=25, beam_delta=0.5, hash_ratio=2.0,
    prune_scale=0.1, allow_partial=True
)

def decode(
        log_likelihoods, decode_fst_filename, model_filename,
        decode_opts=default_DecodeOpts
    ):
    """ Decode given log likelihoods, a transition model and the HCLG graph.

    Args:
        log_likelihoods (Tensor): Log likelihoods for each frame for each state
        decode_fst_filename (string): Filename for the HCLG graph
        model_filename (string): Filename for the transition model
        decode_opts (DecodeOpts): Options for the decode

    """
    log_likelihoods = tf.convert_to_tensor(log_likelihoods)
    assert os.path.isfile(decode_fst_filename)
    assert os.path.isfile(model_filename)
    assert log_likelihoods.get_shape().ndims == 2
    dec = _kaldi_module.decode(
        log_likelihoods,
        decode_fst_filename, model_filename,
        *decode_opts
    )
    return Decode(
        dec[0], dec[1], dec[2], dec[3], dec[4]
    )


FbankOptsBase = namedtuple(
    'FbankOpts',
    ('use_energy', 'energy_floor', 'raw_energy', 'htk_compat',
     'use_log_fbank', 'use_power')
)
class FbankOpts(FbankOptsBase):
    """
    Args:
        use_energy (bool): Append an extra dimension with energy.
        energy_floor (float):
        raw_energy (bool): Compute energy before preemphasis and windowing.
        htk_compat (bool): Put energy last (if using energy).
        use_log_fbank (bool): Produce log-filterbank, else linear.
        use_power (bool): Use power in filterbank analysis, else magnitude.
    """
    __slots__ = ()
default_FbankOpts = FbankOpts(
    use_energy=False, energy_floor=0.0, raw_energy=True, htk_compat=False,
    use_log_fbank=True, use_power=True
)

FrameOptsBase = namedtuple(
    'FrameOpts',
    ('frame_shift_ms', 'frame_length_ms', 'dither', 'preemph_coeff',
     'remove_dc_offset', 'window_type', 'round_to_power_of_two',
     'blackman_coeff', 'snip_edges')
)
class FrameOpts(FrameOptsBase):
    """
    Args:
        frame_shift_ms (float): Frame shift in milliseconds.
        frame_length_ms (float): Frame length in milliseconds.
        dither (float): Amount of dithering, 0.0 means no dither.
        preemph_coeff (float): Preemphasis coefficient.
        remove_dc_offset (bool): Substract mean of wave before FFT.
        window_type (string): e.g. Hamming window.
        round_to_power_of_two (bool):
        blackman_coeff (float):
        snip_edges (bool):
    """
    __slots__ = ()

default_FrameOpts = FrameOpts(
    frame_shift_ms=10.0, frame_length_ms=25.0, dither=1.0, preemph_coeff=0.97,
    remove_dc_offset=True, window_type='povey', round_to_power_of_two=True,
    blackman_coeff=0.42, snip_edges=True
)

MelOptsBase = namedtuple(
    'MelOpts',
    ('num_bins', 'low_freq', 'high_freq', 'vtln_low', 'vtln_high')
)
class MelOpts(MelOptsBase):
    """
    Args:
        num_bins (int32): Number of triangular bins.
        low_freq (float): Lower frequency cutoff.
        high_freq (float): An upper frequency cutoff; 0 -> no cutoff,
            negative -> added to the Nyquist frequency to get the cutoff.
        vtln_low (float): vtln lower cutoff of warping function.
        vtln_high (float): vtln upper cutoff of warping function:
            if negative, added to the Nyquist frequency to get the cutoff.
    """
    __slots__ = ()
default_MelOpts = MelOpts(
    num_bins=25, low_freq=20, high_freq=0, vtln_low=100, vtln_high=-500
)

def fbank(
        wav_data, vtln_warp=tf.constant(1.0, dtype=tf.float32),
        fbank_opts=default_FbankOpts, frame_opts=default_FrameOpts,
        mel_opts=default_MelOpts
    ):
    """ Creates Mel-Filterbank features given raw audio data.

    Args:
        wav_data (Tensor[float]): 1-D tensor with audio data. Must be between 0
            and max(int16)
        vtln_warp (Tensor[float]): 0-D tensor with vtln warp factor
        fbank_opts (FbankOpts): Options for Fbank extraction
        frame_opts (FrameOpts): Options for windowing
        mel_opts (MelOpts): Options for the construction of the filter bank.

    ..note:: Only single channel data is supported for now and the data must be
        scaled such that the highest value is max(int16)

    """

    assert wav_data.shape.ndims == 1
    return _kaldi_module.fbank(
        wav_data, vtln_warp,
        *fbank_opts, *frame_opts, *mel_opts
    )

DeltaOptsBase = namedtuple('DeltaOpts', ('order', 'window'))
class DeltaOpts(DeltaOptsBase):
    """
    Args:
        order (int): Order of delta computation.
        window (int): Parameter controlling window for delta computation
            (actual window size for each delta order is 1 + 2*delta-window-size)
    """
    __slots__ = ()

default_DeltaOpts = DeltaOpts(order=2, window=2)

def add_deltas(features, delta_opts=default_DeltaOpts):
    """ Compute deltas of `features` according to `delta_opts`.

    Args:
        features (Tensor): A 2D tensor with the features for each frame.
        delta_opts (DeltaOpts): Options for the delta computation.

    """

    assert features.shape.ndims == 2
    return _kaldi_module.add_deltas(features, *delta_opts)

def decode_wav(raw_data):
    """ Reads wav data from filename

    Args:
        raw_data (Tensor[string]): 0-D tensor with raw_data

    """
    return _kaldi_module.decode_wav(raw_data)


def read_word_table(words_txt):
    """ Reads the ID -> word mapping from `words.txt`

    Args:
        words_txt (str): Kaldi words.txt containing the mapping

    """
    with open(words_txt) as fid:
        return {int(line.strip().split(' ')[1]): line.split(' ')[0]
                for line in fid
                if len(line.split(' ')) == 2}
