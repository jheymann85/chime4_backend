#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "thread/kaldi-task-sequence.h"
#include "decoder/faster-decoder.h"
#include "feat/feature-fbank.h"
#include "feat/wave-reader.h"

template <typename T>
kaldi::Matrix<T>* MatrixFromTensor(const tensorflow::Tensor &tensor) {
  tensorflow::TensorShape shape_ = tensor.shape();
  auto tf_tensor = tensor.tensor<T, 2>();
    kaldi::Matrix<T> *kaldi_matrix = \
        new kaldi::Matrix<T>(
            shape_.dim_size(0), shape_.dim_size(1)
        );

    for (uint32 row=0; row < shape_.dim_size(0); row++) {
        for (uint32 col=0; col < shape_.dim_size(1); col++) {
            (*kaldi_matrix)(row, col) = tf_tensor(row, col);
        }
    }
    return kaldi_matrix;
}

template <typename T>
kaldi::Vector<T>* VectorFromTensor(const tensorflow::Tensor &tensor) {
  auto tf_tensor = tensor.flat<T>();
    kaldi::Vector<T> *kaldi_vector = \
        new kaldi::Vector<T>(
            tf_tensor.dimensions()[0]
        );

    for (uint32 row=0; row < tf_tensor.dimensions()[0]; row++) {
          (*kaldi_vector)(row) = tf_tensor(row);
    }
    return kaldi_vector;
}

using namespace tensorflow;

/**
+++++++++++++++++ Decode +++++++++++++++++
**/

REGISTER_OP("Decode")
    .Attr("fst_in_filename: string")
    .Attr("model_in_filename: string")
    .Attr("acoustic_scale: float")
    .Attr("beam: int")
    .Attr("max_active: int")
    .Attr("min_active: int")
    .Attr("lattice_beam: float")
    .Attr("prune_interval: int")
    .Attr("beam_delta: float")
    .Attr("hash_ratio: float")
    .Attr("prune_scale: float")
    .Attr("allow_partial: bool")
    .Attr("lattice_ark_file: string")
    .Input("posteriors: float")
    .Input("utt_id: string")
    .Output("decode_sequence: int32")
    .Output("alignment: int32")
    .Output("num_frames: int32")
    .Output("likelihood: double")
    .Output("partial: bool");

class KaldiDecodeOp : public OpKernel {
 public:
  explicit KaldiDecodeOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
        context->GetAttr("acoustic_scale", &_acoustic_scale));
      OP_REQUIRES_OK(context,
        context->GetAttr("beam", &_beam));
      OP_REQUIRES_OK(context,
        context->GetAttr("max_active", &_max_active));
      OP_REQUIRES_OK(context,
        context->GetAttr("min_active", &_min_active));
      OP_REQUIRES_OK(context,
        context->GetAttr("lattice_beam", &_lattice_beam));
      OP_REQUIRES_OK(context,
        context->GetAttr("prune_interval", &_prune_interval));
      OP_REQUIRES_OK(context,
        context->GetAttr("beam_delta", &_beam_delta));
      OP_REQUIRES_OK(context,
        context->GetAttr("hash_ratio", &_hash_ratio));
      OP_REQUIRES_OK(context,
        context->GetAttr("prune_scale", &_prune_scale));
      OP_REQUIRES_OK(context,
        context->GetAttr("allow_partial", &_allow_partial));
      OP_REQUIRES_OK(context,
        context->GetAttr("lattice_ark_file", &_lattice_ark_file)); 

      OP_REQUIRES_OK(context,
             context->GetAttr("fst_in_filename", &_fst_in_filename));
      OP_REQUIRES_OK(context,
             context->GetAttr("model_in_filename", &_model_in_filename));

      _config.beam = _beam;
      _config.max_active = _max_active;
      _config.min_active = _min_active;
      _config.lattice_beam = _lattice_beam;
      _config.prune_interval = _prune_interval;
      _config.beam_delta = _beam_delta;
      _config.hash_ratio = _hash_ratio;
      _config.prune_scale = _prune_scale;

      
 
      if (kaldi::ClassifyRspecifier(_fst_in_filename, NULL, NULL) == \
        kaldi::kNoRspecifier) {
        // Input FST is just one FST, not a table of FSTs.
        try {
          fst::ReadFstKaldi(_fst_in_filename, &_decode_fst);
        } catch(...) {
          std::stringstream msg;
          msg << "Could not load decode fst " << _fst_in_filename;
          OP_REQUIRES(context, false, errors::Internal(msg.str()));
        }
      } else {
          OP_REQUIRES(
              context, false,
              errors::Internal("Only one FST is supported for now")
          );
      }

      try {
        kaldi::ReadKaldiObject(_model_in_filename, &_trans_model);
      } catch(...) {
        std::stringstream msg;
        msg << "Could not load transition model " << _model_in_filename;
        OP_REQUIRES(context, false, errors::Internal(msg.str()));
      }

      if (!_lattice_ark_file.empty()) {
        if (!_compact_lattice_writer.Open("ark:" + _lattice_ark_file))
        {
          KALDI_ERR << "Could not open table for writing lattices: "
                    << _lattice_ark_file;
        }
      }
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& tf_posteriors = context->input(0);
    // Copy to a Kaldi matrix
    kaldi::Matrix<kaldi::BaseFloat>* loglikes = \
      MatrixFromTensor<kaldi::BaseFloat>(tf_posteriors);
    
    kaldi::DecodableMatrixScaledMapped decodable(
      _trans_model, _acoustic_scale, loglikes
    );
    kaldi::LatticeFasterDecoder decoder(_decode_fst, _config);

    // Decode
    bool success = false;
    bool partial = false;
    std::string msg;
    if (!decoder.Decode(&decodable)) {
        OP_REQUIRES(context, false, errors::Internal("Failed to decode file"));
    }
    success = true;
    if (!decoder.ReachedFinal()) {
        if (_allow_partial) {
        LOG(WARN) << "Outputting partial output "
                    << " since no final-state reached\n";
        partial = true;
        } else {
        msg = "Not producing output " \
                    " since no final-state reached and " \
                    "--allow-partial=false.\n";
        success = false;
        }
    }
    OP_REQUIRES(context, success, errors::Internal(msg));
    
    // Get Best Path
    double likelihood;
    kaldi::LatticeWeight weight;
    std::vector<int32> alignment;
    std::vector<int32> words;
    fst::VectorFst<kaldi::LatticeArc> decoded;
    decoder.GetBestPath(&decoded);
    if (decoded.NumStates() == 0) {
        // Shouldn't really reach this point as already checked success.
        KALDI_ERR << "Failed to get traceback for utterance ";
    }
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    int32 num_frames = alignment.size();
    likelihood = -(weight.Value1() + weight.Value2());

    // Optional: write lattice
    if (!_lattice_ark_file.empty()) {
      std::string utt_ = context->input(1).flat<string>()(0);
      kaldi::Lattice *lat_ = new kaldi::Lattice;
      decoder.GetRawLattice(lat_);
      if (lat_->NumStates() == 0)
        KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt_;
      fst::Connect(lat_);
      kaldi::CompactLattice *clat_ = new kaldi::CompactLattice;
      if (!fst::DeterminizeLatticePhonePrunedWrapper(
              _trans_model,
              lat_,
              decoder.GetOptions().lattice_beam,
              clat_,
              decoder.GetOptions().det_opts))
        KALDI_WARN << "Determinization finished earlier than the beam for "
                  << "utterance " << utt_;
      delete lat_;
      lat_ = NULL;
      // We'll write the lattice without acoustic scaling.
      if (_acoustic_scale != 0.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / _acoustic_scale), clat_);
      if (clat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        _compact_lattice_writer.Write(utt_, *clat_);
      }
      delete clat_;
      clat_ = NULL;
    }

    // Create an output tensors
    Tensor* word_output = NULL;
    auto word_output_shape = TensorShape({static_cast<int>(words.size())});
    OP_REQUIRES_OK(context, context->allocate_output(0, word_output_shape,
                                                     &word_output));
    auto flat_word_output = word_output->flat<int32>();
    for(uint i=0; i < words.size(); i++) {
        flat_word_output(i) = words[i];
    }

    Tensor* alignment_output = NULL;
    auto alignment_output_shape = TensorShape(
      {static_cast<int>(alignment.size())}
    );
    OP_REQUIRES_OK(
      context, context->allocate_output(
        1, alignment_output_shape, &alignment_output
     )
    );
    auto flat_alignment_output = alignment_output->flat<int32>();
    for(uint i=0; i < alignment.size(); i++) {
        flat_alignment_output(i) = alignment[i];
    }

    Tensor* num_frames_output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}),
      &num_frames_output));
    auto flat_num_frames_output = num_frames_output->flat<int32>();
    flat_num_frames_output(0) = num_frames;

    Tensor* likelihood_output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({}),
      &likelihood_output));
    auto flat_likelihood_output = likelihood_output->flat<double>();
    flat_likelihood_output(0) = likelihood;

    Tensor* partial_flag_output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({}),
      &partial_flag_output));
    auto flat_partial_flag_output = partial_flag_output->flat<bool>();
    flat_partial_flag_output(0) = partial;

  }
 private:
  float _acoustic_scale;
  int32 _beam;
  int32 _max_active;
  int32 _min_active;
  float _lattice_beam;
  int32 _prune_interval;
  float _beam_delta;
  float _hash_ratio;
  float _prune_scale;
  bool _allow_partial;
  std::string _fst_in_filename;
  std::string _model_in_filename;
  std::string _lattice_ark_file;
  kaldi::TransitionModel _trans_model;
  fst::VectorFst<fst::StdArc> _decode_fst;
  kaldi::LatticeFasterDecoderConfig _config;
  kaldi::CompactLatticeWriter _compact_lattice_writer;
};

REGISTER_KERNEL_BUILDER(Name("Decode").Device(DEVICE_CPU), KaldiDecodeOp);

/**
+++++++++++++++++ Fbank +++++++++++++++++
**/

REGISTER_OP("Fbank")
    .Attr("use_energy: bool")
    .Attr("energy_floor: float")
    .Attr("raw_energy: bool")
    .Attr("htk_compat: bool")
    .Attr("use_log_fbank: bool")
    .Attr("use_power: bool")
    .Attr("frame_shift_ms: float")
    .Attr("frame_length_ms: float")
    .Attr("dither: float")
    .Attr("preemph_coeff: float")
    .Attr("remove_dc_offset: bool")
    .Attr("window_type: string")
    .Attr("round_to_power_of_two: bool")
    .Attr("blackman_coeff: float")
    .Attr("snip_edges: bool")
    .Attr("num_bins: int")
    .Attr("low_freq: float")
    .Attr("high_freq: float")
    .Attr("vtln_low: float")
    .Attr("vtln_high: float")
    .Input("wav_data: float")
    .Input("vtln_warp: float")
    .Output("fbank_features: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input;
      // Assert an input vector
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      // The second dimension corresponds to the number of bins/filterbanks.
      // The first dimension is the number of frames. This COULD be inferred
      // but since the length is normally undefined anyway we leave it
      // unknown.
      int num_bins;
      c->GetAttr("num_bins", &num_bins);
      auto output = c->Matrix(
        c->UnknownDim(), num_bins
      );
      c->set_output(0, output);
      return Status::OK();
    });


class KaldiFbankOp : public OpKernel {
 public:
  explicit KaldiFbankOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
        context->GetAttr("use_energy", &_use_energy));
      OP_REQUIRES_OK(context,
        context->GetAttr("energy_floor", &_energy_floor));
      OP_REQUIRES_OK(context,
        context->GetAttr("raw_energy", &_raw_energy));
      OP_REQUIRES_OK(context,
        context->GetAttr("htk_compat", &_htk_compat));
      OP_REQUIRES_OK(context,
        context->GetAttr("use_log_fbank", &_use_log_fbank));
      OP_REQUIRES_OK(context,
        context->GetAttr("use_power", &_use_power));
      OP_REQUIRES_OK(context,
        context->GetAttr("frame_shift_ms", &_frame_shift_ms));
      OP_REQUIRES_OK(context,
        context->GetAttr("frame_length_ms", &_frame_length_ms));
      OP_REQUIRES_OK(context,
        context->GetAttr("dither", &_dither));
      OP_REQUIRES_OK(context,
        context->GetAttr("preemph_coeff", &_preemph_coeff));
      OP_REQUIRES_OK(context,
        context->GetAttr("remove_dc_offset", &_remove_dc_offset));
      OP_REQUIRES_OK(context,
        context->GetAttr("window_type", &_window_type));
      OP_REQUIRES_OK(context,
        context->GetAttr("round_to_power_of_two", &_round_to_power_of_two));
      OP_REQUIRES_OK(context,
        context->GetAttr("blackman_coeff", &_blackman_coeff));
      OP_REQUIRES_OK(context,
        context->GetAttr("snip_edges", &_snip_edges));
      OP_REQUIRES_OK(context,
        context->GetAttr("num_bins", &_num_bins));
      OP_REQUIRES_OK(context,
        context->GetAttr("low_freq", &_low_freq));
      OP_REQUIRES_OK(context,
        context->GetAttr("high_freq", &_high_freq));
      OP_REQUIRES_OK(context,
        context->GetAttr("vtln_low", &_vtln_low));
      OP_REQUIRES_OK(context,
        context->GetAttr("vtln_high", &_vtln_high));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& tf_input_wav_data = context->input(0);
    const Tensor& tf_input_vtln_warp = context->input(1);
    kaldi::BaseFloat vtln_warp = tf_input_vtln_warp.flat<float>()(0);
    auto wav_data = VectorFromTensor<kaldi::BaseFloat>(tf_input_wav_data);

    kaldi::FbankOptions fbank_opts = kaldi::FbankOptions();
    kaldi::BaseFloat vtln_warp_local = vtln_warp;  // TODO: As Input

    fbank_opts.use_energy = _use_energy;
    fbank_opts.energy_floor = _energy_floor;
    fbank_opts.raw_energy = _raw_energy;
    fbank_opts.htk_compat = _htk_compat;
    fbank_opts.use_log_fbank = _use_log_fbank;
    fbank_opts.use_power = _use_power;
    fbank_opts.frame_opts.frame_shift_ms = _frame_shift_ms;
    fbank_opts.frame_opts.frame_length_ms = _frame_length_ms;
    fbank_opts.frame_opts.dither = _dither;
    fbank_opts.frame_opts.preemph_coeff = _preemph_coeff;
    fbank_opts.frame_opts.remove_dc_offset = _remove_dc_offset;
    fbank_opts.frame_opts.window_type = _window_type;
    fbank_opts.frame_opts.round_to_power_of_two = _round_to_power_of_two;
    fbank_opts.frame_opts.blackman_coeff = _blackman_coeff;
    fbank_opts.frame_opts.snip_edges = _snip_edges;
    fbank_opts.mel_opts.num_bins = _num_bins;
    fbank_opts.mel_opts.low_freq = _low_freq;
    fbank_opts.mel_opts.high_freq = _high_freq;
    fbank_opts.mel_opts.vtln_low = _vtln_low;
    fbank_opts.mel_opts.vtln_high = _vtln_high;

    kaldi::Fbank fbank(fbank_opts);

    kaldi::Matrix<kaldi::BaseFloat> features;
    try {
      fbank.Compute(*wav_data, vtln_warp_local, &features, NULL);
    } catch (...) {
      std::stringstream msg;
      msg << "Failed to compute FBank features";
      OP_REQUIRES(context, false, errors::Internal(msg.str()));
    }

    Tensor* fbank_output = NULL;
    auto fbank_output_shape = TensorShape({features.NumRows(), features.NumCols()});
    OP_REQUIRES_OK(context, context->allocate_output(0, fbank_output_shape,
                                                    &fbank_output));
    auto fbank_output_tensor = fbank_output->tensor<kaldi::BaseFloat, 2>();
    for(uint row=0; row < features.NumRows(); row++) {
      for (uint col=0; col < features.NumCols(); col++)
        fbank_output_tensor(row, col) = features(row, col);
    }

  }
private:
  bool _use_energy;
  float _energy_floor;
  bool _raw_energy;
  bool _htk_compat;
  bool _use_log_fbank;
  bool _use_power;
  float _frame_shift_ms;
  float _frame_length_ms;
  float _dither;
  float _preemph_coeff;
  bool _remove_dc_offset;
  std::string _window_type;
  bool _round_to_power_of_two;
  float _blackman_coeff;
  bool _snip_edges;
  int32 _num_bins;
  float _low_freq;
  float _high_freq;
  float _vtln_low;
  float _vtln_high;
};
REGISTER_KERNEL_BUILDER(Name("Fbank").Device(DEVICE_CPU), KaldiFbankOp);

/**
+++++++++++++++++ Add Deltas +++++++++++++++++
**/

REGISTER_OP("AddDeltas")
    .Attr("order: int")
    .Attr("window: int")
    .Input("features: float")
    .Output("deltas: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input;
      // Assert an input vector
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      
      // The first dimension (frames) will remain the same. The deltas are 
      // added to the second dimension.
      auto first_dim = c->Dim(input, 0);
      auto last_dim = c->Dim(input, 1);

      int order;
      c->GetAttr("order", &order);
      TF_RETURN_IF_ERROR(c->Multiply(last_dim, (1 + order), &last_dim));

      auto output = c->Matrix(first_dim, last_dim);
      c->set_output(0, output);
      return Status::OK();
    });;


class KaldiAddDeltasOp : public OpKernel {
 public:
  explicit KaldiAddDeltasOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
        context->GetAttr("order", &_order));
      OP_REQUIRES_OK(context,
        context->GetAttr("window", &_window));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& tf_input_features = context->input(0);
    auto feats = MatrixFromTensor<kaldi::BaseFloat>(tf_input_features);

    kaldi::DeltaFeaturesOptions opts;
    opts.order = _order;
    opts.window = _window;

    kaldi::Matrix<kaldi::BaseFloat> new_feats;
    try {
      kaldi::ComputeDeltas(opts, *feats, &new_feats);
    } catch(...) {
      OP_REQUIRES(context, false, errors::Internal(
        "Could not compute delta features."
        ));
    }

    Tensor* deltas_output = NULL;
    auto deltas_output_shape = TensorShape(
      {new_feats.NumRows(), new_feats.NumCols()}
    );
    OP_REQUIRES_OK(context, context->allocate_output(0, deltas_output_shape,
                                                    &deltas_output));
    auto deltas_output_tensor = deltas_output->tensor<kaldi::BaseFloat, 2>();
    for(uint row=0; row < new_feats.NumRows(); row++) {
      for (uint col=0; col < new_feats.NumCols(); col++)
        deltas_output_tensor(row, col) = new_feats(row, col);
    }

  }
private:
  int _order;
  int _window;
};
REGISTER_KERNEL_BUILDER(Name("AddDeltas").Device(DEVICE_CPU), KaldiAddDeltasOp);

/**
+++++++++++++++++ Read wave +++++++++++++++++
**/


REGISTER_OP("DecodeWav")
    .Input("raw_wav_data: string")
    .Output("audio_data: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // The output has the shape [channel, samples]. Both are unknown
      // without reading the file contents.
      auto output = c->Matrix(
        c->UnknownDim(), c->UnknownDim()
      );
      c->set_output(0, output);
      return Status::OK();
    });;

class KaldiDecodeWavOp : public OpKernel {
 public:
  explicit KaldiDecodeWavOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    std::stringstream raw_data;
    const Tensor& tf_raw_data = context->input(0);
    const auto tf_raw_data_flat = tf_raw_data.flat<string>();
    raw_data << *(tf_raw_data_flat.data());
    
    kaldi::WaveData wav_data;
    kaldi::Matrix<float> audio_data;
    try {
      wav_data.Read(raw_data);
      audio_data = wav_data.Data();
    } catch(...) {
      OP_REQUIRES(context, false, errors::Internal("Error decoding wav data"));
    }

    Tensor* audio_output = NULL;
    auto audio_output_shape = TensorShape(
      {audio_data.NumRows(), audio_data.NumCols()}
    );
    OP_REQUIRES_OK(context, context->allocate_output(0, audio_output_shape,
                                                    &audio_output));
    auto audio_output_tensor = audio_output->tensor<kaldi::BaseFloat, 2>();
    for(uint row=0; row < audio_data.NumRows(); row++) {
      for (uint col=0; col < audio_data.NumCols(); col++)
        audio_output_tensor(row, col) = audio_data(row, col);
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("DecodeWav").Device(DEVICE_CPU), KaldiDecodeWavOp);
