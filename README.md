# CHiME 4 WRBN backend (Tensorflow port)

This is a minimal port of our CHiME 4 backend to Tensorflow. Originally it was trained and used with (our internal version of) Chainer. Due to the many dependencies it was not possible to publish it in it's original form. Using Tensorflow allows us to get rid of all internal dependencies with Kaldi the only dependency left.

The backend has been tested against the Chainer version and only minimal differences between the estimated likelihoods exists.

## Installation

1. Install Tensorflow
1. Compile Kaldi in shared mode (we need to link against the `.so`      objects)
    ```
    git clone https://github.com/kaldi-asr/kaldi.git kaldi
    cd kaldi
    export KALDI_ROOT=`pwd`
    cd tools; make; cd ..
    cd src
    ./configure --shared
    make
    ```
1. Set the environment variable `KALDI_ROOT` to the root dir of your (shared) Kaldi version (see above)
1. Compile the Kaldi OPs
```
cd user_ops; make; cd..
```
1. Download assets
```
chmod +x download_assets.sh; ./download_assets.sh
```
1. That's all, you're good to go! Try running
```
python decode.py test.wav
```

## Usage

A minimal usage example can be found in `decode.py`. You can just pass a `16kHz` wav audio file as argument and the script should output the 1-best decode.

## Known limitations

1. There are minor numerical differences compared to the Chainer version which might lead to different decode results
2. No batch-processing
3. The decode only outputs the 1-best paths. However, it should be trivial to support storing the search graph (pull requests are welcome!)
