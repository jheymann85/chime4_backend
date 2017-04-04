#!/bin/bash
mkdir -p assets/kaldi_files/
cd assets/kaldi_files/
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/final.mdl
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/disambig_tid.int
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/final.occs
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/HCLG.fst
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/num_pdfs
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/phones.txt
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/tree
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/kaldi_files/words.txt
cd ..
mkdir -p network
cd network
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/network/best.nnet
wget https://homepages.uni-paderborn.de/jheymann/chime_4_backend/assets/network/tr_config.json
cd ../..