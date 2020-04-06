#!/bin/bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
mkdir build && cd build && cmake ..
make && make install
#cd .. && pip install .