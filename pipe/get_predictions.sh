#!/bin/bash

filename=$RANDOM

while read LINE; do
   echo ${LINE}
done > $filename.input

./fastText/build/fasttext predict ../models/ft_tuned.ftz $filename.input 1 0.10 | sed 's/__label__//g' > $filename.preds

paste -d ',' $filename.input $filename.preds > $filename.output && rm $filename.input $filename.preds
cat $filename.output
rm $filename.output