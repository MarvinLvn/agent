#!/bin/bash

OUTPUT=$1

if [ "$#" -ne 1 ]; then
    echo "Please provide the output folder where data will be stored."
fi

# 1) Download
mkdir -p ${OUTPUT}
for i in {0..34}; do
  wget https://openslr.elda.org/resources/139/audiocite.net_${i}.zip -P $OUTPUT;
done;

# 2) Unzip
for zip in *.zip; do
  unzip $zip;
done

# 3) Convert