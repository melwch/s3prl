#!/bin/bash +x
clear
#DATA_DIR="/media/m/F439-FA9D/workshop/callhome/callhome_data/data/wav"
#DATA_DIR="/mnt/e/workshop/callhome/callhome_data/data/wav"
#DEST_DIR="/media/m/F439-FA9D/workshop/callhome/callhome_data/data/adapt"
DATA_DIR="../../../../data_augmentation/sample"
DEST_DIR="../../../../data_augmentation/wav"

if ! [ -f $DEST_DIR ]; then
    mkdir -p $DEST_DIR
fi

script_file="make_distorted_wavs.sh"
if [ -f $script_file ]; then
    rm $script_file
fi

#echo "python3 preprocess.py "$DATA_DIR/wav" "$DATA_DIR/adapt" "config.conf" --verbose"
python3 preprocess.py --verbose "$DATA_DIR" "$DEST_DIR" "config.conf"
chmod +x make_distorted_wavs.sh
./make_distorted_wavs.sh
rm make_distorted_wavs.sh