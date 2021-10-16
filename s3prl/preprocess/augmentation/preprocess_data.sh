clear
#DATA_DIR="/media/m/F439-FA9D/workshop/callhome/wav"
#DATA_DIR="/mnt/e/workshop/callhome/callhome_data/data/wav"
#DEST_DIR="/media/m/F439-FA9D/workshop/callhome/adapt"
#DATA_DIR="../../../../data_augmentation/sample"
DATA_DIR="../../../../data_augmentation/wav/target"
DEST_DIR="../../../../data_augmentation/wav"

if [ -f $DEST_DIR ]; then
    rm -rf $DEST_DIR
fi
mkdir -p $DEST_DIR

script_file="make_distorted_wavs.sh"
if [ -f $script_file ]; then
    rm $script_file
fi

log_file="cocktail.json"
if [ -f $log_file ]; then
    rm $log_file
fi

#echo "python3 preprocess.py "$DATA_DIR/wav" "$DATA_DIR/adapt" "config.conf" --verbose"
python3 preprocess.py --verbose "$DATA_DIR" "$DEST_DIR" "config/config.conf"
chmod +x make_distorted_wavs.sh
./make_distorted_wavs.sh
#rm make_distorted_wavs.sh
echo $'\n\nCheck out "cocktail.json" for the actual augmentation methods and settings applied on each source audio file\n\n'