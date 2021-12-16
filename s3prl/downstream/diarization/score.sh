#!/bin/bash
#   *********************************************************************************************"""
#   FileName     [ score.sh ]
#   Synopsis     [ Speaker Diarization Scoring, use NIST scoring metric ]
#   Source       [ Refactored From https://github.com/hitachi-speech/EEND ]
#   Author       [ Wong Chee Hoong, Melvin ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
#   *********************************************************************************************"""

set -e
set -x

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <expdir> <test_set>"
  echo "e.g., ./downstream/diarization/score.sh result/downstream/test data/test"
  exit 1
fi

scoring_dir="$1/scoring"
infer_dir="${scoring_dir}/predictions"
test_set="$2"
# directory where you cloned dscore (https://github.com/ftshijt/dscore)
dscore_dir=/home/users/u100085/dscore

frame_shift_file=$1/frame_shift
if [ -f $frame_shift_file ]; then
    frame_shift=$(cat $frame_shift_file)
else
    printf "[Warning] File not found: $frame_shift_file. Degenerate to use frame shift 160 for "`
         `"RTTM conversion. If your downstream model was not trained by the label in frame shift 160, please "`
         `"create a file $frame_shift_file with a single number: the training label frame shift. "`
         `"You can check a checkpoint's training label frame shift by: "`
         `"python3 utility/print_settings.py [ckpt] config.downstream_expert.datarc.frame_shift\nOr, "`
         `"re-inference your checkpoint with the S3PRL version newer than: " `
         `"https://github.com/s3prl/s3prl/commit/852db2e5f65fc9baea4a5877ffda6dd7470c72fc (re-training "`
         `"the model is not required). The correct $frame_shift_file will then appear in the expdir, since "`
         `"the training label frame_shift during you previous trainig was already saved in the checkpoint."
    frame_shift=160
fi
sr=16000

echo "scoring at $scoring_dir"
scoring_log_dir=$scoring_dir/log
mkdir -p $scoring_log_dir || exit 1;
find $infer_dir -iname "*.h5" > $scoring_log_dir/file_list
for med in 1 3 5 7 9 11; do
    for th in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95; do
        python3 downstream/diarization/make_rttm.py --median=$med --threshold=$th --frame_shift=${frame_shift} --subsampling=1 --sampling_rate=${sr} ${scoring_log_dir}/file_list ${scoring_dir}/hyp_${th}_${med}.rttm

        for weight_collar in 0 0.25; do
            python3 ${dscore_dir}/score.py --collar $weight_collar -r ${test_set}/rttm -s ${scoring_dir}/hyp_${th}_$med.rttm \
                > ${scoring_dir}/result_th${th}_med${med}_speakerALL_collar${weight_collar} 2>/dev/null || exit

            # NIST scoring
            # md-eval.pl \
            #     -r ${test_set}/rttm \
            #     -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0 2>/dev/null || exit

            # 2,3,4 speaker meeting
            for speaker_count in 2 3 4;
            do
                python3 downstream/diarization/split_rttm.py $scoring_dir/hyp_${th}_$med.rttm \
                    ${test_set}/speaker${speaker_count}_id \
                        ${scoring_dir}/hyp_${th}_${med}_speaker${speaker_count}.rttm

                python3 ${dscore_dir}/score.py --collar $weight_collar -R ${test_set}/speaker${speaker_count}_id -s ${scoring_dir}/hyp_${th}_${med}_speaker${speaker_count}.rttm > ${scoring_dir}/result_th${th}_med${med}_speaker${speaker_count}_collar${weight_collar}_DER_overlaps_${weight_collar}
            done
        done
    done
done

echo "BEGIN SCORES ALL SPEAKERS"
grep OVER ${scoring_dir}/result_th0.[^_]*_med[^_]*_speakerALL_collar[^_]* | sort -nrk 4
echo "END SCORES ALL SPEAKERS\n\n"

for speaker_count in 2 3 4;
do
    echo "BEGIN SCORES ${speaker_count} SPEAKERS"
    grep OVER ${scoring_dir}/result_th0.[^_]*_med[^_]*_speaker${speaker_count}_collar[^_]*_DER_overlaps_[^-]* | sort -nrk 4
    echo "END SCORES ${speaker_count} SPEAKERS\n\n"
done

