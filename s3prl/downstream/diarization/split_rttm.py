import os
import argparse

# python3 downstream/diarization/split_rttm.py $scoring_dir/hyp_${th}_$med.rttm \
#                    ${test_set}/speaker${speaker_count}_id \
#                        $scoring_dir/${scoring_dir}/hyp_${th}_${med}_speaker${speaker_count}.rttm
def match(speaker_ids, all_speakers_rttm):
    rttms = []
    for speaker_id in speaker_ids:
            for rttm in all_speakers_rttm:
                if speaker_id in rttm:
                    rttms.append(rttm)
    return rttms

def main(args):
    speaker_ids = []
    speaker_rttm = []
    with open(args.reference_speaker_ids, mode='r', encoding='utf-8') as f:
        speaker_ids = f.readlines()

    for i, speaker_id in enumerate(speaker_ids):
        speaker_ids[i], _ = os.path.splitext(os.path.basename(speaker_id.strip()))

    all_speakers_rttm = []
    with open(args.all_speakers_rttm_file, mode='r', encoding='utf-8') as f:
        all_speakers_rttm = f.readlines()

    rttms = match(speaker_ids, all_speakers_rttm)
    if len(rttms) > 0:
        speaker_rttm.extend(rttms)

    with open(args.output_speakers_rttm_file, mode='w+', encoding='utf-8') as f:
        for rttm in speaker_rttm:
            f.write(rttm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split rttm by number of speakers")
    parser.add_argument("all_speakers_rttm_file")
    parser.add_argument("reference_speaker_ids")
    parser.add_argument("output_speakers_rttm_file")
    args = parser.parse_args()

    main(args)
