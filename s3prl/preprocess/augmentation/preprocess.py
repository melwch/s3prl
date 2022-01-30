# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess.py ]
#   Synopsis     [ Data Augmentation Pipeline ]
#   Source       [ https://github.com/melwch/s3prl/tree/master/s3prl/preprocess/augmentation ]
#   Author       [ Wong Chee Hoong, Melvin ]
#   Copyright    [ Copyright(c), Nanyang Technological University ]
"""*********************************************************************************************"""

def runner(commands):
    import subprocess

    print(f'Process {os.getpid()} has {len(commands)} commands')
    for i, command in enumerate(commands):
        print(f'Process {os.getpid()}: running command ({i+1}/{len(commands)}) "{command}"')
        p = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,  # Apply stdin isolation by creating separate pipe.
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
            shell=True
        )

        # simple running of command
        stdout, stderr = p.communicate()

        stdout = stdout.decode("utf8", errors="replace")
        stderr = stderr.decode("utf8", errors="replace")

        print(f'Process {os.getpid()}: {str(stdout)}')

        if p.returncode == 0:
            print(f'Process {os.getpid()}: Command ({i+1}/{len(commands)}) successfully executed "{command}"')
        else:
            print(f"Process {os.getpid()}: RUNNER ERROR: \"{command}\"\n{str(stderr)}")

    print(f'Process {os.getpid()} completed {len(commands)} commands!')

'''
Generate a bash script that runs the FFMPEG commands to process the configured audio augmentation pipeline
'''
if __name__ == "__main__":
    import os, shutil
    import torch
    import argparse
    from glob import glob
    from augmentation import augment
    from utils import download_and_extract_musan, download_and_extract_BUT_Speech, download_and_extract_RIRS_NOISES

    SEED = torch.initial_seed() % 2**32

    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source wave files")
    parser.add_argument("dest", help="destination wave folder")
    parser.add_argument("config", help="configuration file")
    parser.add_argument("-p", "--num-procs", default=3, type=int, help="set number of parallel threads", required=False)
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="show debug info", required=False)
    parser.add_argument("-s", "--random-state", default=SEED, type=int, help="set random state", required=False)
    args = parser.parse_args()

    import random
    import torch
    import numpy as np
    print('RANDOM STATE:', args.random_state)
    torch.manual_seed(args.random_state)
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    python_command = "python"
    reverdb_dir = "reverdb"
    noise_dir = "noise"
    distortion_config = "distortion_codecs.conf"
    scheme = { "clean": 0.5,
               "channel": 0,
               "target_sample_rate": "8000",
               "normalize": { 'insert': [], 'type': 'ebu', 'level': -23.0, 'loudness': -2.0, 'peak': 2.0, 'offset': 0.0 },
               "noise": { 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 },
               "perturbation": { "mode" : "tempo", "value": 1. },
               "rir": {'mode': 'RAW', 'id': 0, 'mixing_level': -1, 'gain': 4},
               "distortions": [.2, .2, .2],
               "number of codecs mixture": 1 }
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            lines = f.readlines()
            for line in lines:
                #if args.verbose:
                #    print(line)
                line = line.strip()
                if not line.startswith('#') and len(line) > 0:
                    [key, value] = line.split("=")
                    if key == "PYTHON_COMMAND":
                        python_command = value
                        assert len(value) > 0, "Please provide the python command to use for generating commands in the script"
                    elif key == "CHANNEL":
                        scheme['channel'] = int(value)
                    elif key == "TARGET_SAMPLE_RATE":
                        scheme['target_sample_rate'] = value
                    elif key == "NORMALIZE":
                        scheme['normalize']['insert'] = value.split(',')
                    elif key == "NORMALIZE_TYPE":
                        scheme['normalize']['type'] = value
                        assert any((scheme['normalize']['type'] == 'ebu', scheme['normalize']['type'] == 'rms', scheme['normalize']['type'] == 'peak')), "Please provide either ebu OR rms OR peak for NORMALIZE_TYPE"
                    elif key == "NORMALIZE_TARGET_LEVEL":
                        scheme['normalize']['level'] = float(value)
                        assert any((scheme['normalize']['type'] == 'ebu' and scheme['normalize']['level'] <= -5.0 and scheme['normalize']['level'] >= -70.0,
                                    scheme['normalize']['type'] != 'ebu' and scheme['normalize']['level'] <= 0.0 and scheme['normalize']['level'] >= -99.0)), "Please provide target value between -99.0 and 0.0"
                    elif key == "NORMALIZE_EBU_LOUDNESS_RANGE_TARGET":
                        scheme['normalize']['loudness'] = float(value)
                        assert scheme['normalize']['type'] != 'ebu' or (scheme['normalize']['loudness'] <= 20.0 and scheme['normalize']['loudness'] >= 1.0), "Please provide target value between 20.0 and 1.0"
                    elif key == "NORMALIZE_EBU_TRUE_PEAK":
                        scheme['normalize']['peak'] = float(value)
                        assert scheme['normalize']['type'] != 'ebu' or (scheme['normalize']['peak'] <= 0.0 and scheme['normalize']['peak'] >= -9.0), "Please provide NORMALIZE_EBU_TRUE_PEAK value between -9.0 and 0.0"
                    elif key == "NORMALIZE_EBU_OFFSET":
                        scheme['normalize']['offset'] = float(value)
                        assert scheme['normalize']['type'] != 'ebu' or (scheme['normalize']['offset'] <= 99.0 and scheme['normalize']['offset'] >= -99.0), "Please provide NORMALIZE_EBU_OFFSET value between -99.0 and 99.0"
                    elif key == "SIGNAL_NOISE_RATIO":
                        scheme['noise']['snr'] = [float(v) for v in value.split(',')]
                    elif key == "INSERT_SNR":
                        scheme['noise']['insert'] = value.split(',')
                    elif key == "NOISE_ID":
                        scheme['noise']['id'] = value.split(',') if ',' in value else int(value)
                    elif key == "NUM_NOISE_TYPES":
                        scheme['noise']['ntypes'] = [int(v) for v in (value.split(',') if ',' in value else [value])]
                        assert isinstance(scheme['noise']['id'], (list)) or all([v >= 0 for v in scheme['noise']['ntypes']]), "Please provide number of types greater than 0"                    
                    elif key == "NOISE_DIR":
                        noise_dir = value
                        #assert os.path.exists(noise_dir), "Please provide NOISE_DIR with valid folder location"
                    elif key == "NOISE_APPLICATION_MODE":
                        scheme['noise']['mode'] = int(value)
                        assert scheme['noise']['mode'] == 0 or scheme['noise']['mode'] == 1, "Please provide NOISE_APPLICATION_MODE 0 or 1" 
                    elif key == "TEMPO_SHIFT" and scheme['perturbation']['mode'] == "tempo":
                        scheme['perturbation']['value'] = [float(v) for v in value.split(',')] if ',' in value else [float(value)]
                        #assert scheme['perturbation']['value'] >= 0.5 and scheme['perturbation']['value'] <= 100.0, "Please provide SPEED_PERTURBATION within the range 0.5 and 100.0"
                    elif key == "PITCH_SHIFT" and scheme['perturbation']['mode'] == "pitch":
                        scheme['perturbation']['value'] = [float(v) for v in value.split(',')] if ',' in value else [float(value)]
                    elif key == "PERTURBATION_MODE":
                        if value == "TEMPO_SHIFT":
                            scheme['perturbation']['mode'] = "tempo"
                        elif value == "PITCH_SHIFT":
                            scheme['perturbation']['mode'] = "pitch"
                        else:
                            scheme['perturbation']['mode'] = value
                    elif key == "REVERB_MODE":
                        scheme['rir']['mode'] = value
                    elif key == "REVERB_DRY_WET_GAIN":
                        scheme['rir']['gain'] = int(value)
                    elif key == "REVERB_MIXING_LEVEL":
                        scheme['rir']['mixing_level'] = int(value)
                        assert not (scheme['rir']['mode'] == 'small room' or \
                                    scheme['rir']['mode'] == 'large room') or \
                                    (scheme['rir']['mixing_level'] >= 80 and \
                                     scheme['rir']['mixing_level'] <= 111), \
                                     "Please provide REVERB_MIXING_LEVEL within the range 80 and 111"
                    elif key == "REVERB_ID":
                        scheme['rir']['id'] = value
                    elif key == "CLEAN_SET":
                        scheme["clean"] = min(1, max(0, float(value)))
                    elif key == "MINOR_DISTORTION_PERCENTAGE":
                        scheme["distortions"][0] = float(value)
                        assert scheme["distortions"][0] >= 0.0 and scheme["distortions"][0] <= 1.0, "Please provide MINOR_DISTORTION_PERCENTAGE within the range 0.0 and 1.0"
                    elif key == "MEDIUM_DISTORTION_PERCENTAGE":
                        scheme["distortions"][1] = float(value)
                        assert scheme["distortions"][1] >= 0.0 and scheme["distortions"][1] <= 1.0, "Please provide MEDIUM_DISTORTION_PERCENTAGE within the range 0.0 and 1.0"
                    elif key == "HIGH_DISTORTION_PERCENTAGE":
                        scheme["distortions"][2] = float(value)
                        assert scheme["distortions"][2] >= 0.0 and scheme["distortions"][2] <= 1.0, "Please provide HIGH_DISTORTION_PERCENTAGE within the range 0.0 and 1.0"                    
                    elif key == "MIXED_CODECS_PER_CATEGORY":
                        scheme["number of codecs mixture"] = int(value)
                        assert scheme["number of codecs mixture"] >= 1, "Please provide MIXED_CODECS_PER_CATEGORY between 1 and max number of codecs per distortion category"
                    elif key == "DISTORTION_CONFIG_FILE":
                        distortion_config = value
                        assert os.path.exists(distortion_config), "Please provide DISTORTION_CONFIG_FILE with valid configuration file"
                    elif key == "REVERDB_DIR":
                        reverdb_dir = value
                        #assert os.path.exists(reverdb_dir), "Please provide REVERDB_DIR with valid folder location"
    else:
        print("Configuration file does not exists", args.config)
        parser.print_help()
        exit(1)
    
    if os.path.exists(args.src):
        src = os.path.abspath(args.src)
    else:
        print("Source file does not exists", args.src)
        parser.print_help()
        exit(1)

    if os.path.exists(args.dest):
        dest = os.path.abspath(args.dest)
    else:
        print("Destination folder does not exists", args.dest)
        parser.print_help()
        exit(1)

    if not os.path.exists(noise_dir) or len(glob(os.path.join(noise_dir, '*.wav'))) == 0:
        print("Required noise files not available, downloading...")
        download_and_extract_musan(noise_dir)

    if not os.path.exists(reverdb_dir) or len(glob(os.path.join(reverdb_dir, '*.wav'))) == 0:
        print("Required BUT_ReverbDB not available, downloading...")
        download_and_extract_BUT_Speech(reverdb_dir)

    if not os.path.exists(reverdb_dir) or len(glob(os.path.join(reverdb_dir, '*_*room_Room*-*.wav'))) == 0:
        print("Required RIRs noises not available, downloading...")
        download_and_extract_RIRS_NOISES(reverdb_dir)

    #(minor distortion percentage, 
    # medium distortion percentage, 
    # highly distortion percentage,
    # speed perturbation from 0.5 to 100.0
    # number of additive codecs per recording)
    scheme_dir = ""
    chars_limit=3
    value_limit=4
    for k, v in scheme.items():
        k = k[:chars_limit]
        if isinstance(v, (dict)):
            scheme_dir += ("_" if len(scheme_dir) > 0 else "") + \
                           "_".join(['{}_{}'.format(k, "_".join([str(_v).replace("/", "-").replace("\\", "-")[:value_limit] for _v in v])) if isinstance(v, (list)) else '{}_{}'.format(k[:chars_limit], str(v).replace("/", "-").replace("\\", "-")[:value_limit]) for k, v in v.items()])
        elif isinstance(v, (list)):
            scheme_dir += ("_" if len(scheme_dir) > 0 else "") + "_".join([str(_v).replace("/", "-").replace("\\", "-")[:value_limit] for _v in v])
        else: 
            scheme_dir += ("_" if len(scheme_dir) > 0 else "") + str(k).replace(" ", "-") + "_" + str(v).replace("/", "-").replace("\\", "-")[:value_limit]

    print("Stage 0: Clean and create directories")

    dest_dir = os.path.join(dest, scheme_dir)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=False, onerror=None)
    os.makedirs(dest_dir)

    print("Stage 1: Preprocessing")
    
    src_wavs = glob(os.path.join(src, "*.wav"))
    sets = augment(src_wavs, dest_dir, 
                   distortion_config, 
                   scheme=scheme, 
                   reverdb_dir=reverdb_dir,
                   noise_dir=noise_dir,
                   python_command=python_command,
                   info_fn="cocktail.json",
                   random_state=args.random_state,
                   verbose=args.verbose)

    print("Stage 2: Augmenting")
    
    batches = {}
    sets = [batch for batches in sets for batch in batches]
    for set_id in range(0, len(sets), args.num_procs):
        for cpu_id in range(args.num_procs):
            if set_id + cpu_id >= len(sets):
                break
            
            if cpu_id not in batches:
                batches[cpu_id] = []

            batches[cpu_id].extend(sets[set_id + cpu_id])
        #break

    from multiprocessing import Process

    processes = []
    for cpu_id, commands in batches.items():
        print(cpu_id, len(commands))
        p = Process(target=runner, args=(commands,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    print('Data augmentation completed!')

    #with open(f"make_distorted_wavs.sh", "w+") as f:
    #    f.write("#!/bin/bash -x\n")
        
    #    f.write(f"if [ -f \"{dest_dir}\" ]; then\n")
    #    f.write(f"\trm -rf \"{dest_dir}\"\nfi\n")
    #    f.write(f"mkdir -p \"{dest_dir}\"\n")

    #    f.write("\n\n")
    #    for commands in sets:
    #        for command in commands:
    #            f.write(f"{command}\n")

    #    f.write("\n\n")
    #    f.write("echo \"Completed!\"\n")

    print('\n\nCheck out "cocktail.json" for the actual augmentation methods and settings applied on each source audio file\n\n')