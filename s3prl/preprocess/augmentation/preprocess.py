'''
Generate a bash script that runs the FFMPEG commands to process the configured audio augmentation pipeline
'''
if __name__ == "__main__":
    import os, shutil
    import argparse
    from glob import glob
    from augmentation import augment
    from utils import download_and_extract_musan, download_and_extract_BUT_Speech, download_and_extract_RIRS_NOISES

    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source wave files")
    parser.add_argument("dest", help="destination wave folder")
    parser.add_argument("config", help="configuration file")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="show debug info", required=False)
    args = parser.parse_args()

    python_command = "python"
    reverdb_dir = "reverdb"
    noise_dir = "noise"
    distortion_config = "distortion_codecs.conf"
    scheme = { "clean": 0.5,
               "noise": { 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 },
               "speed": 1.,
               "pitch": 1.,
               "rir": {'mode': 'RAW', 'id': 0, 'mixing_level': -1},
               "distortions": [.2, .2, .2],
               "number of codecs mixture": 1 }
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            lines = f.readlines()
            for line in lines:
                #if args.verbose:
                #    print(line)
                if not line.startswith('#'):
                    [key, value] = line.strip().split("=")
                    if key == "PYTHON_COMMAND":
                        python_command = value
                        assert len(value) > 0, "Please provide the python command to use for generating commands in the script"
                    if key == "SIGNAL_NOISE_RATIO":
                        scheme['noise']['snr'] = [float(v) for v in value.split(',')]
                    elif key == "INSERT_SNR":
                        scheme['noise']['insert'] = value.split(',')
                    elif key == "NOISE_ID":
                        scheme['noise']['id'] = int(value)
                    elif key == "NUM_NOISE_TYPES":
                        scheme['noise']['ntypes'] = int(value)
                        assert scheme['noise']['id'] == 0 and scheme['noise']['ntypes'] > 0, "Please provide number of types greater than 1"                    
                    elif key == "NOISE_DIR":
                        noise_dir = value
                        #assert os.path.exists(noise_dir), "Please provide NOISE_DIR with valid folder location"
                    elif key == "NOISE_APPLICATION_MODE":
                        scheme['noise']['mode'] = int(value)
                        assert scheme['noise']['mode'] == 0 or scheme['noise']['mode'] == 1, "Please provide NOISE_APPLICATION_MODE 0 or 1" 
                    elif key == "SPEED_PERTURBATION":
                        scheme['speed'] = float(value)
                        assert scheme['speed'] >= 0.5 and scheme['speed'] <= 100.0, "Please provide SPEED_PERTURBATION within the range 0.5 and 100.0"
                    elif key == "PITCH_PERTURBATION":
                        scheme['pitch'] = float(value)
                    elif key == "REVERB_MODE":
                        scheme['rir']['mode'] = value
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
                        scheme["minor distortion"] = float(value)
                        assert scheme["minor distortion"] >= 0.0 and scheme["minor distortion"] <= 1.0, "Please provide MINOR_DISTORTION_PERCENTAGE within the range 0.0 and 1.0"
                    elif key == "MEDIUM_DISTORTION_PERCENTAGE":
                        scheme["medium distortion"] = float(value)
                        assert scheme["medium distortion"] >= 0.0 and scheme["medium distortion"] <= 1.0, "Please provide MEDIUM_DISTORTION_PERCENTAGE within the range 0.0 and 1.0"
                    elif key == "HIGH_DISTORTION_PERCENTAGE":
                        scheme["high distortion"] = float(value)
                        assert scheme["high distortion"] >= 0.0 and scheme["high distortion"] <= 1.0, "Please provide HIGH_DISTORTION_PERCENTAGE within the range 0.0 and 1.0"                    
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
    for k, v in scheme.items():
        if isinstance(v, (dict)):
            scheme_dir += ("_" if len(scheme_dir) > 0 else "") + \
                           "_".join([f'{k}_{"_".join([str(_v) for _v in v])}' if isinstance(v, (list)) else f'{k}_{str(v)}' for k, v in v.items()])
        elif isinstance(v, (list)):
            scheme_dir += ("_" if len(scheme_dir) > 0 else "") + "_".join([str(_v) for _v in v])
        else: 
            scheme_dir += ("_" if len(scheme_dir) > 0 else "") + str(k).replace(" ", "-") + "_" + str(v)

    dest_dir = os.path.join(dest, scheme_dir)
    
    src_wavs = glob(os.path.join(src, "*.wav"))
    sets = augment(src_wavs, dest_dir, 
                   distortion_config, 
                   scheme=scheme, 
                   reverdb_dir=reverdb_dir,
                   noise_dir=noise_dir,
                   python_command=python_command,
                   info_fn="cocktail.json",
                   verbose=args.verbose)

    with open(f"make_distorted_wavs.sh", "w+") as f:
        f.write("#!/bin/bash -x\n")
        
        f.write(f"if [ -f \"{dest_dir}\" ]; then\n")
        f.write(f"\trm -rf \"{dest_dir}\"\nfi\n")
        f.write(f"mkdir -p \"{dest_dir}\"\n")

        f.write("\n\n")
        for commands in sets:
            for command in commands:
                f.write(f"{command}\n")

        f.write("\n\n")
        f.write("echo \"Completed!\"")

    print("Preprocessing completed!\n\n")
    print('\n\nCheck out "cocktail.json" for the actual augmentation methods and settings applied on each source audio file\n\n')