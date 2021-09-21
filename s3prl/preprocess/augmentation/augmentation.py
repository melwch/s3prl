def mix_rir_package(destination: str, source: str):
    import os, shutil

    ReverDB_root = source
    Room_name = ['Hotel_SkalskyDvur_ConferenceRoom2', 
                 'Hotel_SkalskyDvur_Room112', 
                 'VUT_FIT_E112',
                 'VUT_FIT_L207',
                 'VUT_FIT_L212', 
                 'VUT_FIT_L227', 
                 'VUT_FIT_Q301', 
                 'VUT_FIT_C236', 
                 'VUT_FIT_D105']

    step = 1
    for i in range(9):
        print("Moving", Room_name[i])
        speaker_name = os.listdir(os.path.join(ReverDB_root, Room_name[i], 'MicID01'))

        for j in range(len(speaker_name)):
            position_name = []
            for lists in os.listdir(os.path.join(ReverDB_root, Room_name[i], 'MicID01', speaker_name[j])):
                sub_path = os.path.join(ReverDB_root, Room_name[i], 'MicID01', speaker_name[j], lists)

                if os.path.isdir(sub_path):
                    position_name.append(sub_path)

            for k in range(len(position_name)):
                selected_rir_path = os.path.join(position_name[k], 'RIR')
                rir_wav_path = os.path.join(selected_rir_path, os.listdir(selected_rir_path)[0])
                basis = rir_wav_path[len(source):].replace("/", "_").replace("\\", "_").split(".")[0]
                dest_path = os.path.join(destination, str(step) + "_" + basis + '.wav')
                print(f"Copy from {rir_wav_path} to {dest_path}")
                shutil.copyfile(rir_wav_path, dest_path)
                step = step + 1

def add_noise(command, src_file, dest_file, info={}, 
              config:dict={ 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 }, 
              noise_source=None, python_command="python"):
    from glob import glob
    import random
    import os

    snr = [25]
    no_id = True
    noise_id = 0
    num_noise_types = 1
    noise_wav_path = None
    num_noise = len([name for name in os.listdir(noise_source) if name.endswith(".wav") and os.path.isfile(os.path.join(noise_source, name))])
    if config is not None:
        if "snr" in config:
            snr = config['snr']
        if "id" in config: 
            if type(config['id']) == 'int' and config['id'] >= 1 and config['id'] <= num_noise:
                no_id = False
                noise_id = config['id']
        if "ntypes" in config and no_id:
            num_noise_types = config['ntypes']
    snr = ",".join([str(s) for s in snr])
    while noise_wav_path is None or any([not os.path.exists(fn) for fn in noise_wav_path]):
        try:
            if no_id:
                noise_wav_path = random.sample(glob(os.path.join(noise_source, '*.wav')), num_noise_types)
            else:
                noise_wav_path = glob(os.path.join(noise_source, noise_id + '_*.wav'))                
        except:
            print("Error loading noise", noise_wav_path) # sample again
            no_id = True
            noise_wav_path = None

    if noise_wav_path is None or any([not os.path.exists(fn) for fn in noise_wav_path]):
        raise FileExistsError("Noise file not found:", noise_wav_path)
    else:
        noise_fns = []
        for fn in noise_wav_path:
            dir, fname = os.path.split(fn)
            if len(noise_fns) == 0:
                noise_fns.append(dir)
            noise_fns.append(fname)
        info['snr'] = snr
        info['snr'] = noise_fns
        command += f'\n{python_command} addnoise.py -s {snr} {config["mode"]} "{",".join(noise_fns)}" "{src_file}" "{dest_file}"\n'
        
    return command

def add_rir(command, input_format, src_file, wav_info, info=None, rir_source="reverdb", 
            config:dict={'mode': 'RAW', 'id': 0, 'mixing_level': -1}, 
            verbose=False):
    import os, random
    from utils import glob_re
    from glob import glob
    
    output_format = wav_info.format.lower()
    if os.path.exists(rir_source):
        if verbose:
            print("RIR source:", rir_source)

        no_id = True
        rir_id = 0
        room_type = 0
        mixing_level = -1
        rir_mode = None
        rir_type = None
        rir_wav_path = None
        num_rir = len([name for name in os.listdir(rir_source) if name.endswith(".wav") and os.path.isfile(os.path.join(rir_source, name))])
        if config is not None:
            if "mode" in config:
                if config['mode'] == 'large room':
                    rir_mode = "ROOM"
                    room_type = 1
                elif config['mode'] == 'small room':
                    rir_mode = "ROOM"
                    room_type = 2
                else:
                    rir_mode = config['mode']
            if "mixing_level" in config and config['mixing_level'] >= 80 and config['mixing_level'] <= 111:
                mixing_level = config['mixing_level']
            if "id" in config:                
                if isinstance(config['id'], (int)) and config['id'] >= 1 and config['id'] <= num_rir:
                    no_id = False
                    rir_id = config['id']
                elif isinstance(config['id'], (str)) and config['id'] == "small" or config['id'] == "middle" or config['id'] == "large":
                    rir_type = config['id']

        while rir_wav_path is None or not os.path.exists(rir_wav_path):
            try:
                if rir_wav_path is None:
                    if rir_type is not None:
                        room_type = { 'small': ['Hotel_SkalskyDvur_Room112', 'VUT_FIT_L207', 'VUT_FIT_L227'],
                                      'medium': ['VUT_FIT_L212', 'VUT_FIT_C236'],
                                      'large': ['Hotel_SkalskyDvur_ConferenceRoom2', 'VUT_FIT_E112', 'VUT_FIT_Q301', 'VUT_FIT_D105'], }
                        fn = os.path.join(rir_source, r".*_("+'|'.join(room_type[rir_type])+r")_[^.]+\.wav")
                        #print(fn)
                        files = glob_re(fn)
                        rir_wav_path = random.sample(files, 1)[0]
                    else:
                        if no_id:
                            rir_id = str(random.randint(1, num_rir))                        
                        rir_wav_path = glob(os.path.join(rir_source, rir_id + '_*.wav'))[0]
                else:
                    rir_wav_path = random.sample(glob(os.path.join(rir_source, '*.wav')), 1)[0]
            except:
                print("Error loading rir", rir_wav_path) # sample again
                no_id = True
                rir_wav_path = None

        if rir_wav_path is not None:
            basis = '_'.join(os.path.basename(rir_wav_path)[:-4].split('_')[1:])
            if verbose:
                print(f"Using {rir_mode} mode, adding RIR {basis}{'' if src_file == '-' else ' to ' + os.path.basename(src_file)}")
            if rir_mode == "AMIX":
                command += f"ffmpeg {'' if input_format is None else '-f ' + input_format + ' ' }-i {src_file} -i \"{rir_wav_path}\" -filter_complex '[0] [1] afir=dry=10:wet=10 [reverb]; [0] [reverb] amix=inputs=2:weights=3 1' -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {output_format} -f {wav_info.format.lower()} - | "
            elif rir_mode == "ROOM":
                output_format = "ac3"
                command += f"ffmpeg {'' if input_format is None else '-f ' + input_format + ' ' }-i {src_file} -i \"{rir_wav_path}\" -filter_complex '[0] [1] afir=dry=10:wet=10' -c:a {output_format} -room_type {room_type} -mixing_level {mixing_level} -f {output_format} - | "
            else:
                command += f"ffmpeg {'' if input_format is None else '-f ' + input_format + ' ' }-i {src_file} -i \"{rir_wav_path}\" -filter_complex '[0] [1] afir=dry=10:wet=10' -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} - | "
            info["RIR"] = basis
    else:
        print("File or RIR folder not exists:", rir_source)

    #return reverb_sig
    return command, output_format

def mix_cocktail(src_dir, dest_dir, 
                 codecs, info={}, 
                 scheme={ "noise": { 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 },
                          "speed": 1.,
                          'pitch': 1.,
                          "rir": {'mode': 'RAW', 'id': 0, 'mixing_level': -1},
                          "number of codecs mixture": 1 }, 
                 reverdb_dir="reverdb",
                 noise_dir="noise",
                 python_command="python",
                 verbose=False):
    from utils import get_audio_info
    from glob import glob
    import random, os
    
    commands = []
    if isinstance(src_dir, str):
        src_dir = glob(f"{src_dir}/*.wav")

    for wav_fn in src_dir:
        fname = os.path.basename(wav_fn)
        fname = "".join(fname.split(".")[:-1])
        wav_info = get_audio_info(wav_fn, verbose)
        path = os.path.join(dest_dir, f"{fname}.wav")
        info[fname] = {"codecs": []}

        command = ""
        if 'A' in scheme['noise']['insert']:
            command = add_noise(command, wav_fn, path, info[fname], config=scheme['noise'], noise_source=noise_dir, python_command=python_command)
            wav_fn = path

        output_format = wav_info.format.lower()
        perturbation = None
        if 'speed' in scheme:
            speed_perturbation = min(100.0, max(0.5, scheme['speed']))
            if speed_perturbation != 1.0:
                #if verbose:
                print("Apply speed perturbation:", speed_perturbation)
                info[fname]['speed_perturbation'] = speed_perturbation
                
                if 'pitch' not in scheme and 'B' in scheme['noise']['insert']:
                    command += f"ffmpeg -i \"{wav_fn}\" -af \"atempo={speed_perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} \"{path}\"\n"
                    wav_fn = path
                else:
                    command += f"ffmpeg -i \"{wav_fn}\" -af \"atempo={speed_perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} - | "
                    wav_fn = "-"

        if 'pitch' in scheme and scheme['pitch'] != 1.0:
            pitch_perturbation = int(wav_info.samplerate * float(max(8000.0 / wav_info.samplerate, scheme['pitch'])))
            if pitch_perturbation != wav_info.samplerate:
                scheme['pitch'] = pitch_perturbation / wav_info.samplerate
                tempo_times = ((pitch_perturbation / wav_info.samplerate) % 2) - 1
                tempo = min(100.0, max(0.5, 1/scheme['pitch']))
                perturbation = f"atempo={tempo}"
                if tempo_times > 1:
                    perturbation += (f",atempo={tempo}" * tempo_times)

                #if verbose:
                print("Apply pitch perturbation:", pitch_perturbation)
                perturbation = f"asetrate={pitch_perturbation},{perturbation}"
                info[fname]['pitch_perturbation'] = perturbation
                if 'B' in scheme['noise']['insert']:
                    command += f"ffmpeg -i \"{wav_fn}\" -af \"{perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} \"{path}\"\n"
                    wav_fn = path
                else:
                    command += f"ffmpeg -i \"{wav_fn}\" -af \"{perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} - | "
                    wav_fn = "-"

        if 'B' in scheme['noise']['insert']:
            command = add_noise(command, wav_fn, path, info[fname], config=scheme['noise'], noise_source=noise_dir, python_command=python_command)
            wav_fn = path

        if scheme['rir'] is not None:
            command, output_format = add_rir(command, output_format, 
                                             wav_fn, wav_info,
                                             info=info[fname], 
                                             config=scheme['rir'], 
                                             rir_source=reverdb_dir, 
                                             verbose=verbose)

        num_codecs = len(codecs)
        
        if num_codecs > 0:
            sampled_codecs = codecs.copy()            
            mixed_codecs_left = max(1, min(num_codecs, scheme['number of codecs mixture']))
            while mixed_codecs_left > 0 and len(sampled_codecs) > 0:
                codec = random.sample(sampled_codecs, 1)[0]
                sampled_codecs.remove(codec)
                if not codec.startswith("#"): # not commented out
                    codec = codec.split(",")
                    # Pipe commands to follow Kaldi style
                    if len(codec) == 4:
                        info[fname]["codecs"].append(codec)
                        input_fn = '\"{wav_fn}\"' if len(command) == 0 else '-'
                        command += f"ffmpeg -f {output_format} -i {input_fn} -c:a {codec[1]} -b:a {codec[2]} -ac 1 -ar {codec[3]} -f {codec[0]} - | "
                        output_format = codec[0]
                        mixed_codecs_left -= 1
                    else:
                        print("Unknown codec specification:", ",".join(codec))

            if len(command) == 0:
                command = f"cp \"{wav_fn}\" \"{path}\""
            else:
                command += f"ffmpeg -y -f {output_format} -i - -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {wav_info.format} \"{path}\""
            commands.append(command)
        else:
            command += f"ffmpeg -y {'-f ' + output_format + ' ' if len(command) > 0 else ''}-i {wav_fn if len(command) == 0 else '-'} -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {wav_info.format} \"{path}\""
            commands.append(command)

    if verbose:
        print(f"Generated {len(commands)} commands")
    return commands

def augment(dataset, dest_dir, 
            config, 
            scheme={ "noise": { 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 },
                     "speed": 1.,
                     "pitch": 1.,
                     "rir": {'mode': 'RAW', 'id': 0, 'mixing_level': -1},
                     "distortions": [.2, .2, .2],
                     "number of codecs mixture": 1 }, 
            reverdb_dir="reverdb",
            python_command="python",
            noise_dir="noise",
            info_fn="cocktail.json",
            verbose=False):
    import os, random
    import json

    from pathlib import Path
    parent_dir = Path(dest_dir).parents[1]
    dest_info_fn = os.path.join(parent_dir, info_fn)

    info = {}
    with open(dest_info_fn, "w+") as f:
        json.dump(info, f)

    categories = ["[MINOR DISTORTION CODECS]",
                  "[MEDIUM DISTORTION CODECS]",
                  "[HIGH DISTORTION CODECS]"]
    codecs = {category:[] for category in categories}
    codec = None
    with open(config, "r") as f:
        lines = f.readlines()
        for line in lines:
            config = line.strip()
            if len(config) > 0:
                if config in codecs:
                    codec = config
                    codecs[codec] = []
                elif codec is None:
                    print("Codec Category not found", config)
                else:
                    codecs[codec].append(config)
            config = f.readline()

    random.shuffle(dataset)
    data_length = len(dataset)
    end_index = int(data_length * (1 - sum(scheme["distortions"])))
    clean_set = dataset[:end_index]
    print(f"Copy {len(clean_set)} clean files")
    clean_set = [f"cp \"{src}\" \"{os.path.join(dest_dir, os.path.basename(src))}\"" for src in clean_set]
    distorted_sets = []
    import math
    for category, percent in zip(categories, scheme["distortions"]):
        if percent > 0:
            delta = min(data_length, end_index + int(math.ceil(data_length * percent)))
            if end_index < delta:
                distorted_sets.append(mix_cocktail(dataset[end_index:delta],
                                                dest_dir,
                                                codecs[category], info,
                                                scheme={'noise': scheme["noise"], 
                                                        'speed': scheme["speed"],
                                                        'pitch': scheme['pitch'],
                                                        'rir': scheme["rir"],
                                                        'distortions': scheme["distortions"],
                                                        'number of codecs mixture': scheme["number of codecs mixture"]},
                                                reverdb_dir=reverdb_dir,
                                                noise_dir=noise_dir,
                                                python_command=python_command,                                               
                                                verbose=verbose))
                end_index = delta
            else:
                print("No more files left")
                distorted_sets.append([])
                break
        else:
            distorted_sets.append([])

    distorted_sets.insert(0, clean_set)

    with open(dest_info_fn, "w+") as f:
        json.dump(info, f)
    print('Saved info to', dest_info_fn)

    return distorted_sets