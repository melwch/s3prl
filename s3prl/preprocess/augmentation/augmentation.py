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

def valid_noise(wav_fn, src_info, verbose=False):
    from utils import get_audio_info
    wav_info = get_audio_info(wav_fn, verbose)
    if verbose:
        print(wav_fn, src_info.duration, '<', wav_info.duration)
    return src_info.duration < wav_info.duration

def add_noise(command, dest_file, src_file, wav_info, info={}, 
              config:dict={ 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 }, 
              noise_source=None, python_command="python"):
    from glob import glob
    import random
    import os

    snrs = [25]
    no_id = True
    noise_id = 0
    num_noise_types = 1
    noise_wav_path = None
    num_noise = len([name for name in os.listdir(noise_source) if name.endswith(".wav") and os.path.isfile(os.path.join(noise_source, name))])
    if config is not None:
        if "snr" in config:
            snrs = config['snr']
        if "id" in config:
            if isinstance(config['id'], (list)):
                noise_id = config['id']
            elif type(config['id']) == 'int' and config['id'] >= 1 and config['id'] <= num_noise:
                no_id = False
                noise_id = config['id']
        if "ntypes" in config and no_id:
            num_noise_types = config['ntypes']
    no_long_noise = False
    while noise_wav_path is None or any([not os.path.exists(fn) or (no_id and not no_long_noise and not valid_noise(fn, wav_info)) for fn in noise_wav_path]):
        try:
            if isinstance(noise_id, (list)):
                paths = []
                for id in noise_id:
                    fn = list(glob(os.path.join(noise_source, id + '_*.wav')))
                    if len(fn) > 0:
                        if os.path.exists(fn[0]):
                            paths.append(fn[0])
                        else:
                            print(f'***ALERT: noise {fn} does not exists')
                if len(paths) > 0:
                    noise_wav_path = paths
                else:
                    no_id = True
                noise_id = None
            elif no_id: 
                wavs = []               
                for wav_fn in glob(os.path.join(noise_source, '*.wav')):
                    if valid_noise(wav_fn, wav_info):
                        wavs.append(wav_fn)
                no_long_noise = len(wavs) < num_noise_types
                if no_long_noise:
                    print(f'****ALERT: source audio length {wav_info.duration} secs > noises')
                    wavs = glob(os.path.join(noise_source, '*.wav'))
                noise_wav_path = random.sample(wavs, num_noise_types)
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
            SNR = random.sample(snrs, 1)[0]
            dir, fname = os.path.split(fn)
            if len(noise_fns) == 0:
                noise_fns.append(dir)
            noise_fns.append(f'{fname}@{SNR}')            
        
        noise_fns = ",".join(noise_fns)        
        if python_command is None:
            from addnoise import apply_noise_to_speech
            command += apply_noise_to_speech(src_file, noise_wav_path)
        
        
        if python_command is not None:            
            command += f'\n{python_command} addnoise.py {config["mode"]} "{noise_fns}" "{src_file}" "{dest_file}"\n'
        info['snr'] = { 'mode': config['mode'], 'fns': noise_fns }
    
    return command

def add_rir(command, input_format, dest_fn, src_file, wav_info, info=None, rir_source="reverdb", 
            config:dict={'mode': 'RAW', 'id': 0, 'mixing_level': -1, 'gain': 4}, 
            verbose=False):
    import os, random
    from glob import glob
    hasFFMPEG = False
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
                        files = glob(os.path.join(rir_source, f'*_{rir_type}room_*.wav'))
                        rir_wav_path = random.sample(files, 1)[0]
                    else:
                        if no_id:
                            rir_id = str(random.randint(1, num_rir))                        
                        rir_wav_path = glob(os.path.join(rir_source, f'{rir_id}_*.wav'))[0]
                else:
                    rir_wav_path = random.sample(glob(os.path.join(rir_source, '*.wav')), 1)[0]
            except:
                import traceback, sys
                print("Error loading rir", rir_wav_path) # sample again
                no_id = True
                rir_wav_path = None
                traceback.print_exception(*sys.exc_info())

        if rir_wav_path is not None:
            basis = '_'.join(os.path.basename(rir_wav_path)[:-4].split('_'))
            subcommand = ''
            if verbose:
                print(f"Using {rir_mode} mode, adding RIR {basis}{'' if src_file == '-' else ' to ' + os.path.basename(src_file)}")
            if src_file != '-': 
                src_file = f'"{src_file}"'

            temp_fn = None
            if not dest_fn.startswith('-'):
                base_dir, temp_fn = os.path.split(dest_fn)
                temp_fn = os.path.join(base_dir, f'rir_temp{os.path.splitext(temp_fn)[1]}')
                temp_fn = f'"{temp_fn}"\n'

            if rir_mode == "AMIX":
                subcommand = f"ffmpeg {'' if input_format is None else '-f ' + input_format + ' ' }-i {src_file} -i \"{rir_wav_path}\" -filter_complex '[0] [1] afir=dry={config['gain']}:wet={config['gain']} [reverb]; [0] [reverb] amix=inputs=2:weights=3 1' -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {output_format} -f {wav_info.format.lower()} {dest_fn if temp_fn is None else temp_fn}"
            elif rir_mode == "ROOM":
                output_format = "ac3"
                subcommand = f"ffmpeg {'' if input_format is None else '-f ' + input_format + ' ' }-i {src_file} -i \"{rir_wav_path}\" -filter_complex '[0] [1] afir=dry={config['gain']}:wet={config['gain']}' -c:a {output_format} -room_type {room_type} -mixing_level {mixing_level} -f {output_format} {dest_fn if temp_fn is None else temp_fn}"
            elif rir_mode == 'RAW':
                subcommand = f"ffmpeg {'' if input_format is None else '-f ' + input_format + ' ' }-i {src_file} -i \"{rir_wav_path}\" -filter_complex '[0] [1] afir=dry={config['gain']}:wet={config['gain']}' -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} {dest_fn if temp_fn is None else temp_fn}"

            if temp_fn is not None:
                subcommand += f'mv {temp_fn[:-1]} "{dest_fn}"'

            if len(subcommand) > 0:
                command += subcommand
                info["RIR"] = basis
                hasFFMPEG = True
    else:
        print("File or RIR folder not exists:", rir_source)

    #return reverb_sig
    return hasFFMPEG, command, output_format

def normalize(src_fn, dest_fn, wav_info,
              normalization_type="ebu",
              target_level=-23.0,
              loudness_range_target=7.0,
              true_peak=-2.0,
              offset=0.0,
              keep_original_audio=False,
              pre_filter=None,
              post_filter=None,
              extra_input_options=None,
              extra_output_options=None,
              output_format=None,
              return_mode='cmd',
              verbose=False):

    if return_mode == 'program':
        command = f'ffmpeg-normalize "{src_fn}" -nt {normalization_type} -t {target_level} '
        command += f"-c:a {wav_info.codec} -b:a {wav_info.brate} -ar {wav_info.samplerate} "
        if normalization_type == "ebu":
            command += f"-lrt {loudness_range_target} "
            command += f"-tp {true_peak} "
            if offset != 0:
                command += f"--offset {offset} "
        if keep_original_audio:
            command += "-koa "
        if pre_filter is not None:
            command += f"-prf {pre_filter} "
        if post_filter is not None:
            command += f"-pof {post_filter} "
        if extra_input_options is not None:
            command += f"-ei {extra_input_options} "
        if extra_output_options is not None:
            command += f"-ei {extra_output_options} "
        if output_format is not None:
            command += f"-ofmt {output_format} "
        if verbose:
            command += f"-p "
        command += f'-f -o "{dest_fn}"'
        command = [command]
    else:
        audio_codec = wav_info.codec
        audio_bitrate = wav_info.brate
        sample_rate = wav_info.samplerate
        from extools.ffmpeg_normalize.normalize import FFmpegNormalize
        ffmpeg_normalize = FFmpegNormalize(normalization_type=normalization_type,
                                           target_level=target_level,
                                           print_stats=verbose,
                                           loudness_range_target=loudness_range_target,
                                           # threshold=cli_args.threshold,
                                           true_peak=true_peak,
                                           offset=offset,
                                           dual_mono=False,
                                           audio_codec=audio_codec,
                                           audio_bitrate=audio_bitrate,
                                           sample_rate=sample_rate,
                                           keep_original_audio=keep_original_audio,
                                           pre_filter=pre_filter,
                                           post_filter=post_filter,
                                           video_codec='copy',
                                           video_disable=True,
                                           subtitle_disable=True,
                                           metadata_disable=True,
                                           chapters_disable=True,
                                           extra_input_options=extra_input_options,
                                           extra_output_options=extra_output_options,
                                           output_format=output_format,
                                           dry_run=False,
                                           return_cmd=True,
                                           debug=False,
                                           progress=False)
        ffmpeg_normalize.add_media_file(src_fn, dest_fn)
        command = ffmpeg_normalize.run_normalization()
    return command

def mix_cocktail(src_dir, dest_dir, 
                 categories, codecs_cats, info={}, 
                 scheme={ "normalize": { 'insert': [], 'type': 'ebu', 'level': -23.0, 'loudness': -2.0, 'peak': 2.0, 'offset': 0.0 },
                          "noise": { 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 },
                          "perturbation": { "mode" : "tempo", "value": 1. },
                          "rir": {'mode': 'RAW', 'id': 0, 'mixing_level': -1, 'gain': 4},
                          "distortions": [.2, .2, .2],
                          "number of codecs mixture": 1 }, 
                 reverdb_dir="reverdb",
                 noise_dir="noise",
                 python_command="python",
                 verbose=False):
    from utils import get_audio_info
    from glob import glob
    import random, os, math
    
    commands = []
    if isinstance(src_dir, str):
        src_dir = glob(f"{src_dir}/*.wav")

    data_length = len(src_dir)
    distortions = [""] * data_length
    start_index = 0
    end_index = 0
    for category, percent in zip(categories, scheme["distortions"]):
        #print(category, percent)
        end_index = start_index if percent <= 0 else math.ceil(min(data_length, data_length * percent + start_index)) + 1
        if start_index < end_index:
            distortions[start_index:end_index] = [category] * (end_index - start_index + 1)
            start_index = end_index
    if verbose:
        print("distortions", distortions)
    # run the pipeline
    for i, wav_fn in enumerate(src_dir):
        fname = os.path.basename(wav_fn)
        fname = "".join(fname.split(".")[:-1])
        wav_info = get_audio_info(wav_fn, verbose)
        path = os.path.join(dest_dir, f"{fname}.wav")
        info[fname] = {"codecs": []}

        command = ""
        hasFFMPEG = False
        pickFromFile = False
        if 'normalize' in scheme and 'insert' in scheme['normalize'] and 'V' in scheme['normalize']['insert']:
            command = normalize(wav_fn, path, wav_info, 
                                normalization_type=scheme['normalize']['type'], 
                                target_level=scheme['normalize']['level'],
                                loudness_range_target=scheme['normalize']['loudness'],
                                true_peak=scheme['normalize']['peak'],
                                offset=scheme['normalize']['offset'],
                                return_mode='cmd')
            commands.append(' '.join(command))
            wav_fn = path
            command = ""
        
        if 'noise' in scheme and 'A' in scheme['noise']['insert']:
            command += add_noise(command, path, wav_fn, wav_info, info[fname], config=scheme['noise'], noise_source=noise_dir, python_command=None)
            wav_fn = '-'
            hasFFMPEG = True

        output_format = wav_info.format.lower()
        perturbation = None
        if 'perturbation' in scheme:
            if scheme['perturbation'].get('mode', None) == 'tempo':
                print('perturbation value:', scheme['perturbation']['value'])
                speed_perturbation = min(100.0, max(0.5, scheme['perturbation'].get('value', 1.0)))
                if speed_perturbation != 1.0:
                    #if verbose:
                    print("Apply tempo shift:", speed_perturbation)
                    info[fname]['TEMPO_SHIFT'] = speed_perturbation
                    
                    if 'pitch' not in scheme and ('B' in scheme['noise']['insert'] or 'C' in scheme['noise']['insert']):
                        command += f"ffmpeg -y -i \"{wav_fn}\" -af \"atempo={speed_perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} \"{path}\"\n"
                        wav_fn = path
                    else:
                        command += f"ffmpeg -y -i \"{wav_fn}\" -af \"atempo={speed_perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} - | "
                        wav_fn = "-"
                    hasFFMPEG = True
            elif scheme['perturbation'].get('mode', None) == 'pitch':
                pitch_perturbation = int(wav_info.samplerate * float(max(8000.0 / wav_info.samplerate, scheme['perturbation'].get('value', -1))))
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
                    info[fname]['PITCH_SHIFT'] = perturbation
                    if 'B' in scheme['noise']['insert']:
                        command += f"ffmpeg -y -i \"{wav_fn}\" -af \"{perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} \"{path}\"\n"
                        wav_fn = path
                    else:
                        command += f"ffmpeg -y -i \"{wav_fn}\" -af \"{perturbation}\" -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {output_format} - | "
                        wav_fn = "-"
                    hasFFMPEG = True
        if 'noise' in scheme and 'B' in scheme['noise']['insert']:
            command = add_noise(command, path, wav_fn, wav_info, info[fname], config=scheme['noise'], noise_source=noise_dir, python_command=python_command)
            wav_fn = path
            pickFromFile = True

        if 'rir' in scheme and scheme['rir'] is not None:
            _hasFFMPEG, command, output_format = add_rir(command, output_format,
                                                         path if 'C' in scheme['noise']['insert'] else '- | ', wav_fn, wav_info,
                                                         info=info[fname], 
                                                         config=scheme['rir'], 
                                                         rir_source=reverdb_dir,
                                                         verbose=verbose)
            hasFFMPEG = hasFFMPEG or _hasFFMPEG
            if _hasFFMPEG:
                pickFromFile = False

        if 'noise' in scheme and 'C' in scheme['noise']['insert']:
            command = add_noise(command, path, wav_fn, wav_info, info[fname], config=scheme['noise'], noise_source=noise_dir, python_command=python_command)
            wav_fn = path
            pickFromFile = True

        saved_to_file = False
        if distortions[i] in codecs_cats:
            subcommand = ""
            codecs = codecs_cats[distortions[i]]
            num_codecs = len(codecs)        
            if num_codecs > 0:
                sampled_codecs = codecs.copy()            
                num_mixed_codecs = max(1, min(num_codecs, scheme['number of codecs mixture']))
                for codec in random.sample(sampled_codecs, num_mixed_codecs):
                    if not codec.startswith("#"): # not commented out
                        codec = codec.split(",")
                        # Pipe commands to follow Kaldi style
                        if len(codec) == 4:
                            info[fname]["codecs"].append(codec)
                            info[fname]["codecs_type"] = distortions[i]
                            input_fn = f'\"{wav_fn}\"' if pickFromFile or (len(command) == 0 and len(subcommand) == 0) else '-'
                            if distortions[i] == "[HIGH DISTORTION CODECS]":
                                subcommand += f'ffmpeg -y -f {output_format} -i {input_fn} -filter:a "volume=0.8" -c:a {codec[1]} -b:a {codec[2]} -ac 1 -ar {codec[3]} -f {codec[0]} - | '
                            else:
                                subcommand += f'ffmpeg -y -f {output_format} -i {input_fn} -c:a {codec[1]} -b:a {codec[2]} -ac 1 -ar {codec[3]} -f {codec[0]} - | '
                            output_format = codec[0]
                            hasFFMPEG = True
                            pickFromFile = False
                        else:
                            print("Unknown codec specification:", ",".join(codec))
            
            if len(subcommand) > 0:
                temp_fn = None
                if os.path.basename(input_fn) != os.path.basename(path):
                    base_dir, temp_fn = os.path.split(path)
                    temp_fn = os.path.join(base_dir, f'distortion_temp{os.path.splitext(temp_fn)[1]}')
                command += subcommand + f"ffmpeg -y -f {output_format} -i - -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {wav_info.format} \"{path if temp_fn is None else temp_fn}\""
                commands.append(command)
                if temp_fn is not None:
                    commands.append(f'mv "{temp_fn}" "{path}"')
                saved_to_file = True

        if len(command) == 0:
                dir, fname = os.path.split(path)
                format = fname.split('.')
                fname = '_'.join(format[:-1])
                format = format[-1]
                #command = f"cp \"{wav_fn}\" \"{os.path.join(dir, fname + '_original.' + format)}\""
                command = f"cp \"{wav_fn}\" \"{os.path.join(dir, fname + '.' + format)}\""
                commands.append(command)
        elif not saved_to_file:
            if hasFFMPEG:
                base_dir, temp_fn = os.path.split(path)
                temp_fn = os.path.join(base_dir, f'temp{os.path.splitext(temp_fn)[1]}')
                if wav_fn != '-':
                    wav_fn = f'"{wav_fn}"'
                command += f"ffmpeg -y {'-f ' + output_format + ' ' if len(command) > 0 else ''}-i {wav_fn if pickFromFile or len(command) == 0 else '-'} -c:a {wav_info.codec} -b:a {wav_info.bitrate} -ac {wav_info.channels} -ar {wav_info.samplerate} -f {wav_info.format} \"{temp_fn}\""
                commands.append(command)
                commands.append(f'if [ -f "{path}" ]; then\n\trm "{path}"\nfi')
                commands.append(f'mv "{temp_fn}" "{path}"')
            elif len(command) > 0:
                commands.append(command)

        if 'normalize' in scheme and 'insert' in scheme['normalize'] and 'H' in scheme['normalize']['insert']:
            command = normalize(path, path, wav_info, 
                                normalization_type=scheme['normalize']['type'], 
                                target_level=scheme['normalize']['level'],
                                loudness_range_target=scheme['normalize']['loudness'],
                                true_peak=scheme['normalize']['peak'],
                                offset=scheme['normalize']['offset'],
                                return_mode='program')
            commands.append(' '.join(command))

    if verbose:
        print(f"Generated {len(commands)} commands")

    return commands

def augment(dataset, dest_dir, 
            config, 
            scheme={ "clean": 0.5,
                     "normalize": { 'insert': [], 'type': 'ebu', 'level': -23.0, 'loudness': -2.0, 'peak': 2.0, 'offset': 0.0 },
                     "noise": { 'mode': 0, 'snr': [25, 20, 15, 5, 0], 'id': 0, 'insert': ['A'], 'ntypes': 1 },
                     "perturbation": { "mode" : "tempo", "value": 1. },
                     "rir": {'mode': 'RAW', 'id': 0, 'mixing_level': -1, 'gain': 4},
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
    end_index = int(data_length * min(1, max(0, scheme["clean"])))
    clean_set = []
    print(f"Copy {len(clean_set)} clean files")
    if end_index > 0:
        for src in dataset[:end_index]:
            fname = os.path.basename(src)
            format = fname.split('.')
            fname = '_'.join(format[:-1])
            format = format[-1]
            #clean_set.append(f"cp \"{src}\" \"{os.path.join(dest_dir, fname + '_original.' + format)}\"")
            clean_set.append(f"cp \"{src}\" \"{os.path.join(dest_dir, fname + '.' + format)}\"")
    if end_index < data_length:
        distorted_sets = []
        distorted_sets.append(mix_cocktail(dataset[end_index:], 
                                           dest_dir,
                                           categories, codecs, info,
                                           scheme={'normalize': scheme['normalize'],
                                                   'noise': scheme["noise"],
                                                   'perturbation': scheme["perturbation"],
                                                   'rir': scheme["rir"],
                                                   'distortions': scheme["distortions"],
                                                   'number of codecs mixture': scheme["number of codecs mixture"]},
                                           reverdb_dir=reverdb_dir,
                                           noise_dir=noise_dir,
                                           python_command=python_command,                                               
                                           verbose=verbose))
    else:
        print("No more files left to process")

    distorted_sets.insert(0, clean_set)

    with open(dest_info_fn, "w+") as f:
        json.dump(info, f)
    print('Saved info to', dest_info_fn)

    return distorted_sets