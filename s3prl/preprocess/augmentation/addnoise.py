# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ addnoise.py ]
#   Synopsis     [ Data Augmentation Pipeline ]
#   Source       [ https://github.com/melwch/s3prl/tree/master/s3prl/preprocess/augmentation ]
#   Author       [ Wong Chee Hoong, Melvin ]
#   Copyright    [ Copyright(c), Nanyang Technological University ]
"""*********************************************************************************************"""

def read_audio(src_file, target_sr=None, target_bitrate=None):
    import math
    from pydub import AudioSegment, effects
    from pydub.utils import mediainfo

    #print('src_file', src_file)
    format = os.path.basename(src_file).split('.')[-1]
    media_info = mediainfo(src_file)
    bitrate = f"{math.floor(int(media_info['bit_rate'])/1000)}"
    audio = AudioSegment.from_file(src_file, format=format, codec=media_info['codec_name'], parameters=["-ar", media_info['sample_rate'], "-ac", "1", "-ab", bitrate])
    audio = audio.set_sample_width(2)

    audio = effects.normalize(audio)

    if target_sr is not None and audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
        print('***** frame rate changed!', src_file, target_sr)

    if target_bitrate is not None and audio.sample_width != target_bitrate:
        audio = audio.set_sample_width(target_bitrate)

    return audio, media_info

def compute_speech_noise_factor(SNR, src_audio, vad_duration, target_noise):
    if vad_duration > 0:
        import math
        import numpy as np

        speech_power = np.sum(src_audio ** 2) / vad_duration
        #speech_power = torch.linalg.vector_norm(speech_seg, ord=2) / vad_duration
        
        noise_power = np.sum(target_noise ** 2) / len(target_noise)
        #noise_power = torch.linalg.vector_norm(speech_seg, ord=2) / noise_sig.shape[0]
        #snr = math.exp(SNR / 10)
        snr = 10 ** (SNR / 10.0)
        scale = 1 / np.sqrt(snr * noise_power / speech_power)
        #noise = noise / np.sqrt(scale)
        print('Noise scaling:', scale, 'Noise Power:', noise_power, 'Speech Power:', speech_power, 'snr:', snr, 'SNR:', SNR, 'VAD duration:', vad_duration)
        return scale

def apply_noise_to_speech(source_speech_fn, noise_wav_path, dest_fn=None, tmp_dir=None):
    import shutil
    import uuid
    import random, os
    import torch, math
    import subprocess
    import numpy as np
    from ffmpeg_normalize._cmd_utils import get_ffmpeg_exe

    speech_sig, speech_info = read_audio(source_speech_fn)
    #speech_sig, sr, wav_info = read_audio(args.source_file)    
    #speech_sig, sr = sf.read(src_file)
    duration, frame_count = math.floor(speech_sig.duration_seconds), math.floor(speech_sig.frame_count())
    print(f'Noise augmentation->reading source:{source_speech_fn}, Before: {duration} secs, {frame_count} frames, sample width: {speech_sig.sample_width}')

    speech = np.array(speech_sig.get_array_of_samples()).astype(np.float32)

    #source: https://github.com/snakers4/silero-vad
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

    (_, get_speech_ts_adaptive, _, _, _, _, _) = utils

    num_steps = 4 # recommended 4 or 8
    ms_per_window = 250
    #num_samples_per_window = int(sample_rate / 1000 * ms_per_window)
    num_samples_per_window = int(speech_sig.frame_rate / 1000 * ms_per_window)
    step = int(num_samples_per_window / num_steps)
    
    speech_timestamps = get_speech_ts_adaptive(torch.from_numpy(speech), model,
                                            # number of samples in each window, 
                                            # our models were trained using 4000 
                                            # samples (250 ms) per window, so this 
                                            # is preferable value (lesser values 
                                            # reduce quality)
                                            num_samples_per_window=num_samples_per_window,
                                            # step size in samples
                                            step=step, 
                                            visualize_probs=False)

    total_vad_duration = 0
    if len(speech_timestamps) > 0:
        for speech_timestamp in speech_timestamps:
            vad_duration = speech_timestamp['end'] - speech_timestamp['start'] + 1
            total_vad_duration += vad_duration

    if tmp_dir is not None and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    subnoises = []
    weights = [1.0]
    for i, noise_fn in enumerate(noise_wav_path):        
        print(f'Noise augmentation->generating noise {i+1}: {os.path.basename(noise_fn)}')

        #noise_fn = random.sample(noise_wav_path, 1)[0]
        #print(f"Applying {noise_fn}...")
        [noise_fn, SNR] = noise_fn.split('@')
        SNR = float(SNR)
        noise_sig, _ = read_audio(noise_fn, speech_sig.frame_rate, speech_sig.sample_width)
        
        subnoise_fn = os.path.basename(noise_fn)
        subnoise_fn = subnoise_fn.split('.')
        subnoise_fn = f'subnoise_{".".join(subnoise_fn[:-1])}.{subnoise_fn[-1]}'
        if tmp_dir is not None:
            subnoise_fn = os.path.join(tmp_dir, subnoise_fn)
        
        precommand = []
        print(f'Noise augmentation->comparing {os.path.basename(noise_fn)} {noise_sig.duration_seconds} > {speech_sig.duration_seconds}')
        if noise_sig.duration_seconds > speech_sig.duration_seconds:
            start = max(0, int(math.floor(noise_sig.duration_seconds - speech_sig.duration_seconds - 1)))
            if start < 2:
                shutil.copy(noise_fn, subnoise_fn)
            else:
                start = random.sample([i for i in range(1, start)], 1)[0]
                print(f'Noise augmentation->adjust noise "{os.path.basename(noise_fn)}" to start time={start} secs')
                precommand = [f'{get_ffmpeg_exe()}', '-y', '-ss', str(start), '-i', f'{os.path.abspath(noise_fn)}', '-to', str(speech_sig.duration_seconds), '-c', 'copy', f'{os.path.abspath(subnoise_fn)}']
                print(f"Noise augmentation->running ffmpeg pre-command: {' '.join(precommand)}")
                p = subprocess.Popen(
                    precommand,
                    stdin=subprocess.PIPE,  # Apply stdin isolation by creating separate pipe.
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=False,
                )

                # simple running of command
                stdout, stderr = p.communicate()

                stdout = stdout.decode("utf8", errors="replace")
                stderr = stderr.decode("utf8", errors="replace")

                print(f'Noise augmentation->{str(stdout)}')

                if p.returncode != 0:
                    raise RuntimeError(f"Noise augmentation->Error running pre-command {precommand}: {str(stderr)}")
        else:
            shutil.copy(noise_fn, subnoise_fn)

        noise_fn = f'{os.path.abspath(subnoise_fn)}'        
        noise_sig, _ = read_audio(subnoise_fn, speech_sig.frame_rate, speech_sig.sample_width)
        noise_frames = np.array(noise_sig.get_array_of_samples())
        noise_frames = noise_frames.astype(np.float32)
        weight = compute_speech_noise_factor(SNR, speech, total_vad_duration, noise_frames)
        subnoises.append(noise_fn)
        weights.append(weight)

        #SNR = random.sample(snrs, 1)[0]
        #precommand = []
        #subnoise_fn = os.path.basename(noise_fn)
        #subnoise_fn = subnoise_fn.split('.')
        #subnoise_fn = f'subnoise-{i}.{subnoise_fn[-1]}'        
        #precommand = [f'{get_ffmpeg_exe()}', '-y', '-i', f'{os.path.abspath(noise_fn)}', '-filter:a', f'volume={weight}', f'{os.path.abspath(subnoise_fn)}']
        #print(f"Noise augmentation->running ffmpeg pre-command: {' '.join(precommand)}")
        #p = subprocess.Popen(
        #    precommand,
        #    stdin=subprocess.PIPE,  # Apply stdin isolation by creating separate pipe.
        #    stdout=subprocess.PIPE,
        #    stderr=subprocess.PIPE,
        #    universal_newlines=False,
        #)

        # simple running of command
        #stdout, stderr = p.communicate()

        #stdout = stdout.decode("utf8", errors="replace")
        #stderr = stderr.decode("utf8", errors="replace")

        #if p.returncode == 0:
        #    os.remove(noise_fn)
        #    noise_fn = os.path.abspath(subnoise_fn)
        #    noise_sig, _ = read_audio(subnoise_fn, speech_sig.frame_rate, speech_sig.sample_width)
        #    subnoises.append(subnoise_fn)
        #else:
        #    raise RuntimeError(f"Noise augmentation->error running pre-command {precommand}: {str(stderr)}")

    #print('NUMBER OF NOISES', len(noise_wav_path), len(subnoises))
    try:
        if len(subnoises) > 0:
            assert os.path.exists(os.path.abspath(source_speech_fn)), print('FILE NOT FOUND:', os.path.abspath(source_speech_fn))

            command = [f'{get_ffmpeg_exe()}', '-y', '-i', f'{os.path.abspath(source_speech_fn)}']
            for subnoise_fn in subnoises:
                command.extend(['-stream_loop', '-1', '-i', subnoise_fn])
            command.append('-filter_complex')   
            vols = ''
            audios = ''
            for j, weight in enumerate(weights):
                vols += f'[{j}:a]volume={weight}[a{j}];'
                audios += f'[a{j}]'
            command.extend([f'{vols}{audios}amix=inputs={len(subnoises) + 1}:duration=first:dropout_transition=0[a];[a]dynaudnorm[final]', '-map', '[final]', '-c:a', speech_info["codec_name"], '-b:a', f'{math.floor(int(speech_info["bit_rate"])/1000)}k', '-ac', speech_info["channels"], '-ar', speech_info["sample_rate"], '-f', speech_info["format_name"]])
            if dest_fn is None:
                command.extend(['-', '| '])
            else:
                base_dir, temp_fn = os.path.split(dest_fn)
                temp_fn = os.path.join(base_dir, f'noise_temp-{uuid.uuid4()}{os.path.splitext(temp_fn)[1]}')
                command.append(temp_fn)    

                #command.append(os.path.abspath(dest_fn))
                print(f"Noise augmentation->running ffmpeg command: {' '.join(command)}")
                p = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,  # Apply stdin isolation by creating separate pipe.
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=False,
                )

                # simple running of command
                stdout, stderr = p.communicate()

                stdout = stdout.decode("utf8", errors="replace")
                stderr = stderr.decode("utf8", errors="replace")

                print(f'Noise augmentation->{str(stdout)}')

                if p.returncode == 0:
                    dest_fn = os.path.abspath(dest_fn)
                    if os.path.exists(dest_fn):
                        os.remove(dest_fn)
                    shutil.move(temp_fn, dest_fn)
                    #source_speech_fn = dest_fn
                else:
                    raise RuntimeError(f'Noise augmentation->error running command "{command}": {str(stderr)}')

            for subnoise_fn in subnoises:
                if os.path.exists(subnoise_fn):
                    os.remove(subnoise_fn)

            return ' '.join(command)
    finally:
        if tmp_dir is not None and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True, onerror=None)

if __name__ == "__main__":
    from glob import glob
    #from vad_torch import VoiceActivityDetector
    import soundfile as sf
    import numpy as np
    import torchaudio
    import argparse
    import random
    import torch
    import math
    import os

    SEED = torch.initial_seed() % 2**32

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=int, default=0, help='0 - Apply noise over entire audio with respect to total speech duration (default) OR \n1 - Apply noise on speech segments only with respect to individual segment duration')
    parser.add_argument('noise_file', type=str, help='Noise wav file')
    parser.add_argument('source_file', type=str, help='Source audio wav file')
    parser.add_argument('dest_file', type=str, help='Target audio wav file')
    parser.add_argument("-t", "--temp-dir", default=f'tmp-{os.getpid()}', type=str, help="set scratch directory", required=False)
    parser.add_argument("-s", "--random-state", default=SEED, type=int, help="set random state", required=False)
    args = parser.parse_args()
    #print('args', args)

    import random
    import torch
    import numpy as np
    print('RANDOM STATE:', args.random_state)
    torch.manual_seed(args.random_state)
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Voice activity detection
    noise_wav_path = args.noise_file.split(',')
    noise_wav_path = [os.path.join(noise_wav_path[0], noise_wav_path[i]) for i in range(1, len(noise_wav_path))]

    apply_noise_to_speech(args.source_file, noise_wav_path, args.dest_file, args.temp_dir)