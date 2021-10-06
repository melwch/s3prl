def read_audio(src_file, target_sr=None, target_bitrate=None):
    import math
    from pydub import AudioSegment, effects
    from pydub.utils import mediainfo

    format = src_file.split('.')[-1]
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
        snr = math.exp(SNR / 10)
        #snr = 10 ** (SNR / 10.0)
        scale = 1 / np.sqrt(snr * noise_power / speech_power)
        #noise = noise / np.sqrt(scale)
        print('noise scaling:', scale)
        return scale

def apply_noise_to_speech(source_speech_fn, snrs, noise_wav_path, dest_fn=None):
    import random, os
    import torch, math
    import subprocess
    import numpy as np

    speech_sig, speech_info = read_audio(source_speech_fn)
    duration, frame_count = math.floor(speech_sig.duration_seconds), math.floor(speech_sig.frame_count())
    print(f'Before: {duration} secs, {frame_count} frames, sample width: {speech_sig.sample_width}')
    #speech_sig, sr, wav_info = read_audio(args.source_file)    
    #speech_sig, sr = sf.read(src_file)

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

    noise_fn = random.sample(noise_wav_path, 1)[0]
    noise_sig, _ = read_audio(noise_fn, speech_sig.frame_rate, speech_sig.sample_width)
    noise_sig = np.array(noise_sig.get_array_of_samples())
    noise_sig = noise_sig.astype(np.float32)

    SNR = random.sample(snrs, 1)[0]    
    weight = compute_speech_noise_factor(SNR, speech, total_vad_duration, noise_sig)

    from ffmpeg_normalize._cmd_utils import get_ffmpeg_exe  

    
    if dest_fn is None:
        command = [f'{get_ffmpeg_exe()}', '-y', '-i', f'"{os.path.abspath(source_speech_fn)}"', '-i', f'"{os.path.abspath(noise_fn)}"', '-filter_complex', f'"amix=inputs=2:duration=first:weights=1.0 {weight}:dropout_transition=0,dynaudnorm"', '-c:a', speech_info["codec_name"], '-b:a', f'{math.floor(int(speech_info["bit_rate"])/1000)}k', '-ac', speech_info["channels"], '-ar', speech_info["sample_rate"], '-f', speech_info["format_name"], '-', '| ']
        return ' '.join(command)

    command = [f'{get_ffmpeg_exe()}', '-y', '-i', os.path.abspath(source_speech_fn), '-i', os.path.abspath(noise_fn), '-filter_complex', f'amix=inputs=2:duration=first:weights=1.0 {weight}:dropout_transition=0,dynaudnorm', '-c:a', speech_info["codec_name"], '-b:a', f'{min(8.0, math.floor(int(speech_info["bit_rate"])/1000))}k', '-ac', speech_info["channels"], '-ar', speech_info["sample_rate"], '-f', speech_info["format_name"], os.path.abspath(dest_fn)]
    #command.append(os.path.abspath(dest_fn))
    print(f"Running ffmpeg command: {' '.join(command)}")
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

    if p.returncode != 0:
        raise RuntimeError(f"Error running command {command}: {str(stderr)}")

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--snr', type=str, default=25, help='Signal Noise Ratio')
    parser.add_argument('mode', type=int, default=0, help='0 - Apply noise over entire audio with respect to total speech duration (default) OR \n1 - Apply noise on speech segments only with respect to individual segment duration')
    parser.add_argument('noise_file', type=str, help='Noise wav file')
    parser.add_argument('source_file', type=str, help='Source audio wav file')
    parser.add_argument('dest_file', type=str, help='Target audio wav file')
    args = parser.parse_args()

    # Voice activity detection
    snrs = [float(snr) for snr in args.snr.split(',')]    
    noise_wav_path = args.noise_file.split(',')
    noise_wav_path = [os.path.join(noise_wav_path[0], noise_wav_path[i]) for i in range(1, len(noise_wav_path))]

    apply_noise_to_speech(args.source_file, args.dest_file, snrs, noise_wav_path)