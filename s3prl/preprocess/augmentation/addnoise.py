def read_audio(src_file, target_sr=None):
    import soundfile as sf
    import torchaudio
    from utils import get_audio_info

    wav_info = get_audio_info(src_file)
    speech_sig, sr = torchaudio.load(src_file)

    if speech_sig.size(0) > 1:
        speech_sig = speech_sig.mean(dim=0, keepdim=True)

    if target_sr is not None and sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        speech_sig = transform(speech_sig)
        sr = target_sr

    assert target_sr is None or sr == target_sr
    return speech_sig.squeeze(0), sr, wav_info

def add_noise(vad_duration, speech_seg, sr, snrs, noise_wav_path):
    if vad_duration > 0:
        speech_power = torch.sum(speech_seg ** 2) / vad_duration
        noise_fn = random.sample(noise_wav_path, 1)[0]
        noise_sig, _, _ = read_audio(noise_fn, sr)
        noise_power = torch.sum(noise_sig ** 2) / noise_sig.shape[0]
        SNR = random.sample(snrs, 1)[0]
        snr = 10 ** (SNR / 10.0)
        noise_seg = noise_sig / torch.sqrt(snr * noise_power / speech_power)

        if speech_seg.shape[0] > noise_seg.shape[0]:
            # padding
            temp_wav = torch.zeros(speech_seg.shape[0])
            temp_wav[0:noise_seg.shape[0]] = noise_sig
            noise_seg = temp_wav
        else:
            # cutting
            noise_seg = noise_seg[0:speech_seg.shape[0]]

        return noise_seg + speech_seg
    else:
        return speech_seg

if __name__ == "__main__":
    from glob import glob
    #from vad_torch import VoiceActivityDetector
    import soundfile as sf
    #import numpy as np
    import torchaudio
    import argparse
    import random
    import torch
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

    speech_sig, sr, wav_info = read_audio(args.source_file)    
    #speech_sig, sr = sf.read(src_file)

    #source: https://github.com/snakers4/silero-vad
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

    (_, get_speech_ts_adaptive, _, _, _, _, _) = utils

    num_steps = 4 # recommended 4 or 8
    ms_per_window = 250
    num_samples_per_window=int(sr / 1000 * ms_per_window)
    step = int(num_samples_per_window / num_steps)
    speech_timestamps = get_speech_ts_adaptive(speech_sig, model,
                                                # number of samples in each window, 
                                                # our models were trained using 4000 
                                                # samples (250 ms) per window, so this 
                                                # is preferable value (lesser values 
                                                # reduce quality)
                                                num_samples_per_window=num_samples_per_window,
                                                # step size in samples
                                                step=step, 
                                                visualize_probs=False)

    if len(speech_timestamps) > 0:
        total_vad_duration = 0 
        for speech_timestamp in speech_timestamps:
            vad_duration = speech_timestamp['end'] - speech_timestamp['start']
            total_vad_duration += vad_duration

            if args.mode == 1 and vad_duration > 0:
                    start_frame = speech_timestamp['start'] - num_samples_per_window // 2
                    end_frame = speech_timestamp['end'] + num_samples_per_window // 2
                    speech_seg = speech_sig[start_frame:end_frame]
                    speech_sig[start_frame:end_frame] = add_noise(vad_duration, speech_seg, sr, snrs, noise_wav_path)

        if args.mode == 0:
            speech_sig = add_noise(total_vad_duration, speech_sig, sr, snrs, noise_wav_path)

    sf.write(args.dest_file, speech_sig.detach().numpy(), sr, sf.default_subtype(wav_info.format), wav_info.endian, format=wav_info.format)