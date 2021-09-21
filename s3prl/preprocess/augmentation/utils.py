def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    # http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this source:
    # https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75
    #
    # MIT license, John Meade, 2020
    import numpy as np
    
    snr = float("inf")

    fn = 'snr.csv'
    g_vals = None
    try:
        with open(fn, 'r') as f:
            g_vals = f.readlines().split(',')
            g_vals = np.array([float(v) for v in g_vals])
    except:
        print("Unable to load gamma distribution model", fn)

    if g_vals is not None:
        # init
        eps = 1e-10
        # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
        db_vals = np.arange(-20, 101)
        

        # peak normalize, get magnitude, clip lower bound
        wav = np.array(wav)
        wav = wav / abs(wav).max()
        abs_wav = abs(wav)
        abs_wav[abs_wav < eps] = eps

        # calcuate statistics
        # E[|z|]
        v1 = max(eps, abs_wav.mean())
        # E[log|z|]
        v2 = np.log(abs_wav).mean()
        # log(E[|z|]) - E[log(|z|)]
        v3 = np.log(v1) - v2

        # table interpolation
        wav_snr_idx = None
        if any(g_vals < v3):
            wav_snr_idx = np.where(g_vals < v3)[0].max()
        # handle edge cases or interpolate
        if wav_snr_idx is None:
            wav_snr = db_vals[0]
        elif wav_snr_idx == len(db_vals) - 1:
            wav_snr = db_vals[-1]
        else:
            wav_snr = db_vals[wav_snr_idx] + \
                (v3-g_vals[wav_snr_idx]) / (g_vals[wav_snr_idx+1] - \
                g_vals[wav_snr_idx]) * (db_vals[wav_snr_idx+1] - db_vals[wav_snr_idx])

        # Calculate SNR
        dEng = sum(wav**2)
        dFactor = 10**(wav_snr / 10)
        dNoiseEng = dEng / (1 + dFactor) # Noise energy
        dSigEng = dEng * dFactor / (1 + dFactor) # Signal energy
        snr = 10 * np.log10(dSigEng / dNoiseEng)

    return snr

def glob_re(path):
    import os, re
    dirname, basename = os.path.split(path)
    return [os.path.join(dirname, fn) for fn in filter(re.compile(basename).match, os.listdir(dirname))]

def get_audio_info(wav_fn, verbose=False):
    import soundfile as sf
    wav_info = sf.info(wav_fn)

    codecs = {}
    with open("codecs.info", "r") as f:
        lines = f.readlines()
        for line in lines:
            [key, value] = line.strip().split("=")
            codecs[key] = value

    codecs_torch = {}
    with open("codecs_for_torch.info", "r") as f:
        lines = f.readlines()
        for line in lines:
            [key, value] = line.strip().split("=")
            codecs_torch[key] = value

    wav_info.codec = codecs[wav_info.subtype_info]
    wav_info.codec_torch = codecs_torch[wav_info.subtype_info]
    
    bitrate_header = "bytes/sec"
    bitrate_header_pos = wav_info.extra_info.lower().index(bitrate_header)
    bitrate_header_pos = wav_info.extra_info.lower().index(":", bitrate_header_pos)
    wav_info.bitrate = float(wav_info.extra_info[bitrate_header_pos+1:wav_info.extra_info.lower().index('\n', bitrate_header_pos)].strip()) / 1000
    wav_info.brate = wav_info.bitrate
    wav_info.bitrate = f"{wav_info.bitrate}k"

    if verbose:
        print("Audio:", wav_fn)
        print("Sample Rate:", wav_info.samplerate)
        print("Channels:", wav_info.channels)
        print("Format:", wav_info.format)
        print("Format Info", wav_info.format_info)
        print("Subtype:", wav_info.subtype_info)
        print("Codec:", wav_info.codec)
        print("Bitrate:", wav_info.bitrate)
        print("Endian:", wav_info.endian)
        print("Check:", sf.check_format(wav_info.format, wav_info.subtype, wav_info.endian))
        print("Extra:", wav_info.extra_info)

    return wav_info