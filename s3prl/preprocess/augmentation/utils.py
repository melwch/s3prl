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

def extract_file(filepath: str, data_dir: str):
    import tarfile, logging

    try:
        with tarfile.open(filepath) as tar:
            tar.extractall(data_dir)
    except:
        logging.info('Not extracting. Maybe already there?')

pbar=None
def show_progress(block_num, block_size, total_size):
    from tqdm import tqdm
    
    global pbar   
    if pbar is None:
        pbar = tqdm(total=total_size)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar = None

def download_file(destination: str, source: str):
    import os, logging
    from urllib import request

    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    if not os.path.exists(destination):
        logging.info(f"{destination} does not exist. Downloading ...")
        request.urlretrieve(source, filename=destination + '.tmp', reporthook=show_progress)
        os.rename(destination + '.tmp', destination)
        logging.info(f"Downloaded {destination}.")
    else:
        logging.info(f"Destination {destination} exists. Skipping.")
    return destination

def copy_and_rename(file_path: str, noise_dir: str):
    import os, shutil

    wavelist = []
    filenames=os.listdir(file_path)
    for filename in filenames:
        _, format = os.path.splitext(os.path.join(file_path, filename))
        if format == '.wav':  
            wavelist.append(filename)

    step = 1
    for file in wavelist:
        f_src = os.path.join(file_path, 'sound-bible', file)
        fname = '_'.join(os.path.basename(f_src).split('.')[:-1])
        f_dst = os.path.join(file_path, noise_dir, f'{step}_{fname}.wav')
        if os.path.exists(f_dst):
            print("File exist:", f_dst)
        else:
            shutil.copyfile(f_src, f_dst)
        step += 1

def download_and_extract_musan(data_root: str):
    import os, shutil, logging

    data_set = 'MUSAN'
    URL = 'https://www.openslr.org/resources/17/musan.tar.gz'

    download_musan_noise_dir = os.path.join(data_root, 'musan')
    if not os.path.exists(download_musan_noise_dir):
        os.makedirs(download_musan_noise_dir)
        
    # Download noise dataset
    file_path = os.path.join(download_musan_noise_dir, data_set + ".tar.gz")
    logging.info(f"Getting {data_set}")
    download_file(file_path, URL)
    logging.info(f"Extracting {data_set}")
    extract_file(file_path, download_musan_noise_dir)
    
    copy_and_rename(os.path.join(download_musan_noise_dir, 'free-sound'), data_root)
    copy_and_rename(os.path.join(download_musan_noise_dir, 'sound-bible'), data_root)

    shutil.rmtree(download_musan_noise_dir, ignore_errors=False, onerror=None)

def download_and_extract_BUT_Speech(data_root: str):
    from tqdm import tqdm
    import os, shutil, logging
    
    # download and extrat RIR files
    data_set = 'ReverDB_dataset'
    URL = 'http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz'

    download_rir_dir = os.path.join(data_root, 'reverbdb')
    if not os.path.exists(download_rir_dir):
        os.makedirs(download_rir_dir)
    
    file_path = os.path.join(download_rir_dir, data_set + ".tar.gz")
    logging.info(f"Getting {data_set}")
    download_file(file_path, URL)
    logging.info(f"Extracting {data_set}")
    extract_file(file_path, download_rir_dir)
      
    print('Processing BUT_ReverbDB dataset...')
    
    Room_name = ['Hotel_SkalskyDvur_ConferenceRoom2', 
                 'Hotel_SkalskyDvur_Room112', 
                 'VUT_FIT_E112',
                 'VUT_FIT_L207',
                 'VUT_FIT_L212', 
                 'VUT_FIT_L227', 
                 'VUT_FIT_Q301', 
                 'VUT_FIT_C236', 
                 'VUT_FIT_D105']
    jobs = []
    for i in range(9):
        print("Moving", Room_name[i])
        speaker_name = os.listdir(os.path.join(data_root, Room_name[i], 'MicID01'))

        for j in range(len(speaker_name)):
            position_name = []
            for lists in os.listdir(os.path.join(data_root, Room_name[i], 'MicID01', speaker_name[j])):
                sub_path = os.path.join(data_root, Room_name[i], 'MicID01', speaker_name[j], lists)

                if os.path.isdir(sub_path):
                    position_name.append(sub_path)

            for k in range(len(position_name)):
                selected_rir_path = os.path.join(position_name[k], 'RIR')

                src = os.path.join(selected_rir_path, os.listdir(selected_rir_path)[0])
                basis = src[len(data_root):].replace("/", "_").replace("\\", "_").split(".")[0]
                dest = os.path.join(data_root, f"{len(jobs) + 1}_{basis}.wav")
                shutil.copyfile(src, dest)
                jobs.append((src, dest))

    for src, dest in tqdm(jobs):
        print(f"Copy from {src} to {dest}")
        shutil.copyfile(src, dest)

    shutil.rmtree(download_rir_dir, ignore_errors=False, onerror=None)