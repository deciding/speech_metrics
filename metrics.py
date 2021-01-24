import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import pyworld as pw
import pysptk
from pesq import pesq
import resampy
import librosa
#import nnmnkwii
from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.metrics import melcd


#============read wav===============#

def read_wav_scipy_typenorm(path):
    sr, sig = wav.read(path)

    if len(sig.shape) == 2:
        sig = sig[:, 0]

    if sig.dtype == np.int16:
        sig = sig / 32768.0
    elif sig.dtype == np.int32:
        sig = sig / 2147483648.0
    elif sig.dtype == np.uint8:
        sig = (sig - 128) / 128.0

    sig = sig.astype(np.float32)

    return sig, sr

def read_wav_scipy(path):
    sr, sig = wav.read(path)
    return sig, sr

#TODO: test difference, pw.dio must need double
def read_wav_scipy_float64(path):
    sr, sig = wav.read(path)
    sig = sig.astype(np.float64)
    return sig, sr

#============get features===============#

def get_mfcc_psf(sig, sr):
    return mfcc(sig, sr)

#TODO: test mcd
def get_mfcc_librosa(sig, sr):
    return librosa.feature.mfcc(sig, sr).T

def get_mfcc_pw(sig, sr):
    # get pitch
    f0, timeaxis = pw.dio(sig, sr, frame_period=5)
    # refine pitch
    f0 = pw.stonemask(sig, f0, timeaxis, sr)
    # get smoothed spec
    spectrogram = pw.cheaptrick(sig, f0, timeaxis, sr)
    # trim zero frames
    spectrogram = trim_zeros_frames(spectrogram)
    # get mfcc using pysptk
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=0.41)
    return mc.astype(np.float32)

def get_f0_pw_sptk(x_r, sr, frame_len='5', method='swipe'):
    # TODO: 要可以改動 frame len (ms) 或者 hop_size
    if method == 'harvest':
        f0_r, t = pw.harvest(x_r.astype(np.double), sr, frame_period=50)
    elif method == 'dio':
        f0_r, t = pw.dio(x_r.astype(np.double), sr, frame_period=50)
        f0_r = pw.stonemask(x_r, f0_r, t, sr)
    elif method == 'swipe':
        f0_r = pysptk.sptk.swipe(x_r.astype(np.double), sr, hopsize=128)
    elif method == 'rapt':
        f0_r = pysptk.sptk.rapt(x_r.astype(np.double), sr, hopsize=128)
    else:
        raise ValueError('no such f0 exract method')
    return f0_r

#============eval metrics===============#

# method: sq, euc, db
# sig: [T, D]
def eval_mattshannon_mcd(sig1, sig2, topK=0, method='euc'):
    if topK<=1:
        sig1=sig1[:, 1:]
        sig2=sig2[:, 1:]
    else:
        sig1=sig1[:, 1:topK]
        sig2=sig2[:, 1:topK]

    if method=='sq':
        res = np.mean(np.sum((sig1 - sig2) ** 2, axis=1))
    elif method=='euc':
        res = np.mean(np.sqrt(np.sum((sig1 - sig2) ** 2, axis=1)))
    elif method=='db':
        K = 10 / np.log(10) * np.sqrt(2)
        res = K * np.mean(np.sqrt(np.sum((sig1 - sig2) ** 2, axis=1)))
    return res

def eval_nn_mcd(sig1, sig2, topK=13):
    if topK<=1:
        sig1=sig1[:, 1:]
        sig2=sig2[:, 1:]
    else:
        sig1=sig1[:, 1:topK]
        sig2=sig2[:, 1:topK]

    return melcd(sig1, sig2)

def direct_mc(path):
    fs, x = wav.read(path)
    x = x.astype(np.float64)
    # get pitch
    f0, timeaxis = pw.dio(x, fs, frame_period=5)
    # refine pitch
    f0 = pw.stonemask(x, f0, timeaxis, fs)
    # get smoothed spec
    spectrogram = pw.cheaptrick(x, f0, timeaxis, fs)
    # trim zero frames
    spectrogram = trim_zeros_frames(spectrogram)
    # get mfcc using pysptk
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=0.41)
    return mc.astype(np.float32)

def direct_nn_mcd(path1, path2):
    x1, x2 = direct_mc(path1), direct_mc(path2)
    return melcd(x1[:, 1:13], x2[:, 1:13])




def pad_to(x, target_len):
    pad_len = target_len - len(x)

    if pad_len <= 0:
        return x[:target_len]
    else:
        return np.pad(x, (0, pad_len), 'constant', constant_values=(0, 0))

# should not normalize
def eval_rmse_f0(f0_r, f0_s, tone_shift=None):
    # length align
    f0_s = pad_to(f0_s, len(f0_r))

    # make unvoice / vooiced frame mask
    f0_r_uv = (f0_r == 0) * 1
    f0_r_v = 1 - f0_r_uv
    f0_s_uv = (f0_s == 0) * 1
    f0_s_v = 1 - f0_s_uv

    tp_mask = f0_r_v * f0_s_v
    tn_mask = f0_r_uv * f0_s_uv
    fp_mask = f0_r_uv * f0_s_v
    #fn_mask = f0_r_v * f0_s_uv

    if tone_shift is not None:
        shift_scale = 2 ** (tone_shift / 12)
        f0_r = f0_r * shift_scale

    # only calculate f0 error for voiced frame
    y = 1200 * np.abs(np.log2(f0_r + f0_r_uv) - np.log2(f0_s + f0_s_uv))
    y = y * tp_mask
    # print(y.sum(), tp_mask.sum())
    f0_rmse_mean = y.sum() / tp_mask.sum()

    # only voiced/ unvoiced accuracy/precision
    vuv_precision = tp_mask.sum() / (tp_mask.sum() + fp_mask.sum())
    vuv_accuracy = (tp_mask.sum() + tn_mask.sum()) / len(y)

    return f0_rmse_mean, vuv_accuracy, vuv_precision

def eval_pesq(ref, syn, sr):
    return pesq(sr, ref, syn, 'wb')

def eval_pesq_8k(ref, syn, sr):
    assert sr==8000, 'the sampling rate for pesq narrow band must be 8k'
    return pesq(sr, ref, syn, 'nb')

if __name__ == '__main__':
    import sys
    path1=sys.argv[1]
    path2=sys.argv[2]

    #============read wav===============#
    #sig1, sr=read_wav_scipy(path1)
    #sig2, sr2=read_wav_scipy(path2)
    #sig1, sr=read_wav_scipy_typenorm(path1)
    #sig2, sr2=read_wav_scipy_typenorm(path2)
    sig1, sr=read_wav_scipy_float64(path1)
    sig2, sr2=read_wav_scipy_float64(path2)

    #============calibrate sr===============#
    # TODO: sr or sr2?
    if sr!=sr2:
        print(f'different sampling rate : {sr}, {sr2}')
    else:
        ref_sr=sr2
        if sr!=ref_sr:
            sig1=resampy.resample(sig1, sr, ref_sr)
        if sr2!=ref_sr:
            sig2=resampy.resample(sig2, sr2, ref_sr)
        sr=ref_sr

    #============calibrate wav length===============#
    if len(sig1)!=len(sig2):
        print(f'different lengths : {len(sig1)}, {len(sig2)}')
    l=min(len(sig1), len(sig2))
    sig1=sig1[:l]
    sig2=sig2[:l]

    #============get resampled wavs===============#
    sig1_16k=resampy.resample(sig1, sr, 16000)
    sig2_16k=resampy.resample(sig2, sr, 16000)
    sig1_8k=resampy.resample(sig1, sr, 8000)
    sig2_8k=resampy.resample(sig2, sr, 8000)

    #============get features===============#
    #mfcc1 = get_mfcc_psf(sig1, sr)
    #mfcc2 = get_mfcc_psf(sig2, sr)
    mfcc1 = get_mfcc_pw(sig1, sr)
    mfcc2 = get_mfcc_pw(sig2, sr)
    #f0_r = get_f0_pw_sptk(sig1, sr)
    #f0_s = get_f0_pw_sptk(sig2, sr)
    f0_r = get_f0_pw_sptk(sig1, sr, method='dio')
    f0_s = get_f0_pw_sptk(sig2, sr, method='dio')

    #============test metrics===============#
    #print('nnmnkwii: ', direct_nn_mcd(path1, path2))
    print('nnmnkwii: ', eval_nn_mcd(mfcc1, mfcc2))
    print('db: ', eval_mattshannon_mcd(mfcc1, mfcc2, method='db'))
    print('euc: ', eval_mattshannon_mcd(mfcc1, mfcc2))
    print('euc 13: ', eval_mattshannon_mcd(mfcc1, mfcc2, topK=13))
    #print('f0 rmse swipe: ', eval_rmse_f0(f0_r, f0_s))
    print('f0 rmse dio: ', eval_rmse_f0(f0_r, f0_s))
    #print('f0 rmse harvest: ', eval_rmse_f0(sig1, sig2, sr, method='harvest'))
    #print('f0 rmse dio: ', eval_rmse_f0(sig1, sig2, sr, method='dio'))
    #print('f0 rmse rapt: ', eval_rmse_f0(sig1, sig2, sr, method='rapt'))
    print('pesq: ', eval_pesq(sig1_16k, sig2_16k, 16000))
    print('pesq nb: ', eval_pesq_8k(sig1_8k, sig2_8k, 8000))

