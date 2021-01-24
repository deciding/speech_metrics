import h5py
import sys
import numpy as np
import resampy
from metrics import read_wav_scipy_float64, get_mfcc_pw, get_f0_pw_sptk, eval_nn_mcd, eval_rmse_f0, eval_pesq, eval_pesq_8k

#filename=sys.argv[1]
#outdir='.'
#
#with h5py.File(filename) as f:
#    wav=np.array(f['wave'])
#
#uttid=os.path.basename(filename)
#sf.write(os.path.join(outdir, f"{uttid}.wav"),
#         wav, 24000, "PCM_16")
#

h5path=sys.argv[1]
wavpath=sys.argv[2]

#============read wav===============#
with h5py.File(h5path) as f:
    sig1, sr=np.array(f['wave']).astype(np.float64), 24000
sig2, sr2=read_wav_scipy_float64(wavpath)

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
    print(sr)

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
mfcc1 = get_mfcc_pw(sig1, sr)
mfcc2 = get_mfcc_pw(sig2, sr)
#f0_r = get_f0_pw_sptk(sig1, sr)
#f0_s = get_f0_pw_sptk(sig2, sr)
f0_r = get_f0_pw_sptk(sig1, sr, method='dio')
f0_s = get_f0_pw_sptk(sig2, sr, method='dio')

#============test metrics===============#
print('nnmnkwii: ', eval_nn_mcd(mfcc1, mfcc2))
print('f0 rmse: ', eval_rmse_f0(f0_r, f0_s))
print('pesq: ', eval_pesq(sig1_16k, sig2_16k, 16000))
print('pesq nb: ', eval_pesq_8k(sig1_8k, sig2_8k, 8000))
