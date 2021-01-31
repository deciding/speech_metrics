import h5py
import os
#import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import resampy
from metrics import read_wav_scipy_float64, get_mfcc_pw, eval_nn_mcd, eval_pesq_8k
#from metrics import get_f0_pw_sptk, eval_rmse_f0
#from metrics import eval_pesq

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

#TODO
exp_dir='/workspace/GANVocoder/egs/vctk/voc1/exp/train_nodev_all_vctk_ms_melgan_v1/'
exp_dir='/workspace/ssd2/hifi-gan/cp_vctk_v1'
#exp_dir='/workspace/GANVocoder/egs/vctk/voc1/exp/train_nodev_all_vctk_multi_band_melgan.v2/'
use_file=False
checkpoint=None
checkpoint='g_00312000'

name=os.path.basename(exp_dir)
if 'cp_vctk' not in name:
    if checkpoint is None:
        checkpoints=os.listdir(f'{exp_dir}/wav')
        checkpoints.sort(key=lambda k: int(''.join(filter(str.isdigit, k))), reverse=True)
        checkpoint=checkpoints[0]
    logfile=f'logs/res_{name}_{checkpoint}.log'

    gt_globs=[
            '/workspace/GANVocoder/egs/vctk/voc1/dump1/train_nodev_all/norm/*/*.h5',
            '/workspace/GANVocoder/egs/vctk/voc1/dump1/dev_all/norm/*/*.h5',
            '/workspace/GANVocoder/egs/vctk/voc1/dump1/eval_all/norm/*/*.h5',
            '/workspace/GANVocoder/egs/vctk/voc1/dump/dev_all/norm/*/*.h5',
            '/workspace/GANVocoder/egs/vctk/voc1/dump/eval_all/norm/*/*.h5',
            ]

    gen_dirs=[
            f'{exp_dir}/wav/{checkpoint}/train_nodev_all',
            f'{exp_dir}/wav/{checkpoint}/dev_all',
            f'{exp_dir}/wav/{checkpoint}/eval_all',
            f'{exp_dir}/wav/{checkpoint}/dev_all',
            f'{exp_dir}/wav/{checkpoint}/eval_all',
            ]

    show_indices={'dev': [0], 'eval': [1], 'unseen_eval': [2, 3, 4]}
    show_map={k:{'mcd':0.0, 'f0_rmse':0.0, 'pesq':0.0} for k in show_indices.keys()}
else:
    if checkpoint is None:
        checkpoints=[x for x in os.listdir(f'{exp_dir}/') if 'g_' in x]
        checkpoints.sort(key=lambda k: int(''.join(filter(str.isdigit, k))), reverse=True)
        checkpoint=checkpoints[0]
    logfile=f'logs/res_{name}_{checkpoint}.log'

    gt_globs=[
            '/workspace/ssd2/hifi-gan/VCTK-Corpus/test22/*.wav',
            ]

    gen_dirs=[
            f'/workspace/ssd2/hifi-gan/generated_{name}_{checkpoint}/',
            ]

    show_indices={'unseen_eval': [0]}
    show_map={k:{'mcd':0.0, 'f0_rmse':0.0, 'pesq':0.0} for k in show_indices.keys()}



map_cnt_pairs=[[{'mcd':0.0, 'f0_rmse':0.0, 'pesq':0.0}, 0] for i in range(len(gt_globs))]

if not use_file:

    outf=open(logfile, 'a+')

    bads=['p330_424']

    #start_ind=1
    #start_ifile=478

    for ind, (gt_glob, gen_dir) in enumerate(list(zip(gt_globs, gen_dirs))):
        #if ind<start_ind:
        #    continue
        gt_files=glob(gt_glob)
        gt_files=[fpath for fpath in gt_files if os.path.basename(fpath).split('.')[0] not in bads]
        map_cnt_pairs[ind][1]=len(gt_files)
        metric_map=map_cnt_pairs[ind][0]
        #for gt_file in tqdm(gt_files):
        for i_file, gt_file in tqdm(list(enumerate(gt_files))):
            #if i_file<start_ifile:
            #    continue
            try:
                if 'cp_vctk' not in name:
                    h5path=gt_file
                    wavpath=f"{gen_dir}/{os.path.basename(gt_file).split('.')[0]}_gen.wav"
                    print(h5path, wavpath)
                else:
                    wav1path=gt_file
                    wavpath=f"{gen_dir}/{os.path.basename(gt_file).split('.')[0]}_generated.wav"
                    print(wav1path, wavpath)

                #============read wav===============#
                if 'cp_vctk' not in name:
                    with h5py.File(h5path) as f:
                        sig1, sr=np.array(f['wave']).astype(np.float64), 24000
                else:
                    #import pdb;pdb.set_trace()
                    sig1, sr=read_wav_scipy_float64(wav1path)
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
                    #print(sr)

                #============calibrate wav length===============#
                if len(sig1)!=len(sig2):
                    print(f'different lengths : {len(sig1)}, {len(sig2)}')
                l=min(len(sig1), len(sig2))
                sig1=sig1[:l]
                sig2=sig2[:l]

                #============get resampled wavs===============#
                #sig1_16k=resampy.resample(sig1, sr, 16000)
                #sig2_16k=resampy.resample(sig2, sr, 16000)
                sig1_8k=resampy.resample(sig1, sr, 8000)
                sig2_8k=resampy.resample(sig2, sr, 8000)

                #============get features===============#
                mfcc1 = get_mfcc_pw(sig1, sr)
                mfcc2 = get_mfcc_pw(sig2, sr)
                #f0_r = get_f0_pw_sptk(sig1, sr)
                #f0_s = get_f0_pw_sptk(sig2, sr)
                #f0_r = get_f0_pw_sptk(sig1, sr, method='dio')
                #f0_s = get_f0_pw_sptk(sig2, sr, method='dio')

                #============test metrics===============#
                mcd = eval_nn_mcd(mfcc1, mfcc2)
                #f0_rmse = eval_rmse_f0(f0_r, f0_s)[0]
                #print('pesq: ', eval_pesq(sig1_16k, sig2_16k, 16000))
                #import pdb;pdb.set_trace()
                pesq = eval_pesq_8k(sig1_8k, sig2_8k, 8000)
                metric_map['mcd'] += mcd
                #metric_map['f0_rmse'] += eval_rmse_f0(f0_r, f0_s)[0]
                ##print('pesq: ', eval_pesq(sig1_16k, sig2_16k, 16000))
                metric_map['pesq'] += pesq
                print(ind, mcd, pesq, metric_map['mcd']/(i_file+1), metric_map['pesq']/(i_file+1))
                #print(h5path, wavpath, mcd, f0_rmse, pesq, ind, i_file, len(gt_files))
                if 'cp_vctk' not in name:
                    outf.write(f'{h5path} {wavpath} {mcd} {pesq} {ind} {i_file} {len(gt_files)}\n')
                else:
                    outf.write(f'{wav1path} {wavpath} {mcd} {pesq} {ind} {i_file} {len(gt_files)}\n')
                outf.flush()
            except Exception as e:
                import pdb;pdb.set_trace()
                print(e)
        #print(metric_map)

    outf.close()
else:
    with open(logfile) as f:
        for line in f:
            line=line.strip()
            if line=='':
                continue
            _,_,mcd,pesq,ind,_,_=line.split()
            mcd=float(mcd)
            pesq=float(pesq)
            ind=int(ind)
            map_cnt_pairs[ind][1]+=1
            map_cnt_pairs[ind][0]['mcd']+=mcd
            map_cnt_pairs[ind][0]['pesq']+=pesq

for k, indices in show_indices.items():
    cur_k_total_num=0
    cur_res=show_map[k]
    for ind in indices:
        cur_metric_map=map_cnt_pairs[ind]
        cur_k_total_num+=cur_metric_map[1]
        for metric in cur_res:
            cur_res[metric]+=cur_metric_map[0][metric]
    for metric in cur_res:
        cur_res[metric]/=cur_k_total_num
    print(k)
    print(cur_res)

