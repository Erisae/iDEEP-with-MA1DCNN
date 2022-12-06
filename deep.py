import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import gzip
import pickle
import timeit
import argparse

from optparse import OptionParser
import copy, os, pdb, random, shutil, subprocess, time

#import h5py
from scipy.stats import spearmanr
import seaborn as sns
from sklearn import preprocessing
#from sklearn.externals import joblib

if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')
#cuda = False        


#weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint ""'
weblogo_opts = '-X NO --fineprint ""'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" U U'

def load_data(path):
    """
        Load data matrices from the specified folder.
    """

    data = dict()

    data["Y"] = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_Response.tab.gz")),
                                            skiprows=1)

def get_motif_proteins(meme_db_file):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    motif_protein = {}
    for line in open(meme_db_file):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
            else:
                motif_protein[a[1]] = a[2]
    return motif_protein


def info_content(pwm, transpose=False, bg_gc=0.415):
    ''' Compute PWM information content.

    In the original analysis, I used a bg_gc=0.5. For any
    future analysis, I ought to switch to the true hg19
    value of 0.415.
    '''
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    bg_pwm = [1-bg_gc, bg_gc, bg_gc, 1-bg_gc]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic


def make_filter_pwm(filter_fasta):
    ''' Make a PWM for this filter from its top hits '''

    nts = {'A':0, 'C':1, 'G':2, 'U':3}
    pwm_counts = []
    nsites = 4 # pseudocounts
    for line in open(filter_fasta):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append(np.array([1.0]*4))

            # count
            for i in range(len(seq)):
                try:
                    pwm_counts[i][nts[seq[i]]] += 1
                except KeyError:
                    pwm_counts[i] += np.array([0.25]*4)

    # normalize
    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j]/float(nsites) for j in range(4)])

    return np.array(pwm_freqs), nsites-4


def meme_add(meme_out, f, filter_pwm, nsites, trim_filters=False):
    ''' Print a filter to the growing MEME file

    Attrs:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    '''
    if not trim_filters:
        ic_start = 0
        ic_end = filter_pwm.shape[0]-1
    else:
        ic_t = 0.2

        # trim PWM of uninformative prefix
        ic_start = 0
        while ic_start < filter_pwm.shape[0] and info_content(filter_pwm[ic_start:ic_start+1]) < ic_t:
            ic_start += 1

        # trim PWM of uninformative suffix
        ic_end = filter_pwm.shape[0]-1
        while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end+1]) < ic_t:
            ic_end -= 1

    if ic_start < ic_end:
        print ('MOTIF filter%d' % f, file=meme_out)
        print ('letter-probability matrix: alength= 4 w= %d nsites= %d' % (ic_end-ic_start+1, nsites), file=meme_out)

        for i in range(ic_start, ic_end+1):
            print ('%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]), file=meme_out)
        print ('',file=meme_out)


def meme_intro(meme_file, seqs):
    ''' Open MEME motif format file and print intro

    Attrs:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs

    Returns:
        mem_out : open MEME file
    '''
    nts = {'A':0, 'C':1, 'G':2, 'U':3}

    # count
    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    # open file for writing
    meme_out = open(meme_file, 'w')

    # print intro material
    print ('MEME version 4', file=meme_out)
    print ('', file=meme_out)
    print ('ALPHABET= ACGU', file=meme_out)
    print ('', file=meme_out)
    print ('Background letter frequencies:', file=meme_out)
    print ('A %.4f C %.4f G %.4f U %.4f' % tuple(nt_freqs), file=meme_out)
    print ('', file=meme_out)

    return meme_out


def name_filters(num_filters, tomtom_file, meme_db_file):
    ''' Name the filters using Tomtom matches.

    Attrs:
        num_filters (int) : total number of filters
        tomtom_file (str) : filename of Tomtom output table.
        meme_db_file (str) : filename of MEME db

    Returns:
        filter_names [str] :
    '''
    # name by number
    filter_names = ['f%d'%fi for fi in range(num_filters)]

    # name by protein
    if tomtom_file is not None and meme_db_file is not None:
        motif_protein = get_motif_proteins(meme_db_file)

        # hash motifs and q-value's by filter
        filter_motifs = {}

        tt_in = pd.read_csv(tomtom_file, sep='\t')
        for index, row in tt_in.iterrows():
            if row.isnull().values.any():
                continue
            fi = int(row['Query_ID'][6:])
            motif_id = row['Target_ID']
            qval = float(row['q-value'])


            filter_motifs.setdefault(fi,[]).append((qval,motif_id))

        # assign filter's best match
        for fi in filter_motifs:
            top_motif = sorted(filter_motifs[fi])[0][1]
            filter_names[fi] += '_%s' % motif_protein[top_motif]

    return np.array(filter_names)


################################################################################
# plot_target_corr
#
# Plot a clustered heatmap of correlations between filter activations and
# targets.
#
# Input
#  filter_outs:
#  filter_names:
#  target_names:
#  out_pdf:
################################################################################
def plot_target_corr(filter_outs, seq_targets, filter_names, target_names, out_pdf, seq_op='mean'):
    num_seqs = filter_outs.shape[0]
    num_targets = len(target_names)

    if seq_op == 'mean':
        filter_outs_seq = filter_outs.mean(axis=2)
    else:
        filter_outs_seq = filter_outs.max(axis=2)

    # std is sequence by filter.
    filter_seqs_std = filter_outs_seq.std(axis=0)
    filter_outs_seq = filter_outs_seq[:,filter_seqs_std > 0]
    filter_names_live = filter_names[filter_seqs_std > 0]

    filter_target_cors = np.zeros((len(filter_names_live),num_targets))
    for fi in range(len(filter_names_live)):
        for ti in range(num_targets):
            cor, p = spearmanr(filter_outs_seq[:,fi], seq_targets[:num_seqs,ti])
            filter_target_cors[fi,ti] = cor

    cor_df = pd.DataFrame(filter_target_cors, index=filter_names_live, columns=target_names)

    sns.set(font_scale=0.3)
    plt.figure()
    sns.clustermap(cor_df, cmap='BrBG', center=0, figsize=(8,10))
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_seq_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    # compute filter output means per sequence
    filter_seqs = filter_outs.mean(axis=2)

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in sequence segments.
#
# Mean doesn't work well for the smaller segments for some reason, but taking
# the max looks OK. Still, similar motifs don't cluster quite as well as you
# might expect.
#
# Input
#  filter_outs
################################################################################
def plot_filter_seg_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    b = filter_outs.shape[0]
    f = filter_outs.shape[1]
    l = filter_outs.shape[2]

    s = 5
    while l/float(s) - (l/s) > 0:
        s += 1
    print ('%d segments of length %d' % (s,l/s))

    # split into multiple segments
    filter_outs_seg = np.reshape(filter_outs, (b, f, s, l/s))

    # mean across the segments
    filter_outs_mean = filter_outs_seg.max(axis=3)

    # break each segment into a new instance
    filter_seqs = np.reshape(np.swapaxes(filter_outs_mean, 2, 1), (s*b, f))

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)
    if whiten:
        dist = 'euclidean'
    else:
        dist = 'cosine'

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], metric=dist, row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
    plt.close()


################################################################################
# filter_motif
#
# Collapse the filter parameter matrix to a single DNA motif.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_motif(param_matrix):
    nts = 'ACGU'

    motif_list = []
    for v in range(param_matrix.shape[1]):
        max_n = 0
        for n in range(1,4):
            if param_matrix[n,v] > param_matrix[max_n,v]:
                max_n = n

        if param_matrix[max_n,v] > 0:
            motif_list.append(nts[max_n])
        else:
            motif_list.append('N')

    return ''.join(motif_list)


################################################################################
# filter_possum
#
# Write a Possum-style motif
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_possum(param_matrix, motif_id, possum_file, trim_filters=False, mult=200):
    # possible trim
    trim_start = 0
    trim_end = param_matrix.shape[1]-1
    trim_t = 0.3
    if trim_filters:
        # trim PWM of uninformative prefix
        while trim_start < param_matrix.shape[1] and np.max(param_matrix[:,trim_start]) - np.min(param_matrix[:,trim_start]) < trim_t:
            trim_start += 1

        # trim PWM of uninformative suffix
        while trim_end >= 0 and np.max(param_matrix[:,trim_end]) - np.min(param_matrix[:,trim_end]) < trim_t:
            trim_end -= 1

    if trim_start < trim_end:
        possum_out = open(possum_file, 'w')
        print ('BEGIN GROUP', file=possum_out)
        print ('BEGIN FLOAT', file=possum_out)
        print ('ID %s' % motif_id, file=possum_out)
        print ('AP DNA', file=possum_out)
        print ('LE %d' % (trim_end+1-trim_start), file=possum_out)
        for ci in range(trim_start,trim_end+1):
            print ('MA %s' % ' '.join(['%.2f'%(mult*n) for n in param_matrix[:,ci]]), file=possum_out)
        print ('END', file=possum_out)
        print ('END', file=possum_out)

        possum_out.close()


################################################################################
# plot_filter_heat
#
# Plot a heatmap of the filter's parameters.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_heat(param_matrix, out_pdf):
    param_range = abs(param_matrix).max()

    sns.set(font_scale=2)
    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(param_matrix, cmap='PRGn', linewidths=0.2, vmin=-param_range, vmax=param_range)
    ax = plt.gca()
    ax.set_xticklabels(range(1,param_matrix.shape[1]+1))
    ax.set_yticklabels('UGCA', rotation='horizontal') # , size=10)
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_logo
#
# Plot a weblogo of the filter's occurrences
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
#weblogo -X NO -Y NO --errorbars NO --fineprint ""  -C "#CB2026" A A -C "#34459C" C C -C "#FBB116" G G -C "#0C8040" T T <filter1_logo.fa >filter1.eps
################################################################################
def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None):
    if maxpct_t:
        all_outs = np.ravel(filter_outs)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_prefix, 'w')
    filter_count = 0
    for i in range(filter_outs.shape[0]):
        for j in range(filter_outs.shape[1]):
            if filter_outs[i,j] > raw_t:
                kmer = seqs[i][j:j+filter_size]
                incl_kmer = len(kmer) - kmer.count('N')
                if incl_kmer <filter_size:
                    continue
                print ('>%d_%d' % (i,j), file=filter_fasta_out)
                print (kmer, file=filter_fasta_out)
                filter_count += 1
    filter_fasta_out.close()
    print ('plot logo')
    # make weblogo
    if filter_count > 0:
        weblogo_cmd = './weblogo/seqlogo -c -F PNG -f %s.fa > %s.png' % (out_prefix, out_prefix)
        os.system(weblogo_cmd)


################################################################################
# plot_score_density
#
# Plot the score density and print to the stats table.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_score_density(f_scores, out_pdf):
    sns.set(font_scale=1.3)
    plt.figure()
    sns.histplot(f_scores, kde=False)
    plt.xlabel('ReLU output')
    plt.savefig(out_pdf)
    plt.close()

    return f_scores.mean(), f_scores.std()

def get_motif_fig(filter_weights, filter_outs, out_dir, seqs, sample_i = 0):
    print ('plot motif fig', out_dir)
    #seqs, seq_targets = get_seq_targets(protein)
    #pdb.set_trace()
    # num_filters = filter_weights.shape[0]
    num_filters = 16
    filter_size = 7

    #pdb.set_trace()
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = meme_intro('%s/filters_meme.txt'%out_dir, seqs)

    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        # plot_filter_heat(filter_weights[f,:,:][:, :filter_size], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        # filter_possum(filter_weights[f,:,:][:, :filter_size], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:,f, :], filter_size, seqs, '%s/filter%d_logo'%(out_dir,f), maxpct_t=0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01 
    os.system('docker run -v /Users/arika/Desktop/ideeps:/home/meme --user `id -u`:`id -g` memesuite/memesuite tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (out_dir, out_dir, 'Ray2013_rbp_RNA.meme'))

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.tsv'%out_dir, './Ray2013_rbp_RNA.meme')


    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt'%out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print ('%3s  %19s  %10s  %5s  %6s  %6s' % header_cols, file=table_out)

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f,:,:])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:,:, f]), '%s/filter%d_dens.pdf' % (out_dir,f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print ('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols, file=table_out)

    table_out.close()


    #################################################################
    # global filter plots
    #################################################################
    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print(filter_outs.shape)
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%out_dir)

def get_feature(model, X_batch, index):
    inputs = [K.learning_phase()] + [model.inputs[index]]
    _convout1_f = K.function(inputs, model.layers[0].layers[index].layers[1].output)
    activations =  _convout1_f([0] + [X_batch[index]])
    
    return activations

def get_motif(filter_weights_old, filter_outs, testing, y = [], index = 0, dir1 = 'motif/', structure  = None):
    #sfilter = model.layers[0].layers[index].layers[0].get_weights()
    #filter_weights_old = np.transpose(sfilter[0][:,0,:,:], (2, 1, 0)) #sfilter[0][:,0,:,:]
    print(filter_weights_old.shape)
    #pdb.set_trace()
    filter_weights = []
    for x in filter_weights_old:
        #normalized, scale = preprocess_data(x)
        #normalized = normalized.T
        #normalized = normalized/normalized.sum(axis=1)[:,None]
        x = x - x.mean(axis = 0)
        filter_weights.append(x)
        
    filter_weights = np.array(filter_weights)
    #pdb.set_trace()
    #filter_outs = get_feature(model, testing, index)
    #pdb.set_trace()
    
    #sample_i = np.array(random.sample(xrange(testing.shape[0]), 500))
    sample_i =0

    out_dir = dir1
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if index == 0:    
        get_motif_fig(filter_weights, filter_outs, out_dir, testing, sample_i)


def padding_sequence_new(seq, max_len = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def split_overlap_seq(seq, window_size = 101):
    
    overlap_size = 20
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            #pdb.set_trace()
            #start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len = window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    print(seq_file)
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def load_graphprot_data(protein, train = True, path = './GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    #trids = get_6_trids()
    #nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        #rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        
        seq_array = get_RNA_seq_concolutional_array(seq)
        #tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(seq_array)
    
    return np.array(rna_array), label

def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        #flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        
        if num_of_ins >channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
              # bag_subt.append(random.choice(bag_subt))
              tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
              bag_subt.append(tri_fea.T)
        
        bags.append(np.array(bag_subt))
    
        
    return bags, labels
    #for data in pairs.iteritems():
    #    ind1 = trids.index(key)
    #    emd_weight1 = embedding_rna_weights[ord_dict[str(ind1)]]

def get_bag_data_1_channel(data, max_len = 160): #aaa
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        # pdb.set_trace()
        #bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        
        bags.append(np.array(bag_subt))
    
        
    return bags, labels

def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for idx, (X, y) in enumerate(train_loader):
              #for X, y in zip(X_train, y_train):
              #X_v = Variable(torch.from_numpy(X.astype(np.float32)))
              #y_v = Variable(torch.from_numpy(np.array(ys)).long())
            X = X.squeeze(1)
            X_v = Variable(X)
            y_v = Variable(y)
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
                  
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item()) # need change to loss_list.append(loss.item()) for pytorch v0.4 or above

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        #X_list = batch(X, batch_size)
        #y_list = batch(y, batch_size)
        #pdb.set_trace()
        print (X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                              torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            print (loss)
            #rint("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, X, y, batch_size=32):
        
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        #lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        #cc = self._accuracy(classes, y)
        return loss.data[0], auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()        
        y_pred = self.model(X)
        return y_pred 

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)


def get_all_data(protein, channel = 7):

    data = load_graphprot_data(protein)
    test_data = load_graphprot_data(protein, train = False)
    #pdb.set_trace()
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data)
        test_bags, true_y = get_bag_data_1_channel(test_data)
    else:
        train_bags, label = get_bag_data(data)
    #pdb.set_trace()
        test_bags, true_y = get_bag_data(test_data)

    return train_bags, label, test_bags, true_y

def run_network(model_type, X_train, test_bags, y_train, channel = 7, window_size = 107):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter = 16, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = 16, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = 16, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size = window_size, channel = channel, labcounts = 4)
    elif model_type == 'MA1DCNN':
        model = MA1DCNN(A_EAM, A_CAM, width= window_size, in_channel=4)
    else:
        print ('only support CNN, CNN-LSTM, ResNet and DenseNet model')

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=100, nb_epoch=50)
    
    print ('predicting')       
    pred = model.predict_proba(test_bags)
    return pred, model

def run_ideepe_on_graphprot(model_type = 'CNN', local = False, ensemble = True):
    data_dir = './GraphProt_CLIP_sequences/'
    
    finished_protein = set()
    start_time = timeit.default_timer()
    if local:
        window_size = 107
        channel = 7
        lotext = 'local'
    else:
        window_size = 160 # aaa
        channel = 1
        lotext = 'global'
    if ensemble:
        outputfile = lotext + '_result_adam_ensemble_' + model_type
    else:
        outputfile = lotext + '_result_adam_individual_' + model_type
        
    fw = open(outputfile, 'w')
    
    for protein in os.listdir(data_dir):
        protein = protein.split('.')[0]
        if protein in finished_protein:
                continue
        finished_protein.add(protein)
        print (protein)
        fw.write(protein + '\t')
        hid = 16
        if not ensemble:
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel = channel)
            predict = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels), protein, channel = channel, window_size = window_size)
        else:
            print ('ensembling')
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel = 1)
            predict1 = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels), channel = 1, window_size = 160)
            train_bags, train_labels, test_bags, test_labels = [], [], [], []
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel = 7)
            predict2 = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels), channel = 7, window_size = 107)
            predict = (predict1 + predict2)/2.0
            train_bags, train_labels, test_bags = [], [], []
        
        auc = roc_auc_score(test_labels, predict)
        print ('AUC:', auc)
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    fw.close()
    end_time = timeit.default_timer()
    print ("Training final took: %.2f s" % (end_time - start_time))

def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label

def detect_motifs(model, test_seqs, X_train, output_dir = 'motifs', channel = 1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para =  param.data.cpu().numpy()
            break
        	#test_data = load_graphprot_data(protein, train = True)
        	#test_seqs = test_data["seq"]
        N = len(test_seqs)
        if N > 15000: # do need all sequence to generate motifs and avoid out-of-memory
        	sele = 15000
        else:
        	sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]
        
        X_train = X_train[ix_test, :, :, :].squeeze(1)
        test_seq = []
        for ind in ix_test:
        	test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)
        print(filter_outs.shape)
        print(model.layer1out(X_train).shape)
        print(X_train.shape)
        print(layer1_para.shape)
        get_motif(layer1_para[:, :, :], filter_outs, test_seqs, dir1 = output_dir)

def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16, motif = False, motif_seqs = [], motif_outdir = 'motifs'):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size = window_size, channel = channel, labcounts = 4)
    elif model_type == 'MA1DCNN':
        print(window_size, channel)
        model = MA1DCNN(A_EAM, A_CAM, width= window_size, in_channel=4)
    else:
        print ('only support CNN, CNN-LSTM, ResNet and DenseNet model')

    if cuda:
        model = model.cuda()
    model.train()
    # clf = Estimator(model)
    # clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    #             loss=nn.CrossEntropyLoss())
    # clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    # torch.save(model.state_dict(), model_file)

    model.load_state_dict(torch.load("./trained_models/HNRNPL.pkl", map_location=torch.device('cpu')))
    # model.eval()
    print(motif, channel)
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)

    
    #print 'predicting'         
    #pred = model.predict_proba(test_bags)
    #return model

def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size = window_size, channel = channel, labcounts = 4)
    elif model_type == 'MA1DCNN':
        model = MA1DCNN(A_EAM, A_CAM, width= window_size, in_channel=4)
    else:
        print ('only support CNN, CNN-LSTM, ResNet and DenseNet model')

    if cuda:
        model = model.cuda()

    X_test = X_test.squeeze(1)
                
    model.load_state_dict(torch.load(model_file))
    model.eval()
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred
        
def run_ideepe(parser):
    #data_dir = './GraphProt_CLIP_sequences/'
    posi = parser.posi
    nega = parser.nega
    model_type = parser.model_type
    ensemble = parser.ensemble
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    motif = parser.motif
    motif_outdir = parser.motif_dir
    max_size = parser.maxsize
    channel = parser.channel
    local = parser.local
    window_size = parser.window_size
    ensemble = parser.ensemble
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    glob = parser.glob
    start_time = timeit.default_timer()
    #pdb.set_trace() 
    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return

    motif_seqs = []
    if motif:
      train = True
      local = False
      glob = True
	    #pdb.set_trace()
      data = read_data_file(posi, nega)
      motif_seqs = data['seq']
      if posi == '' or nega == '':
          print('To identify motifs, you need training positive and negative sequences using global CNNs.')
          return
   
    if local:
        #window_size = window_size + 6
        channel = channel
        ensemble = False
        #lotext = 'local'
    elif glob:
        #window_size = maxsize + 6
        channel = 1
        ensemble = False
        window_size = max_size
        #lotext = 'global'
    #if local and ensemble:
    #	ensemble = False
	
    if train:
        if not ensemble:
            train_bags, train_labels = get_data(posi, nega, channel = channel, window_size = window_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = channel, window_size = window_size + 6,
                                         model_file = model_file, batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir) 
        else:
            print( 'ensembling')
            train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = max_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = max_size + 6,
                                  model_file = model_file + '.global', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)
            train_bags, train_labels = [], []
            train_bags, train_labels = get_data(posi, nega, channel = 7, window_size = window_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = window_size + 6,
                                        model_file = model_file + '.local', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)

            
            end_time = timeit.default_timer()
            print ("Training final took: %.2f s" % (end_time - start_time))
    elif predict:
        fw = open(out_file, 'w')
        if not ensemble:
            X_test, X_labels = get_data(testfile, nega = None, channel = channel, window_size = window_size)
            predict = predict_network(model_type, np.array(X_test), channel = channel, window_size = window_size + 6, model_file = model_file, batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        else:
            X_test, X_labels = get_data(testfile, nega = None, channel = 1, window_size = max_size)
            predict1 = predict_network(model_type, np.array(X_test), channel = 1, window_size = max_size + 6, model_file = model_file+ '.global', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            X_test, X_labels = get_data(testfile, nega = None, channel = 7, window_size = window_size)
            predict2 = predict_network(model_type, np.array(X_test), channel = 7, window_size = window_size + 6, model_file = model_file+ '.local', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
                        
            predict = (predict1 + predict2)/2.0
	#pdb.set_trace()
        #auc = roc_auc_score(X_labels, predict)
        #print auc        
        myprob = "\n".join(map(str, predict))  
        fw.write(myprob)
        fw.close()
    else:
        print('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='/content/gdrive/MyDrive/GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.positives.fa', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='/content/gdrive/MyDrive/GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.negatives.fa', help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='CNN', help='it supports the following deep network models: CNN, CNN-LSTM, ResNet and DenseNet, default model is CNN')
    parser.add_argument('--out_file', type=str, default='/content/gdrive/MyDrive/prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--motif', type=bool, default=False, help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The dir used to store the prediction binding motifs.')
    parser.add_argument('--train', type=bool, default=True, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='/content/gdrive/MyDrive/model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--maxsize', type=int, default=160, help='For global sequences, you need specify the maxmimum size to padding all sequences, it is only for global CNNs (default value: 501)') # aaa
    parser.add_argument('--channel', type=int, default=7, help='The number of channels for breaking the entire RNA sequences to multiple subsequences, you can specify this value only for local CNNs (default value: 7)')
    parser.add_argument('--window_size', type=int, default=101, help='The window size used to break the entire sequences when using local CNNs, eahc subsequence has this specified window size, default 101')
    parser.add_argument('--local', type=bool, default=False, help='Only local multiple channel CNNs for local subsequences')
    parser.add_argument('--glob', type=bool, default=False, help='Only global multiple channel CNNs for local subsequences')
    parser.add_argument('--ensemble', type=bool, default=True, help='It runs the ensembling of local and global CNNs, you need specify the maxsize (default 501) for global CNNs and window_size (default: 101) for local CNNs')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=32, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')
    args = parser.parse_args(args=['--n_epochs', '10'])
    # args = parser.parse_args()
    return args

class CNN(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = (maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool2_size = (out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(int(maxpool2_size*nb_filter), int(hidden_size))
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

def same_padding(W, F, S, rate=1):
    out_rows = (W + S - 1) // S
    effective_k_row = (F - 1) * rate + 1
    p = max(0, (out_rows-1)*S+effective_k_row-W)
    return int((p/2.))

class A_CAM(nn.Module):
  def __init__(self, nb_filter, width):
    super(A_CAM, self).__init__()
    self.nb_filter = nb_filter

    # self.avgpool = nn.AvgPool2d()
    self.conv1 = nn.Conv1d(self.nb_filter, int(self.nb_filter/2), 1, padding=same_padding(width, 1,1))
    self.act1 = nn.ReLU()
    self.conv2 = nn.Conv1d(int(self.nb_filter/2), self.nb_filter, 1, padding=same_padding(width, 1,1))
    self.batchnorm = nn.BatchNorm1d(self.nb_filter)
    self.act2 = nn.Sigmoid()

  def forward(self, x):
    c = torch.mean(x, axis=-1)
    c = c.reshape(x.shape[0], self.nb_filter, 1)

    c = self.conv1(c)
    c = self.act1(c)

    c = self.conv2(c)
    c = self.batchnorm(c)
    c = self.act2(c)

    x_out = x * c
    x_out = x_out + x

    return x_out

class A_EAM(nn.Module):
  def __init__(self, nb_filter, kernel_size, width):
    super(A_EAM, self).__init__()
    self.nb_filter = nb_filter
    self.kernel_size = kernel_size

    self.conv1 = nn.Conv1d(self.nb_filter, 1, 1, padding=same_padding(width,1,1))
    self.batchnorm = nn.BatchNorm1d(1)
    self.act1 = nn.Sigmoid()
    self.conv2 = nn.Conv1d(self.nb_filter, self.nb_filter, self.kernel_size, padding=same_padding(width, self.kernel_size,1))
    self.act2 = nn.ReLU()

  def forward(self, x):
    t = self.conv1(x)
    t = self.batchnorm(t)
    t = self.act1(t)

    x_out = self.conv2(x)
    x_out = self.act2(x_out)

    x_out = x_out * t
    x_out = x_out + x

    return x_out

class MA1DCNN(nn.Module):
    def __init__(self, A_EAM_block, A_CAM_block, width=101, in_channel=4, nb_filter=(16,32,64,64,128), num_classes=2, hidden=32, kernel_size=(13,11,9,7,5), stride=(1,2,2,2,4)):
        super(MA1DCNN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.nb_filter = nb_filter

        self.pre_nb_filter = []
        for i in range(len(self.nb_filter)):
            if i == 0:
                self.pre_nb_filter.append(in_channel)
            else:
                self.pre_nb_filter.append(self.nb_filter[i-1])

        self.out_widths = []
        self.out_widths.append(width)

        # self.layers = []
        # for i in range(num_blocks):
        #   self.layers.append(self.make_layer(A_EAM_block, A_CAM_block, self.pre_nb_filter[i], self.nb_filter[i], self.kernel_size[i], self.stride[i], self.out_widths[-1]))
        self.layer1 = self.make_layer(A_EAM_block, A_CAM_block, self.pre_nb_filter[0], self.nb_filter[0], self.kernel_size[0], self.stride[0], self.out_widths[-1])
        self.layer2 = self.make_layer(A_EAM_block, A_CAM_block, self.pre_nb_filter[1], self.nb_filter[1], self.kernel_size[1], self.stride[1], self.out_widths[-1])
        self.layer3 = self.make_layer(A_EAM_block, A_CAM_block, self.pre_nb_filter[2], self.nb_filter[2], self.kernel_size[2], self.stride[2], self.out_widths[-1])
        self.layer4 = self.make_layer(A_EAM_block, A_CAM_block, self.pre_nb_filter[3], self.nb_filter[3], self.kernel_size[3], self.stride[3], self.out_widths[-1])
        self.layer5 = self.make_layer(A_EAM_block, A_CAM_block, self.pre_nb_filter[4], self.nb_filter[4], self.kernel_size[4], self.stride[4], self.out_widths[-1])
        # layers = []
        # for i in range(num_blocks):
        #   layers.append(nn.Sequential(
        #       nn.Conv1d(self.pre_nb_filter[i], self.nb_filter[i], kernel_size=self.kernel_size[i], padding='same', stride=self.stride[i]),
        #       nn.ReLU(),
        #       A_EAM(nb_filter=self.nb_filter[i], kernel_size=self.kernel_size[i]),
        #       A_CAM(nb_filter=self.nb_filter[i])
        #   ))
        # self.layers = layers

        self.cnn = nn.Conv1d(self.nb_filter[-1], self.nb_filter[-1], kernel_size=self.kernel_size[-1], padding=same_padding(self.out_widths[-1], self.kernel_size[-1], self.stride[-1]), stride=self.stride[-1])
        self.act1 = nn.ReLU()
        # self.pooling = nn.AvgPool2d()
        self.fc1 = nn.Linear(self.nb_filter[-1], hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(p=0.25)
        self.act2 = nn.ReLU()
        self.act3 = nn.Softmax(-1)

    def make_layer(self, A_EAM_block, A_CAM_block, in_channel, out_channel, kernel_size, stride, width):
        layers = []
        num = same_padding(width, kernel_size, stride)
        layers.append(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=num, stride=stride))
        out_size = np.ceil((width-kernel_size+2*same_padding(width, kernel_size, stride))/stride+1)
        self.out_widths.append(out_size)
        layers.append(nn.ReLU())
        layers.append(A_EAM_block(nb_filter=out_channel, kernel_size=kernel_size, width=width))
        layers.append(A_CAM_block(nb_filter=out_channel, width=width))
        layers.append(nn.Dropout(p=0.1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # for layer in self.layers:
        #   x = layer(x)
        #   print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.cnn(x)
        x = self.act1(x)
        x = torch.mean(x, -1)

        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.act3(x)

        return x
    
    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    args = parser.parse_args(args=['--glob', 'True', '--model_type', 'MA1DCNN', '--posi', "./data/PPIG_HepG2_rep01.train.positive.fa", '--nega', "./data/PPIG_HepG2_rep01.train.negative.fa", '--n_epochs', '2', '--maxsize', '160', '--motif', 'True', '--motif_dir','./motif/PPIG', '--model_file', './trained_models/PPIG.pkl'])
    # args = parser.parse_args(args=['--glob', 'True', '--model_type', 'MA1DCNN', '--posi', "./data/U2AF2_HepG2_rep01.train.positive.fa", '--nega', "./data/U2AF2_HepG2_rep01.train.negative.fa", '--n_epochs', '2', '--maxsize', '160', '--motif', 'True', '--motif_dir','./motif/U2AF2', '--model_file', './trained_models/U2AF2.pkl'])
    # args = parser.parse_args(args=['--glob', 'True', '--model_type', 'MA1DCNN', '--posi', "./data/HNRNPL_HepG2_rep01.train.positive.fa", '--nega', "./data/HNRNPL_HepG2_rep01.train.negative.fa", '--n_epochs', '2', '--maxsize', '160', '--motif', 'True', '--motif_dir','./motif/HNRNPL', '--model_file', './trained_models/HNRNPL.pkl'])
    run_ideepe(args)


    # python deep.py --ensemble False --glob True --model_type 'CNN' --nega './data/HNRNPL_HepG2_rep01.train.negative.fa' --posi './data/HNRNPL_HepG2_rep01.train.positive.fa' --n_epochs 5 --maxsize 201 --motif True --motif_dir './motif/HNRNPL'
    # docker run -v /Users/arika/Desktop/ideeps/:/home/meme --user `id -u`:`id -g` memesuite/memesuite tomtom -dist pearson -thresh 0.05 -eps -oc ./motif/HNRNPL/tomtom ./motif/HNRNPL/filters_meme.txt ./Ray2013_rbp_RNA.meme
    # ./weblogo/seqlogo -F PNG -f ./data/HNRNPL_HepG2_rep01.train.negative.fa > globin.png
    # weblogo -X NO -Y NO --errorbars NO --fineprint ""  -C "#CB2026" A A -C "#34459C" C C -C "#FBB116" G G -C "#0C8040" T T <filter1_logo.fa >filter1.eps