import torch
import copy, os, pdb, random, shutil, subprocess, time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from nn_models.conv_net_2_layer import ConvNet2 

weblogo_opts = '-X YES -Y YES --errorbars NO --fineprint ""'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'


################################################################################
# main
################################################################################
def main():

    model_file = 'old_format_pytorch_save/model_nn_epoch_60.pt'
    test_seq_file = 'old_format_pytorch_save/Seq_masked_wo_target_test.pt'
    test_target_file = 'old_format_pytorch_save/Target_C02M02_masked_arcsinh_test.pt'

    test_seq_in = torch.load(test_seq_file)
    test_targets = torch.load(test_target_file)
    np.random.seed(seed=1)
    sample_i = np.array(random.sample(range(test_seq_in.shape[0]), 1000))
    
    test_seq_in = test_seq_in[sample_i]
    test_targets = test_targets[sample_i]

    seqs = onehot2dna(test_seq_in)

    # initialize model
    neural_net = ConvNet2(input_len  = 804, 
             num_channels_c1 = 75, 
             num_channels_c2 = 50, 
             conv_kernel_size_nts_c1 = 11, 
             conv_kernel_size_nts_c2 = 7,
             conv_kernel_stride = 4, 
             pool_kernel_size_c1 = 3, 
             pool_kernel_size_c2 = 2,
             h1_size = 500, 
             #h2_size, 
             dropout_p = 0.5)

    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

    # Restore model state
    neural_net.load_state_dict(checkpoint['model_state_dict'])

    num_filters = neural_net.conv1.weight.shape[0] # shape is three dimensional: [number_of_filters, 1, filter_size in one_hot_enconding not nucleotides]
    filter_size = neural_net.conv1.weight.shape[2]

    print(num_filters)
    print(filter_size)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    activation = {}

    # this function gets the representation of the data after it has gone through the first conv layer 
    def get_activation(name):       
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook 

    neural_net.conv1.register_forward_hook(get_activation("conv1"))

    with torch.no_grad():
        neural_net.eval() 
        predictions = neural_net(test_seq_in.to(device))
        filter_reprs = activation["conv1"]
        #print(filter_reprs)
        #print(filter_reprs.shape) # reprs shape structure [number of rows in input data - so windows for us, number of filters, conv1 layer output size]

    filter_weights = neural_net.conv1.weight.detach().numpy()
    #print(filter_weights)
    #print(filter_weights.shape)
    reformated_filter_weights = reformat_filter_weights(filter_weights)
    

    # also save information contents
    filters_ic = []
    meme_out = meme_intro('motifs_out/filters_meme.txt', seqs)

    for f in range(num_filters):
        print(f'Filter {f}')

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_reprs[:,f,:], int(filter_size/4), seqs, f'motifs_out/filter{f}_logo', maxpct_t = 0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm(f'motifs_out/filter{f}_logo.fa')

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites)

    meme_out.close()

    
    #################################################################
    # annotate filters
    #################################################################
    # run tomtom
    subprocess.call('tomtom -dist pearson -thresh 0.1 -oc motifs_out/tomtom motifs_out/filters_meme.txt Homo_sapiens.meme', shell=True)

    # read in annotations
    filter_names = name_filters(num_filters, 'motifs_out/tomtom/tomtom.txt', 'Homo_sapiens.meme')

    #################################################################
    # print a table of information
    #################################################################
    table_out = open('motifs_out/table.txt', 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print('%3s  %19s  %10s  %5s  %6s  %6s' % header_cols, file=table_out)

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(reformated_filter_weights[f,:,:])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean = filter_reprs[:,f,:].mean()
        fstd = filter_reprs[:,f,:].std()

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols, file=table_out)
        
    table_out.close()
        

def onehot2dna(data):
    data_ar = data.numpy()
    seqs = []
    for row in data_ar:
        pos_start = 0
        dna_seq = []
        while pos_start < len(row):
            quartet = row[pos_start:pos_start + 4]
            if int(quartet[0]) == 1: 
                dna_seq.append("A")
            elif int(quartet[1]) == 1:
                dna_seq.append("C")
            elif int(quartet[2]) == 1:
                dna_seq.append("G")
            elif int(quartet[3]) == 1:
                dna_seq.append("T")
            else:
                print('Malformed sequence in data')
            pos_start += 4
        if len(dna_seq) != 201: 
            print(f"The created dna sequence for window {dna_seq} does not have length 201")
        seqs.append("".join(dna_seq)) 
    return seqs         


def reformat_filter_weights(filter_weights):
    filter_weights = filter_weights.squeeze()
    reformated_weights = np.empty(shape=[filter_weights.shape[0], 4, int(filter_weights.shape[1]/4)])
    for i in range(0, filter_weights.shape[0]):
        row = filter_weights[i,:]
        for j in range(0, int(filter_weights.shape[1]/4)):
            quartet = row[j*4:(j*4) + 4]
            reformated_weights[i, :, j] = quartet
            
    return reformated_weights


def get_motif_proteins(meme_db_file):
    motif_protein = {}
    for line in open(meme_db_file):
        a = line.split()
        if len(a) > 0 and a[0] == 'MOTIF':
            if a[2][0] == '(':
                motif_protein[a[1]] = a[2][1:a[2].find(')')]
            else:
                motif_protein[a[1]] = a[2]
    return motif_protein


def info_content(pwm):
    pseudoc = 1e-9
    bg_pwm = [1-0.415, 0.415, 0.415, 1-0.415]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic


def make_filter_pwm(filter_fasta):
    nts = {'A':0, 'C':1, 'G':2, 'T':3}
    pwm_counts = []
    nsites = 4 
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


def meme_add(meme_out, f, filter_pwm, nsites):
    ic_start = 0
    ic_end = filter_pwm.shape[0]-1

    if ic_start < ic_end:
        print(f'MOTIF filter{f}', file=meme_out)
        print(f'letter-probability matrix: alength= 4 w= {ic_end-ic_start+1} nsites= {nsites}', file=meme_out)

        for i in range(ic_start, ic_end+1):
            print('%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]), file=meme_out)
        print('', file=meme_out)


def meme_intro(meme_file, seqs):
    nts = {'A':0, 'C':1, 'G':2, 'T':3}

    # count
    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            nt_counts[nts[nt]] += 1

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    # open file for writing
    meme_out = open(meme_file, 'w')

    print('ALPHABET= ACGT', file=meme_out)
    print('', file=meme_out)
    print('Background letter frequencies:', file=meme_out)
    print('A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs), file=meme_out)
    print('', file=meme_out)
    
    return meme_out


def name_filters(num_filters, tomtom_file, meme_db_file):
    # name by number
    filter_names = [f'f{fi}' for fi in range(num_filters)]

    # name by protein
    if tomtom_file is not None and meme_db_file is not None:
        motif_protein = get_motif_proteins(meme_db_file)

        # hash motifs and q-value's by filter
        filter_motifs = {}

        tt_in = open(tomtom_file)
        tt_in.readline()
        for line in tt_in:
            a = line.split()
            fi = int(a[0][6:])
            motif_id = a[1]
            qval = float(a[5])

            filter_motifs.setdefault(fi,[]).append((qval,motif_id))

        tt_in.close()

        # assign filter's best match
        for fi in filter_motifs:
            top_motif = sorted(filter_motifs[fi])[0][1]
            filter_names[fi] += '_%s' % motif_protein[top_motif]

    return np.array(filter_names)


def filter_motif(param_matrix):
    nts = 'ACGT'

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



def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None):
    if maxpct_t:
        all_outs = np.ravel(filter_outs)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open(f'{out_prefix}.fa', 'w')
    filter_count = 0
    for i in range(filter_outs.shape[0]):
        for j in range(0, filter_outs.shape[1]-0):
            if filter_outs[i,j] > raw_t:
                kmer = seqs[i][j:j+filter_size]
                print(f'>{i}_{j}', file=filter_fasta_out)
                print(kmer, file=filter_fasta_out)
                filter_count += 1
    filter_fasta_out.close()

    # make weblogo
    if filter_count > 0:
        weblogo_cmd = f'weblogo {weblogo_opts} --format png < {out_prefix}.fa > {out_prefix}.png'
        subprocess.call(weblogo_cmd, shell=True)




if __name__ == '__main__':
    main()
