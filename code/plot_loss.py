import os
import sys
import matplotlib as mpl
mpl.use('Agg') # to enable non ssh -X
import matplotlib.pyplot as plt
#home_dir = "/home/yanlisa/assignment4/log"
import numpy as np
cwd = os.getcwd()

def read_losses(log_fname):
    log_fname = log_fname.split('/')[-1]
    log_f = os.path.join(cwd,'log', log_fname)
    epoch_str = '>>>>'
    loss_str = 'curr loss: '
    train_str = 'Train'
    val_str = 'Val'
    f1_str = 'F1: '
    em_str = 'EM: '
    delim = ','
    norm_str =  'norms: '
    print "fname", log_f
    with open(log_f, 'r') as f:
        log_lines = f.readlines()
        # indices
        epoch_lines = get_inds(epoch_str, log_lines)
        loss_lines = get_inds(loss_str, log_lines)
        train_lines = get_inds(train_str, log_lines)
        val_lines = get_inds(val_str, log_lines)
        # floats
        losses = get_nums(loss_str, loss_lines, log_lines)
        norms = get_nums(norm_str, loss_lines, log_lines, delim='(')
        f1_train = get_nums(f1_str, train_lines, log_lines, delim=delim)
        em_train = get_nums(em_str, train_lines, log_lines, delim=delim)
        f1_val = get_nums(f1_str, val_lines, log_lines, delim=delim)
        em_val = get_nums(em_str, val_lines, log_lines, delim=delim)

    fprefix = log_fname.split('.')[0]
    plot_fig(loss_lines, losses, "loss", fprefix, epoch_lines)
    plot_fig(loss_lines, norms, "global norm", fprefix, epoch_lines)
    plot_acc(train_lines, f1_train, val_lines, f1_val,
            'f1', fprefix, epoch_lines)
    plot_acc(train_lines, em_train, val_lines, em_val,
            'em', fprefix, epoch_lines)

def get_inds(search_str, lines):
    return filter(lambda i: search_str in lines[i], range(len(lines)))

def get_nums(search_str, inds, lines, delim=None):
    if not delim:
        delim = ' '
    num_strs = map(lambda i: lines[i].split(search_str)[1].split(delim)[0].strip(),
        inds)
    return map(float, num_strs)

def plot_fig(x, y, name, prefix, epochs=None):
    if not epochs:
        epochs = []
    min_, max_ = max(0, min(y)-5), max(y)
    if name == 'loss':
        min_ = 0
    fig = plt.figure()
    ax1 = plt.gca()
    ax1.plot(x, y, color='b', label=name)
    #ax1.set_ylim([min_, max_+1])
    for i in epochs:
        ax1.axvline(x=i, color='k')
    #ax1.legend()
    ax1.set_ylabel(name.title())
    ax1.set_xlabel('Iterations')
    dest = '%s_%s.png' % (prefix, name)
    print "Saved %s figure: %s" % (name, dest)
    fig.savefig(dest)

def plot_acc(x_train, y_train, x_val, y_val, name, prefix, epochs=None):
    if not epochs:
        epochs = []
    fig = plt.figure()
    ax1 = plt.gca()
    set_color = ['g', 'r']
    ax1.plot(x_train, 100*np.array(y_train), color=set_color[0],label='Train')
    ax1.plot(x_val, 100*np.array(y_val), color=set_color[1], label='Validation')
    for i in epochs:
        ax1.axvline(x=i, color='k')
    ax1.legend(loc='upper left')
    ax1.set_ylim([0, 100])
    ax1.set_ylabel('%s Accuracy' % name.upper())
    ax1.set_xlabel('Iterations')

    dest = '%s_%s.png' % (prefix, name)
    print "Saved %s figure: %s" % (name, dest)
    fig.savefig(dest)

if __name__ == "__main__":
    read_losses(sys.argv[1])
