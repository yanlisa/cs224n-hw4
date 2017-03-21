import os
import sys
import matplotlib as mpl
mpl.use('Agg') # to enable non ssh -X
import matplotlib.pyplot as plt
home_dir = "/home/yanlisa/assignment4/log"

def read_losses(log_fname):
    log_f = os.path.join(home_dir, log_fname)
    epoch_str = '>>>>'
    loss_str = 'curr loss: '
    print "fname", log_f
    with open(log_f, 'r') as f:
        log_lines = f.readlines()
        # indices
        epoch_lines = filter(lambda i: epoch_str in log_lines[i], range(len(log_lines)))
        loss_lines = filter(lambda i: loss_str in log_lines[i], range(len(log_lines)))
        # floats
        losses = map(lambda i: log_lines[i].split(loss_str)[1].split(' ')[0], loss_lines)
        losses = map(float, losses)
        print "losses", losses

    min_loss, max_loss = 0, max(losses)
    fig = plt.figure()
    ax1 = plt.gca()
    ax1.plot(loss_lines, losses, color='b', label='loss')
    for i in epoch_lines:
        ax1.axvline(x=i, color='k')
    ax1.legend()
    dest = '%s.png' % log_fname.strip('.txt')
    fig.savefig(dest)

if __name__ == "__main__":
    read_losses(sys.argv[1])
