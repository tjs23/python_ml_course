import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from umap import UMAP

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot_training_history(*histories):
    cmap = plt.get_cmap('tab10')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_options = {'linewidth':2, 'alpha':0.5} # A dictionary of inputs for all charts
    
    for i, history in enumerate(histories):
        hd = history.history
        n = np.arange(len(hd['loss'])) + 1
        plot_options['color'] = cmap(float(i % 10)/10)
        
        for key in hd:
            linestyle = '-' if key.startswith('val_') else '--'
            label = 'Test' if key.startswith('val_') else 'Train'
            
            if key.endswith('loss'):
                ax1.plot(n, hd[key], label=f'{label} {i}', linestyle=linestyle, **plot_options)
                ax1.set_title('Loss')

            else:
                ax2.plot(n, hd[key], label=f'{label} {key.replace("val_","")} {i}',
                         linestyle=linestyle, **plot_options)
                ax2.set_title('Metrics')

    ax1.legend()    
    ax2.legend()
    plt.show()



KLASSES = ['GOLGI', 'CYTOSOL', 'NUCLEUS', 'PM', 'ER', 'PLASTID', 'MT',
           'PEROXISOME', 'TGN', 'VACUOLE', 'RIBOSOMAL', 'UNDEF']

KLASS_COLORS = {'GOLGI': '#FFD000', 'CYTOSOL': '#FF0000',
                'NUCLEUS': '#80B0F0', 'PM': '#FF8888',
                'ER': '#0050FF', 'PLASTID': '#50FF20',
                'MT': '#FF7000', 'PEROXISOME': '#FF00FF',
                'TGN': '#B080FF', 'VACUOLE': '#00A000',
                'RIBOSOMAL': '#7020FF', 'UNDEF': '#BBBBBB'}


DIGIT_COLORS = ['#000000','#BB0000','#FF7000','#CCCC00','#008000',
                '#0080FF','#000080','#A080FF','#DD00DD','#FFA0A0']

def load_proteomics_data(data_file, max_zeros=40):

    classifications = []
    profile_data = []
    
    with open(data_file) as file_obj:
        for line in file_obj:
            pid, klass, *vals = line.strip('\n').split('\t')
                
            row = np.array([float(x) for x in vals])
 
            if np.sum(row == 0) > max_zeros:
                continue
 
            profile_data.append(row)
            classifications.append(KLASSES.index(klass))
    
    return np.array(profile_data), np.array(classifications)


def load_sequences(seq_file = 'seqs.fasta'):

    seq_file = 'seqs.fasta'
    seqs = []

    with open(seq_file) as file_obj:
        for line in file_obj:
            if line[0] != '>':
              seqs.append(list(line.strip()))

    return seqs


def colorlist(*colors):

   return LinearSegmentedColormap.from_list(name='CMAP', colors=list(colors), N=10*len(colors))


def plot_scatters(data_sets, s=4, alpha=0.5, cmap=None, **kw):
    
    n = len(data_sets)
    
    if cmap is None:
       cmap = ['#0080FF','#FF0000','#B0B000']
    
    if isinstance(cmap, (list, tuple)):
       cmap = LinearSegmentedColormap.from_list(name='CMAP', colors=list(cmap), N=n)
        
    fig, axarr = plt.subplots(1, n, squeeze=False, sharex=True, sharey=True)
    fig.set_size_inches(5*n,5)
    
    kw['alpha'] = alpha
    kw['s'] = s
    
    for i in range(n):
        data, labels, title = data_sets[i]
        axarr[0,i].scatter(*data.T, c=labels, cmap=cmap, **kw)
        axarr[0,i].set_title(title)
    
     
def plot_proj2d(data, labels, categories=None, prev_model=None, ax=None, title=None, method=UMAP, method_args=(),
                cmap=None, colorbar=False):
    
    if not cmap:
       cmap = LinearSegmentedColormap.from_list(name='CMAP01', colors=['#FF0000', '#0080FF'], N=25)
    
    method_name = method.__name__
    
    if prev_model:
        out_model = None
    
    else: 
        prev_model = method(n_components=2, *method_args)
        prev_model.fit(data)
        out_model = prev_model
           
    umap_proj = prev_model.transform(data)
    
    x_vals, y_vals = umap_proj.T
    
    if not ax:
        fig, ax = plt.subplots()
        fig.set_size_inches(8,8)
    
    if isinstance(categories, str):
        sc = ax.scatter(x_vals, y_vals, s=3, c=labels, cmap=cmap, alpha=0.5)       
        cb = ax.get_figure().colorbar(sc, ax=ax)
        cb.set_label(categories)
    
    else:
        n = len(categories)-1
 
        for i, label in enumerate(categories):
           mask = labels == i
           ax.scatter(x_vals[mask], y_vals[mask], s=3, color=cmap(i/n), label=categories[i])
        
        ax.legend()
        
    if title:
        ax.set_title(title)
    else:
        ax.set_title(method_name)

    ax.set_xlabel(f'{method_name} 1')
    ax.set_ylabel(f'{method_name} 2')
  
    return out_model


def plot_umap(*args, **kw):
    kw['method'] = UMAP
    plot_proj2d(*args, **kw)

    
def plot_proteomics_data(profile_data, classifications, title=None):
    
    colors = [KLASS_COLORS[k] for k in KLASSES]
    n_klasses = len(KLASSES)
    
    cmap1 = LinearSegmentedColormap.from_list(name='CMAP01', colors=['#FF0000', '#0080FF'], N=25)
    cmap2 = LinearSegmentedColormap.from_list(name='CMAP02', colors=colors, N=len(colors))
    
    umap_model = UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric='correlation').fit(profile_data)
    umap_2d = umap_model.transform(profile_data)
    
    plt.rcParams["figure.figsize"] = (21,8) # Set plot size
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    if title:
        ax1.set_title(title)

    nz = np.count_nonzero(profile_data, axis=1)

    x_vals, y_vals = umap_2d.T
    sc = ax1.scatter(x_vals, y_vals, s=5, cmap=cmap2, c=classifications)
    cb = fig.colorbar(sc, ax=ax1)
    cb.ax.set_yticks(np.linspace(0.5, 10.5, n_klasses))
    cb.ax.set_yticklabels(KLASSES)
 
    x_vals, y_vals = umap_2d.T
    sc = ax2.scatter(x_vals, y_vals, s=5, cmap=cmap1, c=nz)
    cb2 = fig.colorbar(sc, ax=ax2)
    cb2.set_label('Non-zero count')
    
    return
    
    
def matrix_dendrogram(fig, data, labels, metric='euclidean'):

    from scipy.spatial import distance
    from scipy.cluster import hierarchy
    
    dmat = distance.squareform(distance.pdist(data, metric=metric))

    ax1 = fig.add_axes([0.05, 0.05, 0.50, 0.50]) # Fractional x, y, w, h
    ax2 = fig.add_axes([0.56, 0.05, 0.20, 0.50])

    linkage = hierarchy.linkage(data, method='average', metric=metric)
    order = hierarchy.leaves_list(linkage)[::-1]
    n = len(data)
    c = len(DIGIT_COLORS)
    
    link_cols = {}
    for i, (x, y) in enumerate(linkage[:,:2].astype(int)):
        c1 = DIGIT_COLORS[labels[x] % c] if x < n else link_cols[x]
        c2 = DIGIT_COLORS[labels[y] %c ] if y < n else link_cols[y]
        link_cols[i+n] = c1 if c1 == c2 else '#BBBBBB'

    ax1.matshow(dmat[order][:,order], cmap='Greys', origin='upper', aspect='auto')
    ax2.set_title('Labels & Distance dendrogram')

    ax1.spines[['top','right', 'bottom', 'left']].set_visible(False)
    ax2.spines[['top','right', 'bottom', 'left']].set_visible(False)
 
    ax2.set_xticks([])
    ax2.set_xticklabels([])

    with plt.rc_context({'lines.linewidth': 0.8}):
        d = hierarchy.dendrogram(linkage, orientation='right', truncate_mode='level',
                                 above_threshold_color='k', p=20, no_labels=True,
                                 color_threshold=35, ax=ax2, distance_sort=True,
                                 link_color_func=lambda j: link_cols[j])
