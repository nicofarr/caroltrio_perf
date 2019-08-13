"""Analyse audio - ratings"""
# code: camila
# version visualization

import os 
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt 
plt.style.use('default')
import numpy as np 

from sklearn.metrics import silhouette_score,calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances,manhattan_distances,paired_manhattan_distances,euclidean_distances
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,classification_report

import soundfile as sf

def load_audio(name,
               filepath = "./data_clean/"):    
    filepath = "{}{}.wav".format(filepath,name)
    sound, sr = sf.read(filepath,dtype ='float32')      
    if sr != 22050:
        print("Sampling rate dif, making")
        # TODO os.system()
        return    
    return sound, sr

def reduce_number_cluster(labels):
    counter = []
    best = len(np.unique(labels))
    for i in np.unique(labels):
        #count by label 
        counter.append((len(labels[labels==i]))/len(labels))

    df = pd.Series(counter,index=np.unique(labels))
    best = len(df[df > 0.1])
    print("xxxxxxx {}".format(best))    
    return best

def spectral_explore_cluster(matrix,max_cluster=10):
    metrics = []
    all_labels = []
    #Find number cluster    
    number = range(2,max_cluster +1)
    for i in tqdm(number): 
        #evaluate using calinski 
        spectral = SpectralClustering(n_clusters=i, 
                                        eigen_solver='arpack',
                                        affinity="precomputed",
                                        random_state=0,n_jobs = -1).fit((matrix))

        labels = spectral.labels_
        metrics.append(calinski_harabasz_score(matrix, labels))
        all_labels.append(labels)
        # calinski: the higher the better
    df = pd.DataFrame(metrics,index = number, columns = ['calinski'])
    #df.plot(y=['calinski'],subplots=True,sharex=True,figsize=(4,4),linewidth=2)

    ##best_cluster: The index where the metrics is maximum.
    best_cluster = int(df.idxmax())
    print("Best number of cluster is {:d}".format(best_cluster))
    #check dispersion of labels
    best_cluster = reduce_number_cluster(all_labels[best_cluster - 2])
    return best_cluster

def make_cluster(matrix,find,c = 8):
    if find:
        c = spectral_explore_cluster(matrix)
    
    spectral = SpectralClustering(n_clusters = c,
                                  eigen_solver='arpack',
                                affinity="precomputed",
                                random_state=0,n_jobs = -1).fit((matrix))
    labels = spectral.labels_ 
    cal = calinski_harabasz_score(matrix, labels)
    return labels,cal

def get_matrix(focus,subjtime,curnpz):    
    X = np.stack([focus,subjtime]).T
    print(X.shape)   
    #k = spectral_explore_neighbors(X)
    S = manhattan_distances(X)
    S = (np.max(S)-S)/np.max(S)
    return S

def get_values():
    to_time = {}    
    with open('./data_clean/relation_layer_seconds.txt', 'r') as reader:
        for i in reader:
            key,m,b = i.split()
            if key != 'name_layer':
                to_time[key] = [float(m),float(b)]
    return to_time

from matplotlib.gridspec import GridSpec
        
def make_plot(info):
    
    fig = plt.figure(figsize = (14,10))
    gs = GridSpec(2, 3, figure = fig)
    
    ax1 = fig.add_subplot(gs[0, :])
    plot_labels(ax1,info['name'],info['labels_audio'],info['labels'],info['cluster_path'])
    
    ax2 = fig.add_subplot(gs[-1, 0])
    plot_contingency_matrix(ax2,info['labels_audio'],info['labels'])
    
    ax3 = fig.add_subplot(gs[-1:, -2])
    plot_scatter(ax3,info['focus'],info['subjtime'],info['labels'],info['calinksi'])
    
    ax4 = fig.add_subplot(gs[-1, -1]) 
    plot_tsne(ax4,info['curnpz'],info['name'],info['cluster_path'])
  
    plt.show()

from scipy.spatial.distance import euclidean, cityblock

#method to add temporal dependency in scatter plot
def add_direction(ax,x,y,space = 10,lim = 20):
    points = np.arange(0,len(x),space)
    for i,p in enumerate(points[:-1]):  
        
        xy = (x[points[i+1]],y[points[i+1]]) #destin
        xytext = (x[points[i]],y[points[i]]) #origin
        text = p
        if(i == 0): 
            text = '0'
            ax.annotate(text, xy=xy, xytext=xytext,
                    xycoords='data', textcoords='data',bbox=dict(boxstyle="round", fc="1")
                        ,arrowprops=dict(lw=1,facecolor = 'white',alpha = 0.8),size=14)
            
        elif cityblock(xy,xytext) > lim:            
            ax.annotate(text, xy=xy, xytext=xytext,
                        xycoords='data', textcoords='data',bbox=dict(boxstyle="round", fc="0.8"),
                    arrowprops=dict(lw=1,facecolor = 'white',alpha = 0.8),size=10)            
        
from scipy.spatial.distance import euclidean

def plot_scatter(ax,focus,subjtime,labels,calinksi):
    
    classes = np.unique(labels)
    cmap = plt.cm.get_cmap('viridis', len(classes))
    im = ax.scatter(focus,subjtime, c = labels,cmap = cmap)
    ax.figure.colorbar(im, ax=ax,ticks = classes)
    ax.set(xlabel = 'Focus',
            ylabel = 'Subjtime',
            title = 'Cluster subjetive')
    #add direction
    space = int(len(labels)*0.1)
    add_direction(ax,focus,subjtime,25,25)
    return ax

from tsnecuda import TSNE

def plot_tsne(ax,curnpz,name,cluster_path='./data_clean/results/'):
    labels = np.load("{}{}/features_conv_v27.npz".format(cluster_path,name))['labels'][:,1]
    conv  = np.load("{}{}/features_conv_v27.npz".format(cluster_path,name))['feature_vector']
    
    
    cal = calinski_harabasz_score(conv, labels)
     
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10,random_seed=0).fit_transform(conv)
    
    classes = np.unique(labels)
    cmap = plt.cm.get_cmap('brg', len(classes))
    im = ax.scatter(X_embedded[:,0],X_embedded[:,1],c=labels,cmap = cmap)
    ax.figure.colorbar(im, ax=ax,ticks = classes)
    title = "Cluster audio, T-SNE representation"
    ax.set(title = title)
    
    space = int(len(conv)*0.1)
    add_direction(ax,X_embedded[:,0],X_embedded[:,1],space,5)
    return ax

def plot_contingency_matrix(ax,labels_audio,labels,cmap=plt.cm.Blues, normalize = True):
    
    np.set_printoptions(precision=2)
    # Compute contingency matrix
    matrix = contingency_matrix(labels_audio,labels)
    
    cm = np.array([i/np.sum(i)for i in matrix])
    title = 'Normalized contingency matrix'
    
    # Only use the labels that appear in the data
    labels_audio = np.unique(labels_audio) #ylabels
    labels = np.unique(labels) #xlabels
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels_audio,
           title=title,
           ylabel='True label [Audio]',
           xlabel='Predicted label [Subjetive]')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.figure.tight_layout()
    return ax
                      
def plot_labels(ax1,name,labels_audio,labels,cluster_path):
    ARI = adjusted_rand_score(labels_audio,labels)
    AMI = adjusted_mutual_info_score(labels_audio,labels,average_method = 'arithmetic')
    
    classes = np.unique(labels_audio)
    cmap = plt.cm.get_cmap('brg')
    ax1.grid('--')
    ax1.set_xlabel('time (s)')    
#     ax1.tick_params(axis='y',color='b')
    ax1.set(yticks = classes)
      
    ax1.set_ylabel('labels_audio', color = 'slategray')  # we already handled the x-label with ax1
    ax1.plot(labels_audio,linewidth=1.5,c = 'slategray')
             
    ax2 = ax1.twinx()  # 
    ax2.set_ylabel('waveform')
    plot_audio(ax2,labels,name,cluster_path)
    
    ax1.set(title = "{} ARI: {:.4f}, AMI: {:.4f}".format(name,ARI,AMI))
    
def plot_audio(ax,labels,name,cluster_path):
    classes = np.unique(labels)
    cmap = plt.cm.get_cmap('viridis', len(classes))
    segments = write_clustered_segments(labels,name,cluster_path)

    for c,i in enumerate(segments):
        ax.plot(i[:,0],i[:,1],linewidth = 0.6,color = cmap(c))
    return ax

def write_clustered_segments(labels,name,cluster_path = '/data_clean/results/'):   
    
    idlayer = np.load("{}{}/features_conv_v27.npz".format(cluster_path,name))['idlayer']
    
    audio,sr = load_audio(name)
    audio = audio[:len(labels)*sr]
    
    t = (np.array(range(0,len(audio))))/sr
    
    name_layer = "conv{}".format(idlayer + 1)
    m,b = get_values()[name_layer] #get slope (m) and interception (b) for a given layer
    seglength = int((1/m)*sr)
    data_end = []
    for i in range(labels.max()+1): # Loop through all cluster labels
         # fetch all labels with the current value
        ind_curlabel = np.argwhere(labels == i)
        # Calculate the offsets in the original wave file 
        offsets_wave = ind_curlabel * seglength        
        bigwave = []   
        for curoffset in (range(len(offsets_wave))):
            # cut the original wave file and save the excerpt
            curwave = audio[offsets_wave[curoffset][0]:(offsets_wave[curoffset][0]+seglength)]      
            time = t[offsets_wave[curoffset][0]:(offsets_wave[curoffset][0]+seglength)]
            if len(curwave) != 0:
                bigwave.append([time,curwave])    
            else:
                continue
        bigwave = np.hstack(bigwave).T
        data_end.append(bigwave)
    return np.asarray(data_end)
    

    