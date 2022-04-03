# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:13:11 2018

@author: gerasimos
"""

from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import sys
#from kegra.layers.graph import GraphConvolution
import time
import math
from keras.models import model_from_json
#from kegra.utils import *

#########################################################################################################
#from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import scipy.sparse as sp #import scipy.sparse
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

import keras.backend as K


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
##########################################################################################################
#from __future__ import print_function

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/Cora_Samples/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading from {}'.format(path)+'the {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    #print(len(idx_features_labels[:,0]))
    #exit(0)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    #print(np.shape(labels))
    #exit()

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    #print(features[0])
    #exit(0)
    return features.todense(), adj, labels, edges, features, idx_features_labels[:,0]
    #return features, adj, labels, edges, features
    #                         nodes,labels,edges,features 

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


    #y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(math.floor(train_percentage*nodescount), math.floor(val_percentage*nodescount), math.floor(test_percentage*nodescount), y)
def get_splits(train_split_size, val_split_size, test_split_size, y):
    #print(nodescount)
    #exit(0)
    indices_for_splitting = np.random.choice(range(nodescount), train_split_size+val_split_size+test_split_size, replace=False)
				#                                            140 + 300 + 1000
    all_indices = indices_for_splitting.tolist()
    #print(type(all_indices))
    idx_train = all_indices[0:train_split_size-1] # np.random.choice(range(nodescount), train_split_size, replace=False) #range(140)
    idx_val = all_indices[train_split_size:train_split_size+val_split_size-1] # range(200, 500)
    idx_test = all_indices[train_split_size+val_split_size:train_split_size+val_split_size+test_split_size] # range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    #np.set_printoptions(threshold=sys.maxint)   
    #print(y_test)
    #exit(0)
    train_mask = sample_mask(idx_train, y.shape[0])

    np.set_printoptions(threshold=sys.maxint)   
    #print(y)
    #print(np.shape(y))
    #exit()
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
###############################################################################
###############################################################################    
# DEFINE PARAMETRES
DATASET = 'cora'
path="data/cora/"
dataset="cora"
FILTER = 'localpool'  # 'chebyshev'  # print('{}'.format(FILTER))
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200 #1000 # 200
PATIENCE = 1000 #1000 # early stopping patience
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))
# arg2= 600 # sys.argv[1] 
OverlapNo = int(float(sys.argv[1])) # 7 #10 # This is the OVERLAP # print("OverlapNo={}".format(OverlapNo))
#index_no_of_file=int(float(arg1))
Aggreg_Data_Size=int(float(sys.argv[2]))
Sample_Size=int(float(sys.argv[3]))
k = Sample_Size # 50 # 30 # This is the size of the data piece # batch
train_portion = 0.05
val_portion = 0.11
test_portion = 0.37
##############################################################################
##############################################################################
TotSampNu = OverlapNo*int((Aggreg_Data_Size-1) / k + 1 ) # iteratNo = 50*int(nodescount / k + 1 ) # N / k 
##############################################################################
##############################################################################
##############################################################################
##############################################################################
########################### READ MOTHER GRAPH ################################
X0, A0, y0, edges0, features0, mother_G_nodes_ids = load_data(path="data/cora/")
if FILTER == 'localpool':
  """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
  print('Using local pooling filters...')
  A0_ = preprocess_adj(A0, SYM_NORM)
  support = 1
  graph0 = [X0, A0_]
  G0 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
elif FILTER == 'chebyshev':
  """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
  print('Using Chebyshev polynomial basis filters...')
  L = normalized_laplacian(A0, SYM_NORM)
  L_scaled = rescale_laplacian(L)
  T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
  support = MAX_DEGREE + 1
  graph0 = [X0]+T_k
  G0 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
else:
  raise Exception('Invalid filter type.')

############# CONSTRUCT DICTIONARY FOR MOTHER GRAPH'S NODES IDs ##############
import collections
Nodes_Dict = collections.defaultdict(int)
i=0
#print(mother_G_nodes_ids)
for node_id in mother_G_nodes_ids:
 Nodes_Dict[node_id]=i
 i+=1
#print(Nodes_Dict[node_id])
#exit(0)
####################### END OF MOTHER GRAPH PROCESSING #######################
##############################################################################
########################### READ FIRST FILE GRAPH ############################
newfile = ''.join(['CORA_SubFile',str(1)]) 
#newfileCites = ''.join(['CORA_SubFile',str(i+1),'.cites']) #'data/BIOGRID.cites'
# print(newfile)
#return features.todense(), adj, labels, edges, features, idx_features_labels[:,0]
X, A, y, edges, features, mother_G_nodes_ids = load_data(dataset=newfile)  
# Gerasimos mini batching: We need to chop the ADJacency matrix & the X INPUT matrix
nodescount = np.shape(A)[0] # This is N number of nodes/rows/columns # dim1 = dim[0]
featurescount = np.shape(X)[1]
labelscount = np.shape(y)[1]

y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(int(math.floor(train_portion*nodescount)), int(math.floor(val_portion*nodescount)), int(math.floor(test_portion*nodescount)), y0) 
#y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(train_percentage*nodescount/100, val_percentage*nodescount/100, test_percentage*nodescount/100, y)
##y_train0, y_val0, y_test0, idx_train0, idx_val0, idx_test0, train_mask0 = get_splits(y0)
#X, A, y, features, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = X0, A0, y0, features0, y_train0, y_val0, y_test0, idx_train0, idx_val0, idx_test0, train_mask0
#X = X.todense()
X /= X.sum(1).reshape(-1, 1)
graph = [X, preprocess_adj(A, SYM_NORM)]

if FILTER == 'localpool':
  """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
  print('Using local pooling filters...')
  A_ = preprocess_adj(A, SYM_NORM)
  support = 1
  graph = [X, A_]
  G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
elif FILTER == 'chebyshev':
  """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
  print('Using Chebyshev polynomial basis filters...')
  L = normalized_laplacian(A, SYM_NORM)
  L_scaled = rescale_laplacian(L)
  T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
  support = MAX_DEGREE + 1
  graph = [X]+T_k
  G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
else:
  raise Exception('Invalid filter type.')
##############################################################################
print('I will build the layers now')
X_in = Input(shape=(X.shape[1],))
######################## Define model architecture ###########################
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
LayersNum = 2
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
#########################################################################################    
#H = Dropout(0.5)(H)
#H = GraphConvolution(500, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
###############################################################################
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)    
# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
model.summary()
##############################################################################
##############################################################################
##############################################################################
unlabelled_nodes = []
unlabelled_nodes_idx = []
##############################################################################
##############################################################################
##############################################################################
with open('results_CORA_NODESno_EDGESno_FEATURESno_LABELSno_LOADt_TRAINt_TESTt_COST_ACCURACY_DataSize_BatchNo_IterNo_LayersNo.txt', 'a') as output_file:
 #with open('results_CORA_NODESno_EDGESno_FEATURESno_LABELSno_LOADt_TRAINt_TESTt_BatchSize_COST_ACCURACY.txt', 'a') as output_file:
 for i in range(TotSampNu): #range(len(ListListsSubgrNodes)) # range(m-1):
    #newfileNodes = ''.join(['data/Cora_Samples/CORA_SubFile',str(i+1),'.node']) #'data/BIOGRID.content'  
    newfile = ''.join(['CORA_SubFile',str(i+1)]) 
    #newfileCites = ''.join(['CORA_SubFile',str(i+1),'.cites']) #'data/BIOGRID.cites'
    print(newfile)
    ##########################################################################

    # Get data
    #X, A, y, edges, features = load_data(dataset=DATASET)  
                            # features.todense(), adj, labels, edges, features
    X, A, y, edges, features, nodes_id = load_data(dataset=newfile)  
    # NOT HERE:Gerasimos mini batching: We need to chop the ADJacency matrix & the X INPUT matrix
    nodescount = np.shape(A)[0] # This is N number of nodes/rows/columns # dim1 = dim[0]
    featurescount = np.shape(X)[1]
    labelscount = np.shape(y)[1]

    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(int(math.floor(train_portion*nodescount)), int(math.floor(val_portion*nodescount)), int(math.floor(test_portion*nodescount)), y)
    ######################### ACCUMMULATE UNLABELLED NODES #######################
    unlabelled_nodes.append(y_test)
    unlabelled_nodes_idx.append(idx_test) 
    ##############################################################################
    #y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(train_percentage*nodescount/100, val_percentage*nodescount/100, test_percentage*nodescount/100, y)
    ##y_train0, y_val0, y_test0, idx_train0, idx_val0, idx_test0, train_mask0 = get_splits(y0)
    #X, A, y, features, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = X0, A0, y0, features0, y_train0, y_val0, y_test0, idx_train0, idx_val0, idx_test0, train_mask0
    #X = X.todense()
    X /= X.sum(1).reshape(-1, 1)
    #print(X[0])
    #exit(0)
    #graph = [X, preprocess_adj(A, SYM_NORM)]

    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
	print('Using local pooling filters...')
	A_ = preprocess_adj(A, SYM_NORM)
	support = 1
	graph = [X, A_]
	G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

    elif FILTER == 'chebyshev':
	""" Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
	L = normalized_laplacian(adj, SYM_NORM)
	L_scaled = rescale_laplacian(L)
	T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
	support = MAX_DEGREE + 1
	graph = [X]+T_k
	G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

    else:
	raise Exception('Invalid filter type.')
    

    #t20 = time.time()  #t21 = t20 # time.time()
    #print('I have just batched the loaded data for the first time and this lasted for {} seconds'.format(t21-t20))
    
    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999
    ##############################################################################
    ##############################################################################
    ################### GERASIMOS 's BATCHING 1 ENDS HERE #####################
    #print('Dataset has {} nodes, {} edges, {} features, {} labels, '.format(A.shape[0],edges.shape[0],features.shape[1],y.shape[1]),file=output_file,end='')
    print('{} nodes, {} edges, {} features, {} labels, '.format(A.shape[0],edges.shape[0],features.shape[1],y.shape[1]),file=output_file,end='')
    # Prepare batch indices
    t3 = time.time()
    print('I am the train.py archive and I am starting to train at {}'.format(t3))
    #print('I am the train.py archive and I am starting to train at {}'.format(t3), file=output_file)
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    # Fit
    with open('Time_Measurements_results_CORA_NODESno_EDGESno_FEATURESno_LABELSno_LOADt_TRAINt_TESTt_COST_ACCURACY_DataSize_BatchNo_IterNo_LayersNo.txt', 'a') as output_file2:
      print('{} nodes, {} edges, {} features, {} labels, '.format(A.shape[0],edges.shape[0],features.shape[1],y.shape[1]),file=output_file2,end='')
      print("OverlapNo={}, BatchSize={}, SamplesNo={}, LayersNo={}\n".format(k,1,TotSampNu,LayersNum), file=output_file2,end='')
      t9 = time.time()
      for epoch in range(1, NB_EPOCH+1):
    
        # Log wall-clock time
        t = time.time()
        
        #for i in range(TotSampNu):
        t20 = time.time()
        t21 = t20
        t22 = t21 # time.time()
        model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
        t23 = time.time()
        print('Model fitting: {} sec. \n'.format(t23-t22), file=output_file2, end='')
        #print('I have just performed a model fitting with this sample and this lasted for {} seconds\n'.format(t23-t22))

        # Should predict on full dataset, but here not.
        preds = model.predict(graph, batch_size=A.shape[0])    
        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))
	
	print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}\n".format(time.time() - t), file=output_file2, end='')
    
        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1
    t10 = time.time()

 ####################################################################################
 ####################################################################################
 ####################################################################################
 ################################ Testing ###########################################
 ################## PROBABLY THIS GETS APPLIED ON THE GLOBAL GRAPH ##################
 preds = model.predict(graph0, batch_size=A0.shape[0])
 print(preds)
 ####################################################################################
 #### Prepare the accummulated vectors coming from the training y_test, idx_test ####
 
 test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
 ####################################################################################
 print("Test set results:",
  "loss= {:.4f}".format(test_loss[0]),
  "accuracy= {:.4f}".format(test_acc[0]))
 print("loss= {:.4f}".format(test_loss[0]),"accuracy= {:.4f}, TrainT={:.4f}, OverlapNo={}, BatchSize={}, Overlap={}, SamplesNo={}, LayersNo={}\n".format(test_acc[0],t10-t9,k,1,OverlapNo, TotSampNu,LayersNum), file=output_file,end='')


