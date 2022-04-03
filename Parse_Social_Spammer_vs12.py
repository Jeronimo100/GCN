# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:23:02 2018

@author: gerasimos
"""

from __future__ import print_function
import collections
import numpy as np
import itertools
import os
from os import getcwd
import sys
import math 

import matplotlib.pyplot as plt
import networkx as nx  # nx.__version__

int(math.ceil(5.4)) # math.floor(5.4)
a3 = np.random.choice(range(18),5,replace=False)
a1 = a.tolist()[2:4] # range(5)
a3[2:4]
a = np.random.choice(range(2000), 140+300+1500, replace=False)

G = nx.Graph()
H = nx.path_graph(10)
G.add_nodes_from(H)
G.number_of_nodes()
plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw(H, with_labels=True, font_weight='bold')
G2 = nx.petersen_graph()
plt.subplot(124)
nx.draw_shell(H, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
[len(c) for c in sorted(nx.connected_components(H), key=len, reverse=True)] # print(nx.connected_components(H))
[len(c) for c in sorted(nx.connected_components(G2), key=len, reverse=True)]

#yme node_connected_component(G, n)
###############################################################################
############################## .content archive  ##############################
Features_D = collections.defaultdict(list)

#Official_N_Synonym_SymInter = set() #collections.defaultdict(list)
#Interactors = set()

#Organism_Interactors_D = collections.defaultdict(set)
#Organism_Interactors = set()
#Different_Organism_Interactors_D = collections.defaultdict(set)
#Different_Organism_Interactors = set() #Different_Organism_Interactors_D = collections.defaultdict(set)

#Class_ES_D = collections.defaultdict(list)
#Class_EST_D = collections.defaultdict(list)
with open('data/Social_Spammer_Evolving_Users.csv', 'r') as f: # open the file for reading
  #headers = f.readline().strip().split('\t')
  for count2, line in enumerate(f):
    row = line.strip().split('\t')
    row_Spammer_id = row[0] # keys of the main dictionary
    #row_interactorB_id = row[4]
    # flags
    Features_D = row[7].replace(' ','_') # flags/features
    Official_Symbol_Interactor_B = row[8].replace(' ','_')
    Synonyms_Interactor_A = row[9].split('|')
    Synonyms_Interactor_B = row[10].split('|')
    Official_N_Synonym_SymInter.add(Official_Symbol_Interactor_A)
    Official_N_Synonym_SymInter.add(Official_Symbol_Interactor_B)
    Official_N_Synonym_SymInter_D[row_interactorA_id].add(Official_Symbol_Interactor_A)
    Official_N_Synonym_SymInter_D[row_interactorB_id].add(Official_Symbol_Interactor_B)
    for entry in Synonyms_Interactor_A:
        Official_N_Synonym_SymInter.add(entry)
        Official_N_Synonym_SymInter_D[row_interactorA_id].add(entry)
    for entry in Synonyms_Interactor_B:
        Official_N_Synonym_SymInter.add(entry)
        Official_N_Synonym_SymInter_D[row_interactorB_id].add(entry)
    # node labels/classes
    Organism_Interactor_A = row[15] # labels/classes
    Organism_Interactor_B = row[16]
    Organism_Interactors_D[row_interactorA_id].add(Organism_Interactor_A)
    Organism_Interactors_D[row_interactorB_id].add(Organism_Interactor_B)
    Different_Organism_Interactors.add(Organism_Interactor_A)
    Different_Organism_Interactors.add(Organism_Interactor_B)
     #Different_Organism_Interactors_D[Organism_Interactor_A] = {}
     #Different_Organism_Interactors_D[Organism_Interactor_B] = {}
    #if count2==20000:
    # break 
    #data.append([float(v) for v in values])
  #basic_data = array(data)
with open('data/BIOGRID-ALL-3.4.163.tab2.txt', 'r') as f: # open the file for reading
  headers = f.readline().strip().split('\t')  
  row = f.readline().strip().split('\t')
  row_Synonyms = row[9].split('|')
  for entry in row_Synonyms:
   
#create mappin
maxNoOfficialSymbols = 0
minNoOfficialSymbols = 1
for row_Interac in Official_Symbol_Interactors_D:
   temp = len(Official_Symbol_Interactors_D[row_Interac])
   if temp>maxNoOfficialSymbols:
     maxNoOfficialSymbols = temp
   if temp<minNoOfficialSymbols:
     minNoOfficialSymbols = temp
#elegxos
#assert max(map(len, Organism_Interactors_D.values())) == 1
for row_interactor_id in Organism_Interactors_D:
  assert len(Organism_Interactors_D[row_interactor_id]) == 1
  Organism_Interactors_D[row_interactor_id] = list(Organism_Interactors_D[row_interactor_id])[0]
  

print('Number of distinct Official Symbol Interactors: %d' % len(Official_N_Synonym_SymInter))#Official_Symbol_Interactors))
print('Number of distinct Organism_Interactors: %d' % len(Different_Organism_Interactors))
print('Length of Interactors Dictionary: %d' % len(Official_N_Synonym_SymInter_D))#Official_Symbol_Interactors_D))
Official_Symbol_Interactors = sorted(list(Official_Symbol_Interactors)) # Turn it into a sorted list in order to create the 0/1 flag matrix
count3 = 0
OffSymInterMap = collections.defaultdict(list)
for p in Official_N_Synonym_SymInter:
#for p in papersD:
    OffSymInterMap[p] = count3
    count3 = count3+1
#idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#idx_map = {} j: i for i, j in enumerate(idx)}
###############################################################################
########################## .content archive reduced ###########################
with open('data/BIOGRID.content', 'w') as output_file1:
  for node_as_entry in Official_N_Synonym_SymInter_D:
    #flags = OffSymInterMap[Official_Symbol_Interactors_D[node_as_entry]]
    #flags = '\t'.join(Official_Symbol_Interactors_D[node_as_entry])
    flags = '\t'.join(['1' if Official_Interactor in Official_N_Synonym_SymInter_D[node_as_entry] else '0' for Official_Interactor in Official_N_Synonym_SymInter])
    print(node_as_entry, flags, Organism_Interactors_D[node_as_entry], file=output_file1, sep='\t')
###############################################################################
###################### .content archive further reduced #######################
count_content_rows = 0
with open('data/BIOGRID.content', 'w') as output_file1:
  for node_as_entry in Official_Symbol_Interactors_D:
   count_content_rows += 1
   if count_content_rows >= 2500 and count_content_rows <= 6500:
    flags = '\t'.join(['1' if Official_Interactor in Official_Symbol_Interactors_D[node_as_entry] else '0' for Official_Interactor in Official_Symbol_Interactors])
    print(node_as_entry, flags, Organism_Interactors_D[node_as_entry], file=output_file1, sep='\t')
    Interactors.add(node_as_entry) # print(len(Interactors))
    """if count_content_rows == 2500:
     break """
###############################################################################
############################### .cites archive  ###############################
with open('data/BIOGRID-ALL-3.4.163.tab2.txt', 'r') as f: # open the file for reading
  with open('data/BIOGRID.cites', 'w') as output_file1:
    headers = f.readline().strip().split('\t')
    for count, line in enumerate(f):
      row = line.strip().split('\t') #.rstrip(" ") #print(row_author_id[0])
      row_interactorA_id = row[3]
      #InteractorA_D[row_interactorA_id].append(row_paper_id)        
      row_interactorB_id = row[4] #.rstrip(" ") #print(row_author_id[0])
      if row_interactorA_id in Interactors and row_interactorB_id in Interactors:
       print("{}\t{}\n".format(row_interactorA_id, row_interactorB_id),file=output_file1, end='')
      """print('{}\t{}\n'.format(row_interactorA_id, row_interactorB_id),file=output_file1, end='')"""
      #print('{} '\t' {}'.format(row_interactorA_id, row_interactorB_id))
      #print(row_interactorA_id) # print(line)
      #print(row_interactorB_id) # print(line)
      #break
      #data.append([float(v) for v in values])
#basic_data = array(data)

###############################################################################
################################# SIZE COUNT ##################################
###############################################################################
def utf8lengt(s):
    return len(s.encode('utf-8'))
#line_strin_sizes = []
count_str_size = 0 
with open('data/BIOGRID.content', 'r') as f:
  for line in f:
   #count_str_size += sum(map(utf8lengt,line.split('\t')))   
   count_str_size += utf8lengt(line) #PRODUCES LIST: map(utf8lengt,line) # sum(map(utf8lengt,line.split('\t')))
  """for i in range(len(line.split('\t'))):
    #line_strin_sizes.append(sys.getsizeof(line.split('\t')[i]))
    count_str_size = count_str_size + len(line.split('\t')[i].encode('utf-8')) # sys.getsizeof(line.split('\t')[i])"""
  print(count_str_size)
  float(count_str_size) / 1024 / 1024 / 1024
###############################################################################  
###############################################################################  
  for node_as_entry in Official_Symbol_Interactors_D:
    flags = '\t'.join(['1' if Official_Interactor in Official_Symbol_Interactors_D[node_as_entry] else '0' for Official_Interactor in Official_Symbol_Interactors])
    print(node_as_entry, flags, Organism_Interactors_D[node_as_entry], file=output_file1, sep='\t')
###############################################################################
###############################################################################
###############################################################################
with open('data/BIOGRID.cites', 'w') as output_file1:
    for line in f:
      row = line.strip().split('\t') #.rstrip(" ") #print(row_author_id[0])
      