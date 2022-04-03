# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:06:37 2018

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
import json

import matplotlib.pyplot as plt
import networkx as nx  # nx.__version__

Content_File_with_FtrVecs_Dicted = collections.defaultdict(list) #A = dict()
#Official_Symbol_Interactors = set() #collections.defaultdict(list)
FtrVecs = set()

#Class_ES_D = collections.defaultdict(list)
#Class_EST_D = collections.defaultdict(list)
""" INPUT THE INPUT FILE AS A DICT """
with open('data/cora/cora.content', 'r') as f:
  for count2, line in enumerate(f):
    row = line.strip().split('\t')
    Content_File_with_FtrVecs_Dicted[row[0]] = line.strip()
                          # json.dumps(row)
                          # json.loads(Content_File_with_FtrVecs_Dicted['...'])
#count5 = 0 
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ##################### COMMUNITY SAMPLING FOR CORA ##################### """
""" ####### READ CORA ARCHIVES TO CONSTRUCT a GRAPH: INPUT THE GRAPH ###### """
#edgelista = []
with open('data/cora/cora.cites', 'r') as f:
    G = nx.read_edgelist(f)   # INPUT THE GRAPH
    #rowCites = fr.readline()
    ##print(rowCites)
    #candidate_node1 = Interactors_Dict[rowCites.split('\t')[0]][0]
    #candidate_node2 = Interactors_Dict[rowCites.strip().split('\t')[1]][0]
    #edgelista.append
plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')
""" ################# PRODUCE COMMUNITIES AS GIRVAN-NEWMAN ################ """
G_with_Comm = nx.algorithms.community.girvan_newman(G)
TOP = next(nx.algorithms.community.girvan_newman(G)) 
                       # networkx.algorithms.community.centrality.girvan_newman
TOPList = list(TOP) # type(TOPList)
""" ############# PRODUCE COMMUNITIES BY MODULARITY CRITERION ############# """

""" ################# PRODUCE COMMUNITIES BY LOUVAIN METHOD ############### """

""" ################# PRODUCE COMMUNITIES AS ... ... ... ################## """

""" ####################################################################### """
nodescount = 2708 #len(Official_Symbol_Interactors_D) # len(Official_N_Synonym_SymInter_D)
featurescount = 1433 # len(Official_Symbol_Interactors)  # len(Official_N_Synonym_SymInter)
labelscount = 7 # len(Different_Organism_Interactors)
Overlap = 1
k = 400
# TOTAL NUMBER OF SAMPLES
TotSampNu = Overlap*int((nodescount-1) / k + 1 ) # iteratNo = 50*int(nodescount / k + 1 ) # N / k 
""" ####################################################################### """
BigListOfNodes = len(TOPList) # TotSampNu = len(TOPList)
with open('data/TOPLIST', 'w') as fw:
  for i in TOPList:
    print(i, file=fw)
""" ####################################################################### """
""" ######################### TEST THE VARIABLES ########################## 
for i in G_with_Comm:
    print(i)

for i in TOP:
    print(i)

a1 = G.subgraph(TOP[0]) # TOP0 is a node set
plt.subplot(124)
nx.draw(a1, with_labels=True, font_weight='bold')
A2 = G.subgraph(TOP[4]) # TOP0 is a node set
nx.draw(A2, with_labels=True, font_weight='bold')
nx.draw_shell(a1, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    ####################################################################### """
""" ################# WRITE A k-sized SAMPLE IN EACH FILE ################# """
""" ############################ 2nd TRY ################################## """
### SPREAD THE COMMUNITIES IN A GLOBAL LIST OF NODES ###
AggregTOPList = []
for i in range(len(TOPList)):
 for j in TOPList[i]:
  AggregTOPList.append(j)

""" ############################## REMOVE US ############################## """
for m in range(TotSampNu):
 newfileNodes = ''.join(['data/Cora_Samples/CORA_SubFile',str(m+1),'.node']) #'data/BIOGRID.content'  
 os.remove(newfileNodes)
""" ############################ 2nd TRY ################################## """
m=1
fillingTheFile=0    # filling: INDEX to WRITING FILE
ListListsSubgrNodes = []
TempListNodes = []

for i in AggregTOPList:
  # 1
  newfileNodes = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.node']) #'data/BIOGRID.content'  
  #newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.content']) #'data/BIOGRID.content'
  #newfileCites = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.cites']) #'data/BIOGRID.cites'
  if fillingTheFile<k:    #while fillingTheFile<k:
   with open(newfileNodes, 'a') as fw:
    print(i,file=fw)
    TempListNodes.append(i) #int(i))
   fillingTheFile+=1
  else:  # Current File full. # In the last m to come, most probably the File
          # doesn't get full up to k elements, so the code skips this branch
   ListListsSubgrNodes.append(TempListNodes)
   m+=1  # MOVE TO NEXT FILE to be written
   print(m)  # fillingTheFile=0    # filling: INDEX to WRITING FILE
   newfileNodes = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.node']) #'data/BIOGRID.content'  
   #newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.content']) #'data/BIOGRID.content'
   #newfileCites = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.cites']) #'data/BIOGRID.cites'
   with open(newfileNodes, 'a') as fw:
    print(i,file=fw)
   fillingTheFile=1    # filling: INDEX to WRITING FILE
   TempListNodes = [i] #int(i)]
  if m==(TotSampNu+1):
   print("EOF")
   break
ListListsSubgrNodes.append(TempListNodes)

""" ####################################################################### """
""" ############################## TESTING ################################ """
with open("test.edgelist", 'w') as f:
    nx.write_edgelist(G, f)

G10=nx.path_graph(4)
with open("test.edgelist", 'w') as f:
    nx.write_edgelist(G,f)

nx.write_edgelist(G10, "test.edgelist")
G11=nx.path_graph(4)
""" ####################################################################### """
""" ####################################################################### """
""" INPUT GRAPH WITH FEATURES AND GENERATE its SUBGRAPH LOCATED IN LIST/ARCHIVES """
#ListListsNodes = []
SubGraphsList = []
for file_index in range(TotSampNu): #range(len(ListListsSubgrNodes)) # range(m-1):
  #newfileNodes = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.node']) #'data/BIOGRID.content'  
  newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.content']) #'data/BIOGRID.content'
  newfileCites = ''.join(['data/Cora_Samples/CORA_SubFile',str(file_index+1),'.cites']) #'data/BIOGRID.cites'
  print(newfileCites)
  #TempListNodes = []
  """ GET SUBGRAPH COMPRISED BY THIS ARCHIVE's NODES """
  CurrSubgr = G.subgraph(ListListsSubgrNodes[file_index])
  SubGraphsList.append(CurrSubgr) # (G.subgraph(ListListsSubgrNodes[file_index]))
  """ CREATE THIS SUBGRAPH's .cites ARCHIVE """
  #with open(newfileNodes, 'r') as fr:  
  CurrSubgr.edges
  with open(newfileCites,'w') as fw:
     nx.write_edgelist(CurrSubgr, fw, data=False) # delimeter
  
""" ############################ PLOT IF U WANT ########################### """
  plt.subplot(121)
  nx.draw(CurrSubgr, with_labels=True, font_weight='bold')  
""" ####################################################################### """
""" ############################## REMOVE US ############################## """
for m in range(TotSampNu):
 newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(m+1),'.content']) #'data/BIOGRID.content'
 os.remove(newfileContent)
""" ####################################################################### """
""" CREATE THIS SUBGRAPH's .content ARCHIVE """
for file_index in range(TotSampNu): #range(len(ListListsSubgrNodes)) # range(m-1):
 newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(file_index+1),'.content']) #'data/BIOGRID.content'
 with open(newfileContent, 'a') as fw:
  for i in ListListsSubgrNodes[file_index]:
   print(Content_File_with_FtrVecs_Dicted[i], file=fw)
""" ####################################################################### """
""" ######################## END OF USEFUL ARCHIVE ######################## """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
  #with open(newfileNodes, 'r') as fr:
  # with open(newfileContent, 'a') as fw:    
   
   
print(SubGraphsList)
  
  nx.write_edgelist(G,fw) 
  line=fr.readline()
      if i==index:

             G.
             print(line.rstrip())
             print(line.rstrip(),file=fw)
  



for m in range(TotSampNu):
 newfileNodes = ''.join(['data/Cora_Samples/CORA_SubFile',str(m+1),'.node']) #'data/BIOGRID.content'  
 newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(m+1),'.content']) #'data/BIOGRID.content'
 newfileCites = ''.join(['data/Cora_Samples/CORA_SubFile',str(m+1),'.cites']) #'data/BIOGRID.cites'
 os.remove(newfileContent)
 os.remove(newfileCites)
 os.remove(newfileNodes)
 
with open('data/TOPLIST_Aggreg', 'w') as fw:
  for i in AggregTOPList:
    print(i, file=fw)

    ####################################################################### """
""" ################# WRITE A k-sized SAMPLE IN EACH FILE ################# """
""" ############################ 1st TRY ################################## """
m=1   # INDEX of FILE TO BE WRITTEN
for i in range(len(TOPList)):# i: INDEX to CURRENT COMMUNITY # range(TotSampNu):
 for j in TOPList[i]: # j: INDEX to CURRENT NODE (in community)
  newfileContent = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.content']) #'data/BIOGRID.content'
  newfileCites = ''.join(['data/Cora_Samples/CORA_SubFile',str(m),'.cites']) #'data/BIOGRID.cites'
  fillingTheFile=0    # filling: INDEX to WRITING FILE
  with open(newfileCites, 'w') as fw:
   while fillingTheFile<k:
   #if fillingTheFile<k: 
    print(j,file=fw)
    fillingTheFile+=1
   #else:  # Current File full.
   m+=1  # MOVE TO NEXT FILE to be written
   if j<len(TOPList[i]):
     i-=i
     fillingTheFile=0    # filling: INDEX to WRITING FILE  
     JUMP_TO_WITH_OPEN
 if m==TotSampNu:
  break

""" ####################################################################### """
""" ####################################################################### """
""" #################### COMMUNITY SAMPLING FOR SSPAMMER ################## """



""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ####################################################################### """
""" ########################### RANDOM SAMPLING ########################### """
def sample_from_cites(indices_to_nodes, output_file):
  with open(output_file, 'w') as fw:
   with open('data/BIOGRID.cites', 'r') as fr:
    for i in range(countEdges): 
        rowCites = fr.readline()
        #print(rowCites)
        candidate_node1 = Interactors_Dict[rowCites.split('\t')[0]][0]
        candidate_node2 = Interactors_Dict[rowCites.strip().split('\t')[1]][0]
        if candidate_node1 in indices_to_nodes and candidate_node2 in indices_to_nodes: # b = row.split('\t')[0]
         #print(rowCites)
         print(rowCites,file=fw)

""" ####################################################################### """
""" ####################################################################### """
# newfile1 = [] # newfile2 = {} # i=1
for i in range(TotSampNu):
 newfileContent = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.content']) #'data/BIOGRID.content'
 newfileCites = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.cites']) #'data/BIOGRID.cites'
 #with open(newfile, 'w') as output_file1:
 # row=output_file1.readline()
 sample_from_content(apothekeA[i],newfileContent)
 sample_from_cites(apothekeA[i],newfileCites)
 # print(row,file=output_file1)
 #with open(newfileContent, 'w') as fw:
 #   1
for i in range(TotSampNu):
 newfileContent = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.content']) #'data/BIOGRID.content'
 newfileCites = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.cites']) #'data/BIOGRID.cites'
 os.remove(newfileContent)
 os.remove(newfileCites)
#####################################################################################################
####################### WRITE TO FILE #############################
def sample_from_cites(indices_to_nodes, output_file):
  with open(output_file, 'w') as fw:
   with open('data/BIOGRID.cites', 'r') as fr:
    for i in range(countEdges): 
        rowCites = fr.readline()
        #print(rowCites)
        candidate_node1 = Interactors_Dict[rowCites.split('\t')[0]][0]
        candidate_node2 = Interactors_Dict[rowCites.strip().split('\t')[1]][0]
        if candidate_node1 in indices_to_nodes and candidate_node2 in indices_to_nodes: # b = row.split('\t')[0]
         #print(rowCites)
         print(rowCites,file=fw)

# newfile1 = [] # newfile2 = {} # i=1
for i in range(TotSampNu):
 newfileContent = ''.join(['data/Social_Spammer_Samples/SSpammer_SubFile',str(i),'.content']) #'data/BIOGRID.content'
 newfileCites = ''.join(['data/Biogrid_Samples/SSpammer_SubFile',str(i),'.cites']) #'data/BIOGRID.cites'
 #with open(newfile, 'w') as output_file1:
 # row=output_file1.readline()
 sample_from_content(apothekeA[i],newfileContent)
 sample_from_cites(apothekeA[i],newfileCites)
 

a1.nodes
a1.save_edgelist()
###########################################################

plt.subplot(122)
nx.draw(a1, with_labels=True, font_weight='bold')
a1.nodes







with open('data/BIOGRID.content', 'r') as f:
    row = f.readline()
    count5 = 0
    #sum(row[1:249551])
    while row:
        row = f.readline()
        count5+=1
###############################################################################
nodescount = len(Official_Symbol_Interactors_D) # len(Official_N_Synonym_SymInter_D)
featurescount = len(Official_Symbol_Interactors)  # len(Official_N_Synonym_SymInter)
labelscount = len(Different_Organism_Interactors)
SampleNum = 1
k = 5000
# TOTAL NUMBER OF SAMPLES
TotSampNu = SampleNum*int((nodescount-1) / k + 1 ) # iteratNo = 50*int(nodescount / k + 1 ) # N / k 
apothekeA = []
apothekeB = []
apothekeC = []
# SAMPLE INDICES - NODES
for index in range(TotSampNu):
    #apotheke.append(batchin(adj0,features0,train_mask0,val_mask0,test_mask0,y_train0,y_val0,y_test0))
    apothekeA.append(sorted(np.random.choice(range(nodescount), k, replace=False))) # np.random.randint(0,)
	                  # WE CONSTRUCTED THE matrices' INDICES TO BE SELECTED 
                  # a = np.random.choice(range(1000), 10, replace=False)		
	                  # adj = np.zeros((1000,1000))
apothekeB.append(np.arange(featurescount)) # construct array range() up to featurescount
apothekeC.append(np.arange(labelscount)) # construct array range() up to labelscount

def sample_from_content(indices_in_content, output_file):
  with open(output_file, 'w') as fw:
   with open('data/BIOGRID.content', 'r') as fr: 
    #print(fr.readline().rstrip())
    r = 0
    index = indices_in_content[0]
    for i in range(count5): #THE LINECOUNT of .CONTENT file
      line=fr.readline()
      if i==index:
             print(line.rstrip())
             print(line.rstrip(),file=fw)
             r+=1
             if r==k:
              break
             index = indices_in_content[r]

  return 1

""" INPUT THE INPUT FILE """
count4 = 0
lista = []
Interactors_Dict = collections.defaultdict(list)
with open('data/BIOGRID.content', 'r') as f: # open the file for reading
  #headers = f.readline().strip().split('\t')
  for line in f:
    row = line.strip().split('\t')
    row_interactor_id = row[0] # keys of the main dictionary
    #row_interactorB_id = row[4]
    # flags
    #lista.append(row_interactor_id)
    Interactors_Dict[row_interactor_id].append(count4)
    count4+=1
    #Interactors_Dict[row_interactorB_id].add([])
    # node labels/classes
#count4=0
#for entry in Official_Symbol_Interactors_D:
# Interactors_Dict[entry] = count6
# count6+=1
"""count4 = 0
Nodes_to_RowIndices = collections.defaultdict(list)
for q in Official_Symbol_Interactors:
#for p in papersD:
    Nodes_to_RowIndices[q] = count4
    count4 += 1 """
with open('data/BIOGRID.cites', 'w') as output_file1:
    headers = f.readline().strip().split('\t')
    for count, line in enumerate(f):
        1
countEdges = count # 1500000
def sample_from_cites(indices_to_nodes, output_file):
  with open(output_file, 'w') as fw:
   with open('data/BIOGRID.cites', 'r') as fr:
    for i in range(countEdges): 
        rowCites = fr.readline()
        #print(rowCites)
        candidate_node1 = Interactors_Dict[rowCites.split('\t')[0]][0]
        candidate_node2 = Interactors_Dict[rowCites.strip().split('\t')[1]][0]
        if candidate_node1 in indices_to_nodes and candidate_node2 in indices_to_nodes: # b = row.split('\t')[0]
         #print(rowCites)
         print(rowCites,file=fw)

# newfile1 = [] # newfile2 = {} # i=1
for i in range(TotSampNu):
 newfileContent = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.content']) #'data/BIOGRID.content'
 newfileCites = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.cites']) #'data/BIOGRID.cites'
 #with open(newfile, 'w') as output_file1:
 # row=output_file1.readline()
 sample_from_content(apothekeA[i],newfileContent)
 sample_from_cites(apothekeA[i],newfileCites)
 # print(row,file=output_file1)
 #with open(newfileContent, 'w') as fw:
 #   1
for i in range(TotSampNu):
 newfileContent = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.content']) #'data/BIOGRID.content'
 newfileCites = ''.join(['data/Biogrid_Samples/BIOGRID_SubFile',str(i),'.cites']) #'data/BIOGRID.cites'
 os.remove(newfileContent)
 os.remove(newfileCites)
#####################################################################################################
