# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:52:12 2018

@author: gerasimos
"""
from __future__ import print_function
import collections
import numpy as np
import itertools
#import string
authorsD = collections.defaultdict(list) 
papersD = collections.defaultdict(list)
normalizedAuthorsD = collections.defaultdict(list)
fullAuthorsD = collections.defaultdict(list)
AuthorNoInPapD = collections.defaultdict(list)
papersClusterIdD = collections.defaultdict(list)
dictAdj = collections.defaultdict(dict)
dictFeat = collections.defaultdict(dict)
dictLabels = collections.defaultdict(dict)

InteractorA_D = collections.defaultdict(list)
InteractorB_D = collections.defaultdict(list)

# data/BIOGRID-ALL-3.4.163.tab2.txt
# data/arxiv-mrdm05.dat
"""with open("data/BIOGRID-ALL-3.4.163.tab2.txt") as f_in:
                             # csv.reader(f_in, delimeter='\n')
    for line in f_in:        # CREATE an anonymous generator
        # 1
        row_interactorA_id = line.split('\t')[1].rstrip(" ") #print(row_author_id[0])
        row_interactorB_id = line.split('\t')[2]# .rstrip(" ") #print(row_author_id[0])
        print(row_interactorA_id) # print(line)
        break"""
f = open('data/BIOGRID-ALL-3.4.163.tab2.txt', 'r') # open the file for reading
data = []
for row_num, line in enumerate(f):
    1
    row_interactorA_id = line.strip().split('\t')
    if row_num == 0: # first line is the header
         header = values
    else:
        data.append([float(v) for v in values])
basic_data = array(data)
f.close() # close the file

        row_interactorA_id = line.split('\t')[1].rstrip(" ") #print(row_author_id[0])
        row_interactorB_id = line.split('\t')[2]# .rstrip(" ") #print(row_author_id[0])
        print(row_interactorA_id) # print(line)
        break


        row_author_id = line.split('|')[1].lstrip(" ").rstrip(" ") #print(row_author_id[0])
        row_paper_id = line.split('|')[5].lstrip(" ").rstrip(" ") #print(row_author_id[0])
        #print(row_author_id,row_paper_id)
        authorsD[row_author_id].append(row_paper_id)        
        # authors[line[0]].append(line[5])
        papersD[row_paper_id].append(row_author_id)
        #
        row_normalizedAuthors = line.split('|')[2].lstrip(" ").rstrip(" ") #print(row_author_id[0])
        normalizedAuthorsD[row_normalizedAuthors].append(row_author_id)
        #
        row_fullAuthors = line.split('|')[3].lstrip(" ").rstrip(" ") #print(row_author_id[0])
        fullAuthorsD[row_fullAuthors].append(row_author_id)
        #
        row_AuthorNoInPap = line.split('|')[4].lstrip(" ").rstrip(" ") #print(row_author_id[0])
        AuthorNoInPapD[row_AuthorNoInPap].append(row_author_id)
        #
        row_papersCluster = line.split('|')[6].lstrip(" ").rstrip(" ") #print(row_author_id[0])
        papersClusterIdD[row_papersCluster].append(row_author_id)

for i in authorsD['41952']:
    print(i)

adj = np.zeros((len(papersD),len(papersD)), np.int32)
feat = np.zeros((len(papersD), len(authorsD)), np.int32)
labels = np.zeros((len(papersD),len(AuthorNoInPapD)), np.int32)
################ SANITY CHECK #####################################
papers = authorsD['41952']
papersList = []
papersList.append(papers)
#adj = np.zeros((len(authorsD),len(authorsD)))

for p in papersD['10']:
    print(p, authorsD[p])
###################################################################
#for author, papers in authors: #AuthorsDict
"""for author in authorsD: #AuthorsDict
    #print(author)
    #print(authors[author])
    #break
    #print(author)
    papersList = []
    papersList.append(authorsD[author]) #papersList.append(papers)
    ###for p in papersList:
        adj[p,pp] =  ###
        
    for p in papersList:
        authorsList = []        
        authorsList.append(papersD[p])
        adj[p,pp] = 
        #papersList.append(pp)
    
    #for k,v in papersList:"""
####################################################################
# with open('arxiv.cites', 'a') as output_file1:
for p in papersD:
    authorsList = papersD[p] #authorsList = []        
                             #authorsList.append(papersD[p]) #has authors (of this paper)
    #papersList.append(pp)
    for author in authorsList:
    #for author in authorsD: #AuthorsDict
        #print(author)
        #print(authors[author])
        #break
        #print(author)
        papersList = authorsD[author] #papersList = []
                                      #papersList.append(authorsD[author]) #has papers
                                            #papersList.append(papers)
        """for p in papersList:
            adj[p,pp] =  """
        for pp in papersList:
            dictAdj[p][pp]=1 # dictAdj[p,pp]=1 
            #print('{:07d} {07d}'.format(key,neighboor),file=output_file1, end='')
####################################################################
""" mapping: newDict
             Read each line
              Take the key of each line
                If it belongs to dictionary: continue
                  else: add to list and increase counter   """
count = 0                                                       
dictMap = collections.defaultdict(dict)
dictReverMap = collections.defaultdict(dict)
for p in dictAdj:
#for p in papersD:
    dictMap[p] = count
    dictReverMap[count] = p
    count = count+1    
    if count == 1:
         print(p)
################ SANITY CHECK #####################################
a = 0
for key in dictMap:
    if key == dictReverMap[dictMap[key]]:
       1 #print('ok')
    else: print('problem')
####################################################################
#matr = np.zeros((len(dictMap),len(dictMap)))
with open('data/arxiv.cites', 'a') as output_file1:
  for key in dictAdj: # for key,value in dictAdj: 
                         # print(value)
    DictedNeighboors = dictAdj[key] # value = dictAdj[key]
    for neighboor in DictedNeighboors:
        #print(neighboor,DictedNeighboors[neighboor])  # print(element)
        #for key1 in element:
        ########################## SANITY PRINT ############################
        #print(key,neighboor)    #print(key1)
        ####################################################################
            #print(element)
            #key1=int(float(key1))
            #print(key1)
            #key= int(float(key))
            #print(key)
        adj[dictMap[key],dictMap[neighboor]]=1  # value[element]
        # OUTPUT THE RESULT
        print('{:d} {:d}\n'.format(dictMap[key],dictMap[neighboor]),file=output_file1, end='')
############################# SANITY CHECK ###############################
store = 0
storeOv = 0 
for j in range(np.shape(adj)[0]):
 store = sum(adj[j])
 storeOv = max((storeOv, store))
 """store = 0
 for i in range(np.shape(adj)[0]):
  store = store + adj[j][i]
 if j % 1000 == 0:
  print(j) 
 if storeOv < store:
  storeOv = store """
####################################################################
####################################################################
####################################################################  
########################## FEATURE MATRIX ##########################
"""with open('arxiv.cites', 'a') as output_file1:
    for i in range(len(np.shape(adj))):
     for j in range(len(np.shape(adj))):
      if adj[i,j]==1:
       print('{:07d} {:07d}'.format(i,j,file=output_file1,end='')"""
### EQUIVALENT TO PAPERSD
for p in papersD:
    authorsList = papersD[p] #authorsList = []        
                             #authorsList.append(papersD[p]) #has authors (of this paper)
    #papersList.append(pp)
    for author in authorsList:
    #for author in authorsD: #AuthorsDict
        #print(author)
        #print(authors[author])
        #break
        #print(author)
        #papersList = authorsD[author] #papersList = []
                                      #papersList.append(authorsD[author]) #has papers
                                            #papersList.append(papers)
        ##for p in papersList:
        ##    adj[p,pp] =  
        #for pp in papersList:
            #dictAdj[p][pp]=1
        dictFeat[p][author]=1 
        
dictMapFeat = collections.defaultdict(dict)
"""MapFeatFlag = collections.defaultdict(dict)#dict(itertools.izip(xrange(len(authorsD)), itertools.repeat(0)))
for author in authorsD:
 MapFeatFlag[author]=0 """
#dictMapFeatFlag = collections.defaultdict(dict)
#dict.fromkeys(dictMapFeatFlag, 0)
#dictReverMapFeat = collections.defaultdict(dict)
count2 = 0    
""" SHOULD BE WORKING, BUT IT IS NOT 
for p in dictFeat:
#for p in papersD:
    DictedAuthors = dictFeat[p] #dictMapFeat[p] = count2
    for author in DictedAuthors:
     if MapFeatFlag[author] == 0:
      dictMapFeat[author] = count2
      #dictReverMapFeat[count2] = p
      count2 = count2+1
      dictMapFeatFlag[author] = 1 """
for author in authorsD:
 dictMapFeat[author] = count2
 count2 = count2+1

with open('data/arxiv.content', 'a') as output_file2:
    for key in papersD: # dictFeat: # for key,value in dictAdj: 
                         # print(value)
     DictedNeighboors = papersD[key] # dictAdj[key] # value = dictAdj[key]
     for neighboor in DictedNeighboors:
        #print(neighboor,DictedNeighboors[neighboor])  # print(element)
        #for key1 in element:
        ########################## SANITY PRINT ############################
        #print(key,neighboor)    #print(key1)
        ####################################################################
            #print(element)
            #key1=int(float(key1))
            #print(key1)
            #key= int(float(key))
            #print(key)
        feat[dictMap[key],dictMapFeat[neighboor]]=1  # value[element]
        # OUTPUT THE RESULT
        print('{:d} {:d}\n'.format(dictMap[key],dictMapFeat[neighboor]),file=output_file2, end='')     
        #print('{:d} {:d}\n'.format(dictMap[key],dictMapFeat[neighboor]),file=output_file2, end='')     
####################################################################
### LABEL MATRIX  

####################################################################
#new_dic = {}
#new_dic[1] = {}
#new_dic[1][2] = 5











