########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

# perfect essays : 37, 118, 147,
import csv
import sys
from nltk.corpus import stopwords
import numpy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy import spatial
from nltk import word_tokenize, pos_tag

esstxt = "@ORGANIZATION1, Computers are great tools and a great piece of modern technology. Almost every family has them. About @PERCENT1 of my class has computers. So many people have them because their helpful and another current learning resource. Also it's a gr"

word = word_tokenize(esstxt)
pos = pos_tag(word)
print(pos)
ncount = 0;
vcount = 0;
#
# pf = open('pos_tags.txt', 'w')
# for i in range(0, len(pos)):
#     pf.write(str(pos[i]) + '\n')
#     # if ((pos[i][1]=='NN') or ( pos[i][1]=='NNS')):
#     #    ncount+=1
#     #    nf.write(pos[i][0]+'\n')
# # print(ncount)
# # for i in range(0,len(pos)):
# #    if (pos[i][1]=='VB'):
# #        vcount+=1
# #        vf.write(pos[i][0]+'\n')
# # nf.close()
# # vf.close()
# pf.close()

