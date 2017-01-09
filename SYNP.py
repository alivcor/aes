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

esstxt = "@ORGANIZATION1, Computers are great tools and a great piece of modern technology. Almost every family has them. About @PERCENT1 of my class has computers. So many people have them because their helpful and another current learning resource. Also it's a gr"


