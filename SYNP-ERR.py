########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

# perfect essays : 37, 118, 147,
import csv
import sys
import nltk
import numpy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.random_projection import sparse_random_matrix
from scipy import spatial
from nltk import word_tokenize, pos_tag
import re

test_essay = "@ORGANIZATION1, Computers are great tools and a great piece of modern technology. Almost every family has them. About @PERCENT1 of my class has computers. So many people have them because their helpful and another current learning resource. Also it's a gr"
test_essay = "I think the use of computers is very good things in the present day @LOCATION2. I think this because people use it to research, stay in touch with friends and family, and music. These are all things adults and kids need a like to do. read on and I will explain why this is so important. My first reason why I think computers are good thing is because the are a good source to do research on a specific topic. Kids all over the @LOCATION1 use the internet to do class projects. Adults can use the computer to look up a certain food resipe. If you need help with something the computer is allways there. Some people even use the computer for a job. Computers can help with a lot of things. Another reason why computers are a big help because you can stay in touch with friends and family. A great thing to use on a computer is a web cam. With this you can see and talk to people on a lillte camray even if they are far away. You can also go on myspace of face book. Theare two websites that are used to post pictures and to talk to chosen friend. It is a good way for people to get to know you. A similar thing to this is instant messaging. You can use this to talk to friends and many on line. Don't stop reading there is more thing! My last reason why computer are sutch a great thing to have is because you can do do mutch with music on the computer. You can use the computer to listen to music, and put music on to and ipod or @NUM1 player. Some people use the computer to make music. You can get a progrem that you can make heats and put it together to make some kind of a song. music is a great thing that all age groups love. That is why I think haveing a computer is a very positive thing. You can do researching, say in touch with friends and family, and do a lot with music. There is so mutch more then thease @NUM2 thing that a computer is good for. I don't think the world would be the same without computers."
test_essay = "Dear local newspaper, I strongly believe computers can be bad for society. My reasons being people are spending to much time on their computers, less time exercising, and lose interacting with family and friends. My first reason why i think computers are bad for our society is people are spending to much time on them. I know when i get home from school the computer is all i can think about until i go on. It's like an addiction. People spend all their time on a social net work other then going out and meeting new people. My second reason is people spend less time exercising. Ive been over a friends house before, and all we did was play computer games instead of enjoying nature outside. Alot of @CAPS1 don't make healthy food choices, so why not go out and exercise? It's because of the computer. Ive seen my little brother spend his whole saterday eating in front of the computer and playing games. It's just unhealthy. My last and final reason is it takes time away from family and friends ive spent my whole day after school in my room on the computer. It's like when your on the computer nothing else matters because you think ""hey why not just talk to my friends over @CAPS2, or myspce, or facebook! you lose a lot of friendships like this. Also you lose friendship with you family because you badly talk to them while your on the computer. Half the time im to busy talking to someone i don't know then helping my little brother with home work, or helping my mom clean the house. In conclusion these are my three reasons why i think the computer is bad for sivilization. I really hope you take all this into consideration and really think what the computer can do to @CAPS1 I people once again those three resons are one, people are spending to much time on their computer. Two, spend less time exercising. Three, lose interactions with family and friends."
transformer = TfidfTransformer(smooth_idf=False)

#
# word = word_tokenize(esstxt)
# pos = pos_tag(word)
# print(pos)
global grammar

grammar = "NP: {<IN>?<IN>?<RB>?<DT>?<JJ>*<NN>}"
grammar = """
	NP:   {<WRB>?<MD>?<PRP>?<IN>?<IN>?<WRB>?<RB>?<DT>?<PRP>?<JJ.*>*<NN.*>+<IN>?<JJ>?<NN>?<CC>?<NN>?}
	CP:   {<JJR|JJS>}
	VP: {<VB.*>}
	NP: {<PRP><MD>}
	NP: {<DT>}
	NP: {<WP>}
	COMP: {<DT>?<NP><RB>?<VP><DT>?<CP><THAN><DT>?<NP>}
	"""

ncount = 0;
vcount = 0;

global nVPflag
global nNPflag

nNPflag = True
nVPflag = True

def check_flags(t, nNPflag, nVPflag):
    try:
        t.label
    except AttributeError:
        return [nNPflag, nVPflag]
    else:
        if t._label == "NP":
            nNPflag = False
        if t._label == "VP":
            nVPflag = False
        for child in t:
            flags = check_flags(child, nNPflag, nVPflag)
            nNPflag = flags[0]
            nVPflag = flags[1]
    return [nNPflag, nVPflag]

def get_type1_errors(esstxt):
    'Incomplete sentence'
    esstxt = re.sub(r'(\@)([A-Za-z]*)([\W]*[\d]*[\W]*)(\s)', " ", esstxt)

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(esstxt.strip())

    n_err = 0
    for sent in sents:
        words = word_tokenize(sent)
        tagged_words = pos_tag(words)
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(tagged_words)
        # result.draw()
        flags = check_flags(result, True, True)
        if flags[0]:
            print "NO NP FOUND : ",
            print sent
            n_err += 1
            # result.draw()
        if flags[1]:
            print "NO VP FOUND : ",
            print sent
            n_err += 1
            # result.draw()
    return n_err

def get_syntax_errors(esstxt):
    nt1 = get_type1_errors(esstxt)
    return nt1


f = open('Dataset/Set1Complete.csv', 'rb')
count = 0
try:
    reader = csv.reader(f)
    for row in reader:
        if count > 0 and count <= 20:
            ess_id = int(row[0])
            ess_set = int(row[1])
            ess_text = unicode(row[2], errors='ignore')
            ess_score_r1 = float(row[3])
            ess_score_r2 = float(row[4])
            ess_score = ess_score_r1 + ess_score_r2
            syn_errors = get_syntax_errors(ess_text)
            print count, syn_errors
            count += 1
        else:
            count += 1
finally:
    f.close()


