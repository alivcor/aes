import EventIssuer
import progressbar
import numpy
import re

bar = progressbar.ProgressBar()

def loadCompleteGloveModel(logfilename):
    global bar
    model = {}
    EventIssuer.issueMessage("Loading GLoVE Word Vectors. This will take a while.", logfilename)
    EventIssuer.issueSleep("Turning to sleep mode.", logfilename)
    f = open("/glove_vectors/glove.42B.300d.txt", 'r').readlines()
    for i in bar(range(len(f))):
        line = f[i]
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    EventIssuer.issueSuccess("Loaded GLoVE Word Vectors", logfilename)
    EventIssuer.issueMessage(len(model) + " words loaded.", logfilename)
    return model

def getWordVec(word, logfilename):
    word = word.strip().lower()
    # EventIssuer.issueMessage("Word2Vec - Lookup : " + word, logfilename)
    with open("/glove_vectors/glove.42B.300d.txt", 'r') as f:
        for line in f:
            if word[0] == line[0]:
                if word in line:
                    splitLine = line.split()
                    if word == splitLine[0]:
                        embedding = [float(val) for val in splitLine[1:]]
                        # EventIssuer.issueSuccess("Word2Vec - Found WordVec ! " + splitLine[0], logfilename)
                        return numpy.array(embedding)
    EventIssuer.issueWarning("Word2Vec - Primary loopkup failed. Trying advanced lookup : " + word, logfilename)
    word = re.sub('[^a-z]+', ' ', word)
    words = word.split()
    EventIssuer.issueMessage("Word2Vec - Advanced Lookup identifies the presence of these words in the clump : " + str(words), logfilename)
    EventIssuer.issueSharpAlert("Word2Vec - Returning a list of embedding vectors instead of just one: " + str(words), logfilename)
    wvecs = []
    for word in words:
        iffound = False
        # EventIssuer.issueMessage("Word2Vec - Lookup : " + word, logfilename)
        with open("/glove_vectors/glove.42B.300d.txt", 'r') as f:
            for line in f:
                if word[0] == line[0]:
                    if word in line:
                        splitLine = line.split()
                        if word == splitLine[0]:
                            embedding = [float(val) for val in splitLine[1:]]
                            # EventIssuer.issueSuccess("Word2Vec - Found WordVec ! " + splitLine[0], logfilename)
                            iffound = True
                            wvecs.append(numpy.array(embedding))
        if not iffound:
            EventIssuer.issueError("Word2Vec - Not found : " + word, logfilename)
            EventIssuer.issueError("Word2Vec - Appending an embedding of -1's", logfilename)
            wvecs.append(numpy.negative(numpy.ones((300,))))
    return wvecs

def getGloveVector(model, word):
    return model[word]