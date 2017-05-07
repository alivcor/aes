import SEAM
import SAM
import SYNAN
import DISAM
import SYNERR
import sys, getopt
import pickle
import numpy as np


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:d:",["ifile=", "dfile="])
    except getopt.GetoptError:
        print 'driver.py -i <inputfile> -d <datafile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'driver.py -i <inputfile> -d <datafile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            EssayFileName = arg
        elif opt in ("-d", "--dfile"):
            DataFileName = arg
    seam_score = SEAM.performLSA(EssayFileName, DataFileName)
    sam_score = SAM.performSA(EssayFileName, DataFileName)
    synan_score = SYNAN.scoreSYN(EssayFileName, DataFileName)
    disam_score = DISAM.scoreDiscourse(EssayFileName, DataFileName)
    synerr_score = SYNERR.scoreSYNERR(EssayFileName)
    # print seam_score
    # print sam_score
    # print synan_score
    # print disam_score
    # print synerr_score

    calibrator = pickle.load(open("calibrated_model.sav", 'rb'))
    # print "FINAL SCORE : ", int(0.19738706*seam_score + 0.12756882*sam_score + 0.465254231*synan_score + 0.03680639*disam_score + 0.0728261*synerr_score)
    scoref = int(calibrator.predict(np.array([seam_score, sam_score, synan_score, disam_score, synerr_score])))
    with open("/Users/abhinandandubey/Documents/untrase.txt", 'w') as fui:
        fui.write(str(scoref))

if __name__ == "__main__":
    main(sys.argv[1:])
