import SEAM
import SAM
import SYNAN
import sys, getopt


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:d:",["ifile=", "dfile="])
    except getopt.GetoptError:
        print 'SEAM.py -i <inputfile> -d <datafile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'SEAM.py -i <inputfile> -d <datafile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            EssayFileName = arg
        elif opt in ("-d", "--dfile"):
            DataFileName = arg
    seam_score = SEAM.performLSA(EssayFileName, DataFileName)
    sam_score = SAM.performSA(EssayFileName, DataFileName)
    synan_score = SYNAN.scoreSYN(EssayFileName, DataFileName)
    print seam_score
    print sam_score
    print synan_score



if __name__ == "__main__":
    main(sys.argv[1:])
