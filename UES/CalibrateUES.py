import SEAM
import SAM
import SYNAN
import DISAM
import SYNERR
import sys, getopt
import csv

def main(argv):
    f = open('../Dataset/Set1Complete.csv', 'rb')
    try:
        reader = csv.reader(f)
        for row in reader:
            ess_id = int(row[0])
            ess_set = int(row[1])
            ess_text = unicode(row[2], errors='ignore')
            ess_score_r1 = float(row[3])
            ess_score_r2 = float(row[4])
            seam_score = SEAM.performLSA(EssayFileName, DataFileName)
            sam_score = SAM.performSA(EssayFileName, DataFileName)
            synan_score = SYNAN.scoreSYN(EssayFileName, DataFileName)
            disam_score = DISAM.scoreDiscourse(EssayFileName, DataFileName)
            synerr_score = SYNERR.scoreSYNERR(EssayFileName)
    finally:
        f.close()



    # print seam_score
    # print sam_score
    # print synan_score
    # print disam_score
    # print synerr_score
    # print "FINAL SCORE : ", (seam_score + sam_score + synan_score + disam_score + synerr_score)/5


if __name__ == "__main__":
    main()
