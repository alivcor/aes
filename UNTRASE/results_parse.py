import csv, re
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

f = open("results.csv", 'rb')
y_pred = []
y_true = []
y_rlxd_pred = []
try:
    reader = csv.reader(f)
    for row in reader:
        if(row[0][0] == "P"):
            # print row[0]
            matched = re.match(r'(Predicted\s\:\s*)([0-9]*)(\s*\|\s*Actual\s\:\s*)([0-9]*\.[0-9])', row[0])
            if(abs(float(matched.group(2)) - float(matched.group(4))) == 1):
                y_rlxd_pred.append(float(matched.group(4)))
            else:
                y_rlxd_pred.append(float(matched.group(2)))
            y_pred.append(float(matched.group(2)))
            y_true.append(float(matched.group(4)))

finally:
    print accuracy_score(y_true, y_rlxd_pred)
    print(classification_report(y_true, y_rlxd_pred))
    print accuracy_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    f.close()