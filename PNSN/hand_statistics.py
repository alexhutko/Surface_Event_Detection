#!/home/ahutko/miniconda3/envs/surface_dl/bin/python

# Inefficient, but carefully calculate accuracy, precision, recall, f1
#   for event classifier results at the event and trace level.
#   Just focuses on one model, SeismicCNN_2d, to serve as a check on
#   the 3000 line AI generated script that produces 100 figures and
#   lots of statistics.

import glob
import numpy as np

correct, incorrectA, incorrectP, EtotalP, EtotalA, ntracesA = {}, {}, {}, {}, {}, {}
ntracesPtracelevel, ntracesPeventlevel, nTfalseneg, nTfalsepos, nTcorrect = {}, {}, {}, {}, {}
nTotal, correctAll, incorrectAll = 0,0,0
averagetracecount = {}

for etype in [ 'eq', 'ex', 'su' ]:
    correct[etype] = 0
    incorrectA[etype] = 0
    incorrectP[etype] = 0
    EtotalP[etype] = 0
    EtotalA[etype] = 0
    ntracesA[etype] = 0
    ntracesPtracelevel[etype] = 0
    ntracesPeventlevel[etype] = 0
    nTcorrect[etype] = 0
    nTfalseneg[etype] = 0
    nTfalsepos[etype] = 0
    averagetracecount[etype] = []

for outfile in glob.glob('RESULTS/6*ou*.txt'):
    print("TYRING: ",outfile)
    f = open(outfile)
    lines = f.readlines()
    f.close()
    etype = 'na'
    ntracesthisevent = 0
    nTeq, nTex, nTsu = 0,0,0
    for line in lines:
      try:
        if "mean_all" in line:
            analyst = line.split()[18]
            if analyst == 'px':
                analyst = 'ex'
        if "SeismicCNN_2d_mean_all" in line:
            nTotal += 1
            evid = int(line.split()[0])
            pred = line.split()[14].lower()
            EtotalP[pred] += 1
            EtotalA[analyst] += 1
            if pred == analyst:
                correctAll += 1
                correct[analyst] += 1
            else:
                incorrectAll += 1
                incorrectA[analyst] += 1
                incorrectP[pred] += 1
        if "PROBS" in line and "SeismicCNN_2d" in line:
           ntracesA[analyst] += 1
           predeq = line.split()[4]
           predex = line.split()[5]
           predsu = line.split()[7]
           pred = 'na'
           if predeq >= predex and predeq >= predsu:
               pred = 'eq'
               nTeq += 1
           if predex >= predeq and predex >= predsu:
               pred = 'ex'
               nTex += 1
           if predsu >= predeq and predsu >= predex:
               pred = 'su'
               nTsu += 1
           ntracesPtracelevel[pred] += 1 
           ntracesthisevent += 1
        if "SeismicCNN_2d_mean_all" in line:
            pred = line.split()[14].lower()
            ntracesPeventlevel[pred] += ntracesthisevent
            if pred == 'eq':
                nTcorrect['eq'] += nTeq
                nTfalseneg['eq'] += nTex + nTsu
                nTfalsepos['su'] += nTsu
                nTfalsepos['ex'] += nTex
            if pred == 'ex':
                nTcorrect['ex'] += nTex
                nTfalseneg['ex'] += nTeq + nTsu
                nTfalsepos['su'] += nTsu
                nTfalsepos['eq'] += nTeq
            if pred == 'su':
                nTcorrect['su'] += nTsu
                nTfalseneg['su'] += nTeq + nTex
                nTfalsepos['eq'] += nTeq
                nTfalsepos['ex'] += nTex
        averagetracecount[analyst].append(ntracesthisevent)
      except:
          pass

# overall event-level accuracy
accuracy = correctAll / nTotal if nTotal else 0
print(f"\nModel considered: SeismicCNN_2d")
print(f"\nRecords processed: {nTotal}")
print(f"Overall accuracy: {accuracy:.4f}\n")

# per‐class precision, recall, F1
for etype in ['eq', 'ex', 'su']:
    tp   = correct[etype]
    fp   = incorrectP[etype]
    fn   = incorrectA[etype]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0
    accuracy = tp / EtotalA[etype] if EtotalA[etype] else 0

    print(f"{etype.upper()}")
    print(f" (n={EtotalA[etype]})")
    print('Event level:')
    print(f"  Precision = {prec:.4f} ({tp}/{tp+fp})")
    print(f"  Recall    = {rec:.4f} ({tp}/{tp+fn})")
    print(f"  F1 score  = {f1:.4f}")
    print(f"  Accuracy = {accuracy:.4f}")
    #---- Now do trace level stats
    tp = nTcorrect[etype]
    fp = nTfalsepos[etype] 
    fn = nTfalseneg[etype]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0
    accuracy = tp / ntracesA[etype] if ntracesA[etype] else 0

    print('Trace level:')
    print(f"  Precision = {prec:.4f} ({tp}/{tp+fp})")
    print(f"  Recall    = {rec:.4f} ({tp}/{tp+fn})")
    print(f"  F1 score  = {f1:.4f}")
    print(f"  Accuracy = {accuracy:.4f}")
    print(f"  Ntraces for {etype} labeled dataset:  {ntracesA[etype]}  Ntraces predicted as {etype} event-level:  {ntracesPeventlevel[etype]}  Ntraces predicted as {etype} trace-level:  {ntracesPtracelevel[etype]}")
    print(f"  nTruePositive: {tp}  nFalsePositive: {fp}  nFalseNegative: {fn}\n")
    print('')

# Average count of traces used for each event type
avgeq = np.mean(np.asarray(averagetracecount['eq']))
avgex = np.mean(np.asarray(averagetracecount['ex']))
avgsu = np.mean(np.asarray(averagetracecount['su']))
print("Average traces per event type.  EQ: ", avgeq, "  EX: ", avgex, "  SU: ", avgsu)

# Median count of traces used by each event type
medeq = np.median(np.asarray(averagetracecount['eq']))
medex = np.median(np.asarray(averagetracecount['ex']))
medsu = np.median(np.asarray(averagetracecount['su']))
print("Median traces per event type.  EQ: ", medeq, "  EX: ", medex, "  SU: ", medsu)


