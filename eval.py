total = 0
nonzero = 0
correct = 0
partial = 0
false_neg = 0
with open('hyper_results_smoothed.txt', 'r') as f:
    for line in f:
        tmp = line.split(' ')
        a = float(tmp[2])
        b = float(tmp[3])
        if a > b:
            partial += 1
            if b == 0.0:
                correct += 1
        elif b > a:
            false_neg += 1
        if a + b > 0.0:
            nonzero += 1
        total += 1
print "Total: %d, Correct: %d, Partial: %d" % (total, correct, partial)
total = float(total)
print false_neg
print nonzero
nonzero = float(nonzero)
print "Correct: %0.3f, Partial: %0.3f" % (correct / total, partial / total)
print "Correct: %0.3f, Partial: %0.3f" % (correct / nonzero, partial / nonzero)
