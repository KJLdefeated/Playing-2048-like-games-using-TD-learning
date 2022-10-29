from mailbox import linesep
import numpy
import matplotlib.pyplot as plt

lr = []
avgscore = []
epoch = []
cnt = 1000

with open('train.log') as f:
    while True:
        lines = f.readline()
        if not lines:
            break
        #print(lines[8:11])
        for i in range(0,len(lines)):
            if(lines[i:i+3] == "avg"):
                s = ""
                i = i + 6
                while(lines[i].isdecimal()):
                    s += lines[i]
                    i+=1
                avgscore.append(int(s))
                epoch.append(cnt)
                cnt += 1000
                break

X_ax = numpy.array(epoch)
Y_ax = numpy.array(avgscore)

plt.plot(X_ax, Y_ax)
plt.show()
