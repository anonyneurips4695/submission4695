import numpy as np
import random
import sys

f = open(sys.argv[1], 'w')
for i in range(100):
    sample_list = np.random.randint(0, 2000, size=int(sys.argv[2])).tolist()
    sample_list = [str(num) for num in sample_list]
    line = ' '.join(sample_list)
    line = str(i) + ' ' + line
    f.write(line + '\n')

