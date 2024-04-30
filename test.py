import pandas as pd
import numpy as np
rating_distribution = np.zeros(11)
dataset = pd.read_csv('ratings.csv')
for _, row in dataset.iterrows():
    rating_distribution[int(float(row['rating'])*2)] += 1
print(rating_distribution)
#[    0.  1370.  2811.  1791.  7551.  5550. 20047. 13136. 26818.  8551. 13211.]
'''
1.1013840376082467
0 0 0 1 38 4823 49711 34570 7757 3652 284 
min[i], max[i] = np.percentile(tmp, (1 + i) * 0.2 ), np.percentile(tmp, 100 - (1 + i) * 0.5)
'''

'''
1.0944379655177952
0 0 0 1 96 1190 11366 58180 23120 6529 354 
min[i], max[i] = np.percentile(tmp, 0.5 + (1 + i) * 0.2 ), np.percentile(tmp, 96 - (1 + i) * 0.5)
'''

'''
XX
min[i], max[i] = np.percentile(tmp, 0.2 + (1 + i) * 0.5 ), np.percentile(tmp, 96 - (1 + i) * 1)
 min[i], max[i] = np.percentile(tmp, 0.1 + 0.1 * i), np.percentile(tmp, 99.9 - i * 0.1)
'''

'''
1.1212381306860972
0 0 0 1 32 1383 12463 46302 30782 8017 1856 
3 64 565 1694 2560 4212 7460 11226 16345 18329 18378 11534 6707 1628 124 6 1 0 0 
min[i], max[i] = np.percentile(tmp, 0.3 + (1 + i) * 0.4 ), np.percentile(tmp, 95 - (1 + i) * 1)
'''

'''
1.2
min[i], max[i] = np.percentile(tmp, (1 + i) * 0.2 ), np.percentile(tmp, 96 - (1 + i) * 0.2)
'''