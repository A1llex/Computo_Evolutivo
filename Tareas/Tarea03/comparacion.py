# Comparativa entre dos algoritmos evolutivos
# Utilizando la prueba de Wilcoxon rank sum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


#Los evolutivos
s1 = [0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001,
 0.0001]
s2 = [0.10037166,   
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166,
 0.10037166]

plt.hist([s1,s2])
plt.show()

plt.boxplot([s1,s2])
plt.show()

print("s1 representa la Tarea 03 , s2 representa la Tarea 02")

x = stats.ranksums(s1, s2)
if x.pvalue <= 0.05:
    print('Gana s1')
elif (1 - x.pvalue) <= 0.05:
    print('Gana s2')
else:
    print('Empate')