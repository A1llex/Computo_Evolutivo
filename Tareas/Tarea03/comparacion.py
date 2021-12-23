# Comparativa entre dos algoritmos evolutivos
# Utilizando la prueba de Wilcoxon rank sum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


#Los evolutivos en eggholder
s1 = 0
s2 = 0

fig2, ax2 = plt.subplots()
ax2.boxplot(np.concatenate((s1, s2)))
plt.show()
print("temina con las gráficas")
print(stats.ranksums(s1, s2, alternative='two-sided'))
print(stats.ranksums(s1, s2, alternative='less'))
print(stats.ranksums(s1, s2, alternative='greater'))


x = stats.ranksums(s1, s2, alternative='less')
if stats.ranksums(s1, s2, alternative='less').pvalue <= 0.05:
    print('Gana s1')
elif stats.ranksums(s2, s1, alternative='less').pvalue <= 0.05:
    print('Gana s2')
else:
    print('Empate')