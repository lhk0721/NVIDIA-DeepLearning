from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

mnist = load_digits()

# print(mnist['data'][:3])
# [[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#   15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#    0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#    0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
#  [ 0.  0.  0. 12. 13.  5.  0.  0.  0.  0.  0. 11. 16.  9.  0.  0.  0.  0.
#    3. 15. 16.  6.  0.  0.  0.  7. 15. 16. 16.  2.  0.  0.  0.  0.  1. 16.
#   16.  3.  0.  0.  0.  0.  1. 16. 16.  6.  0.  0.  0.  0.  1. 16. 16.  6.
#    0.  0.  0.  0.  0. 11. 16. 10.  0.  0.]
#  [ 0.  0.  0.  4. 15. 12.  0.  0.  0.  0.  3. 16. 15. 14.  0.  0.  0.  0.
#    8. 13.  8. 16.  0.  0.  0.  0.  1.  6. 15. 11.  0.  0.  0.  1.  8. 13.
#   15.  1.  0.  0.  0.  9. 16. 16.  5.  0.  0.  0.  0.  3. 13. 16. 16. 11.
#    5.  0.  0.  0.  0.  3. 11. 16.  9.  0.]]

# print(len(mnist['data']))
# 1797

# print(mnist['target'])
# [0 1 2 ... 8 9 8]

features = mnist['data']
labels = mnist['target']

from sklearn.model_selection import cross_validate

RFmodel = RandomForestClassifier()
RF_scores = cross_validate(RFmodel, features, labels, cv = 10)
print(RF_scores['test_score'])
# [0.90555556 0.95555556 0.91666667 0.91666667 0.96666667 0.96111111
#  0.97777778 0.96089385 0.91620112 0.94413408]

DT_scores = cross_validate(tree.DecisionTreeClassifier(), features, labels, cv=10)
print(DT_scores['test_score'])
# [0.77777778 0.85555556 0.84444444 0.79444444 0.76111111 0.87222222
#  0.88888889 0.82681564 0.82681564 0.81564246]

import pandas as pd
df = pd.DataFrame(
    {'random_forest':RF_scores['test_score'],
    'decision_tree': DT_scores['test_score']
    }
)

# print(df)
#    random_forest  decision_tree
# 0       0.894444       0.788889
# 1       0.972222       0.833333
# 2       0.933333       0.822222
# 3       0.927778       0.772222
# 4       0.966667       0.805556
# 5       0.961111       0.883333
# 6       0.972222       0.872222
# 7       0.966480       0.849162
# 8       0.916201       0.832402
# 9       0.944134       0.798883

df.plot()
plt.savefig('cross_validate_RF_and_DT.jpeg')

