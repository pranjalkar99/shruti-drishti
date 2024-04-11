
from model import model
from preprocess import X_test,y_test
from make_folders import actions


import numpy as np


res = model.predict(X_test)
print(actions[np.argmax(res[4])])
print(actions[np.argmax(y_test[4])])