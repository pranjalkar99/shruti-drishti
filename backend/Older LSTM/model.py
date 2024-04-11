from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


from preprocess import X_train,y_train,X_test,y_test
from make_folders import actions
import os
import numpy as np

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))




model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.fit(X_train, y_train, epochs=150, callbacks=[tb_callback])


print(model.summary())

res=model.predict(X_test)

print(actions[np.argmax(res[0])],"vs")

print(y_test[0])


model.save('actionv2new.h5')




