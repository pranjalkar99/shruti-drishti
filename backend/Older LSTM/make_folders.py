# Path for exported data, numpy arrays
import os
import numpy as np

DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'how are you', 'namaste', 'awesome', 'thank you'])#,'again','i/me','you','man','woman','he','she','deaf','hearing','teacher','thank you very much','welcome','please','sorry','namaste','how are you','i\'m fine','my name is','practice'])

# Thirty videos worth of data
no_sequences = 60

# Videos are going to be 30 frames in length
sequence_length = 60

# Folder start
start_folder = 60


for action in actions: 
    # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
