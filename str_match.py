#!/usr/bin/env python
# coding: utf-8

# In[23]:


'''
IMPORT LIBRARIES
'''
import pandas as pd
from typing import List, Tuple, Union
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Input, Concatenate


# In[69]:


'''
DATA PREPROCESSING
1. Load data
2. Divide data into train, validation, test with the first 70%, then 15%, then 15%
3. Tokenize and pad the data
'''
filename = 'genomic_dataset_sr_train.txt'
df = pd.read_csv(filename, sep='\t', header=None, names=['1st_seq', '2nd_seq', 'label'])


# In[70]:


# separate data into train, validation, test
train_data = df[:int(len(df)*0.7)]
validate_data = df[int(len(df)*0.7):int(len(df)*0.85)]
test_data = df[int(len(df)*0.85):]

# divide data into different lists
seq1_list = train_data['1st_seq'].tolist()
seq2_list = train_data['2nd_seq'].tolist()
label_list = train_data['label'].values


# In[17]:


# tokenize and pad data
def data_preprocess(seq1_lst: List[str], seq2_lst: List[str]) -> Tuple[List[int], List[int], int, int]:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seq1_lst + seq2_lst)
    tokenized_seq1_list = tokenizer.texts_to_sequences(seq1_lst)
    tokenized_seq2_list = tokenizer.texts_to_sequences(seq2_lst)

    vocab_size = len(tokenizer.word_index) + 1
    max_len = max([len(seq) for seq in tokenized_seq1_list + tokenized_seq2_list])

    padded_seq1_list = pad_sequences(tokenized_seq1_list, maxlen=max_len, padding='post')
    padded_seq2_list = pad_sequences(tokenized_seq2_list, maxlen=max_len, padding='post')
    
    return padded_seq1_list, padded_seq2_list, vocab_size, max_len


# In[ ]:


padded_seq1_list, padded_seq2_list, vocab_size, max_len = data_preprocess(seq1_list, seq2_list)


# In[97]:


'''
MODEL TRAINING
1. Add embedding layer
2. Add LSTM layer
3. Add Dense layer
4. Compile model
5. Fit model
6. Save/Load model
'''

#input 
X = [padded_seq1_list, padded_seq2_list]
y = label_list


# In[99]:


# setup dimensions
input_dim = max_len
output_dim = 1

# setup model
input1 = Input(shape=(None,))
input2 = Input(shape=(None,))

#embedding layer
embed1 = Embedding(
    input_dim=vocab_size,
    output_dim=128,
) (input1)
embed2 = Embedding(
    input_dim=vocab_size,
    output_dim=128,
) (input2)

#concatenate layer
concat_layer = Concatenate()([embed1, embed2])

#bi-lstm layer
bi_lstm_layer = Bidirectional(
    LSTM(64, return_sequences=True)
) (concat_layer)
bi_lstm_layer = Bidirectional(
    LSTM(64)
) (bi_lstm_layer)
output_layer = Dense(output_dim, activation='sigmoid') (bi_lstm_layer)

model = Model(inputs=[input1, input2], outputs=output_layer)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[111]:


# Setup checkpoint
class CustomModelCheckpoint(Callback):
    def on_epoch_end(self, epoch: int, logs=None):
        print('Epoch:', epoch)
        print('Saving model...')
        self.model.save('model.keras', overwrite=True)
            
custom_checkpoint = CustomModelCheckpoint()


# In[ ]:


'''
CHOOSE EITHER ONE OF THE FOLLOWING:
1. Load trained-model and continue training
2. Train new model from scratch
'''

#! 1. Load trained-model
def load_model(filename: str = 'model.keras'):
    #load model from file
    model = load_model(filename)
    epochs = 10
    batch_size = 32

    #fit model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[custom_checkpoint])


# In[112]:


#! 2. TRAIN NEW MODEL

print('Training info: ')
# print(model.summary())

epochs = 10
batch_size = 256
# model training
model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[custom_checkpoint])


# In[24]:


'''
MODEL PREDICT
1. Load model
2. Load data
3. Predict
4. Save result
5. Analyze result
'''

def predict(filename: str='model.h5') -> List[Union[bool, float]]:
    model = load_model(filename)
    
    sample_data = (
        ['test_sequence_1'], 
        ['test_sequence_2']
    )
    
    processed_data = data_preprocess(sample_data[0], sample_data[1])
    X_test = list(processed_data[:2])
    
    predictions = model.predict(X_test) # output as list of matching probability
    
    
    threshold = 0.5
    bool_predictions = [True if prediction > threshold else False for prediction in predictions]
        
    return predictions
    # return bool_predictions

print(predict(filename='model.keras'))

