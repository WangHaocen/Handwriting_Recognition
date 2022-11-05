#!/usr/bin/env python
# coding: utf-8

# In[336]:


from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2

np.random.seed(42)
tf.random.set_seed(42)


# In[337]:


def GenerateOffLine():
    with open("./data/offline/words.txt") as f:
        line = f.readlines()
    
    lines = []
    for l in line:
        if(l.startswith("#")):
            continue
        else:
            lines.append(l)

    new_lines = []
    for i in range(len(lines)):
        line = lines[i]
        splits = line.split(' ')
        status = splits[1]

        if status == 'ok':
            new_lines.append(lines[i])
                        
        idx = int(0.9 * len(new_lines))
        train_samples = new_lines[:idx]
        test_samples = new_lines[idx:]
        val_idx = int(0.5 * len(test_samples))
        validation_samples = test_samples[:val_idx]
        test_samples = test_samples[val_idx:]
        
    return train_samples,test_samples,validation_samples


# In[341]:


def get_samples(samples):
    paths = []
    labels = []
    for i in range (len(samples)):
        s = samples[i]
        s = s.split(" ")
        file = s[0]
        label = s[len(s)-1]
        label = label.split("\n")[0]
        file_path = file.split("-")
        img_path = "./data/offline/iam/" + file_path[0] + "/" + file_path[0] + "-" + file_path[1] + "/" + file + ".png"
        if os.path.getsize(img_path):
            paths.append(img_path)
            labels.append(label)
            
    return paths, labels


# In[342]:




max_len = 21
characters = {'W', 'Z', 'h', 'R', ')', 'I', 'U', "'", 'S', ':', 'p', 'q', 't', 'r', 's', 'g', '?', 'D', '!', '0', 'N', 'H', 'X', 'e', 'V', 'y', 'n', '7', '/', 'E', '*', '1', 'Q', '5', 'T', 'i', 'o', 'z', '2', '3', 'M', '&', 'O', 'F', '.', 'Y', 'u', 'l', 'J', 'm', ',', 'w', '9', '"', 'f', 'C', 'a', 'x', '+', 'A', 'j', '-', 'v', 'B', '#', 'L', 'P', 'd', '(', ';', 'G', '4', '6', 'K', 'k', 'b', '8', 'c'}


# In[346]:


from tensorflow.keras.layers.experimental.preprocessing import StringLookup


# In[347]:


AUTOTUNE = tf.data.AUTOTUNE
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


# In[348]:


def distortion_free_resize(image, img_size):
   
    w, h = img_size
    image  = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    
    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
    
    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2
   
    image = tf.pad(
        image,
        paddings=[
                  [pad_height_top, pad_height_bottom],
                  [pad_width_left, pad_width_right],
                  [0, 0]
                ]
        )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


# In[349]:


batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


# In[350]:


from deslant_img import deslant_img





# In[354]:




# In[356]:


def get_image(image):
    img = image
    return img

def get_label(label):
    label_ = label
    return label_


def get_dataset(image_path, label):
    image = get_image(image_path)
    label = get_label(label)
    return {"xs": image, "ys": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        get_dataset, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


# In[357]:




# In[22]:


from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Reshape,BatchNormalization, Activation, Input, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[23]:


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


# In[24]:


class Offline_Model(object):
    def __init__(self,preload):
        self.model = self.get_model()
        self.pred_model = self.get_premodel("softmax")
        self.compile()
        
        if preload:
            self.pretrained = "./model/offline/offline_without_children_CNN2_batch64_blstm.h5"
            print("preloading model weights from" + self.pretrained)
            self.load_weights(file_name=self.pretrained)
            
    def get_premodel(self, layer_name):
        pre_model = Model(inputs=self.model.get_layer("xs").output,
                         outputs=self.model.get_layer(layer_name).output)
       
        optimizer = Adam(learning_rate=0.001)
        pre_model.compile(loss={layer_name: lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        return pre_model
    
    def get_model(self):
        input_shape = (image_width,image_height,1)
        inputs =  keras.Input(shape=input_shape, name="xs")
        labels =  keras.layers.Input(name="ys", shape=(None,))

        conv2d_1 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal",padding="same",name="Conv1",)(inputs)
        batch_1 = BatchNormalization()(conv2d_1)
        relu_1 = keras.layers.Activation('relu')(batch_1)
        pool_1 = MaxPooling2D((2, 2), name="pool1")(relu_1)
    
        conv2d_2 =  Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2",)(pool_1)
        batch_2 = BatchNormalization()(conv2d_2)
        relu_2 = keras.layers.Activation('relu')(batch_2)
        pool_2 = keras.layers.MaxPooling2D((2, 2), name="pool2")(relu_2)
    
        new_shape = ((image_width // 4), (image_height // 4) * 64)
        reshape = Reshape(target_shape=new_shape, name="reshape")(pool_2)
        dense =  Dense(64, activation="relu", name="dense1")(reshape)
        dropout =  Dropout(0.2)(dense)
        
        blstm_1 =  keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(dropout)
        blstm_2 =  keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(blstm_1)
        blstm_3 =  keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(blstm_2)
        
        dense_2 =  Dense(len(char_to_num.get_vocabulary()) + 2, name="dense2")(blstm_3)
        y_pred = Activation('softmax', name='softmax')(dense_2)
    
        output = CTCLayer(name="ctc_loss")(labels, y_pred)

        model = Model(inputs=[inputs, labels], outputs=output)
        return model
    
    def fit(self, train_seq, test_seq, epochs=100, earlystop=10):
        
        filepath="offline_without_children_CNN2_blstm.h5"
        early = tf.keras.callbacks.EarlyStopping(patience=earlystop)

        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.history = self.model.fit(
            train_seq,
            validation_data=test_seq,
            shuffle=True,
            verbose=1,
            epochs=epochs,
            callbacks=[checkpoint, early]
        )
        
    def get_history(self):
        return self.history
    
    def compile(self):
        optimizer = Adam()
        self.model.compile(optimizer=optimizer)
        
    def save_weights(self, file_name=None):
        self.model.save_weights(file_name)

    def load_weights(self, file_name=None):
        self.model.load_weights(file_name)
        self.compile()
        
    def predict(self,eval_data):
        pred = self.model.predict(eval_data)
        return pred
    
    def get_model_summary(self):
        return self.model.summary()


# In[358]:




def decode_batch_predictions(pred=None, top_n=1):
    pred = pred
    top_n = top_n
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
   # Use greedy search. For complex tasks, you can use beam search.
   
    if(top_n>1):
        results_beam = []
        for i in range(top_n):
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False,beam_width=25,top_paths=5)[0][i][
                :, :max_len
            ]
    # Iterate over the results and get back the text.
            output_text = []
            for res in results:
                res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
                res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
                output_text.append(res)
            results_beam.append(output_text)
        return results_beam
    
    elif(top_n==1):
        results_beam = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False,beam_width=25,top_paths=1)[0][0][
              :, :max_len]
    # Iterate over the results and get back the text.
        output_beam = []
        for res in results_beam:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_beam.append(res)
            
        results_greedy = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_len]
    # Iterate over the results and get back the text.
        output_greedy = []
        for res in results_greedy:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_greedy.append(res)
            
        return output_beam, output_greedy








