#!/usr/bin/env python
# coding: utf-8

# # 使用RNN分析IMDB電影評論的情意分析

# In[1]:


# 架構KERAS環境
get_ipython().run_line_magic('env', 'KERAS_BACKEND=tensorflow')


# In[2]:


# 讀入IMDB電影數據庫
from keras.datasets import imdb

# 只留下前1萬筆常見資料
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)


# In[4]:


# 檢查輸入資料長度
for i in range(10):
    print(len(x_train[i]), end=',')


# In[5]:


# 檢查輸出資料
for i in range(10):
    print(y_train[i], end=',')


# In[8]:


# 送入神經網路前的輸入處理
from keras.preprocessing import sequence

# 設定輸入文字長度上限100
# 把文字弄成一樣長，太短捕0
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)


# In[9]:


# 用LSTM打造RNN
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

model = Sequential()

# 1.將1萬組文字壓成128維度
model.add(Embedding(10000,128))

# 2.最後使用150個LSTM(此與維度數量無關)
model.add(LSTM(150))

# 3.產出用sigmoid送出0~1之間的值
model.add(Dense(1,activation='sigmoid'))


# In[10]:


# 檢視summary
model.summary()


# In[13]:


# 組裝並用adam學習法
model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])


# In[14]:


# 訓練模型
model.fit(x_train, y_train, batch_size=32, epochs=15)


# In[15]:


# 檢視訓練成果
# loss和分數
score = model.evaluate(x_test, y_test)
print('loss:',score[0])
print('score:',score[1])


# 準確率達82%，可以使用

# In[16]:


# 儲存model
model.json = model.to_json()
open('imdb_rnn_model.json','w').write(model.json)

# 儲存權重
model.save_weights('imdb_model_weights.h5')


# In[17]:


# 另一種儲存方式
model.save('imdb_rnn_model.h5')


# In[ ]:




