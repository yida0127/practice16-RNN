# practice16-RNN
use RNN to analyze sentiment from IMDB database

使用遞迴神經網路(Recurrent Neural Network)判讀IMDB電影評論的情意分析
1. 初始準備 
   - 讀入tensorflow環境
        %env KERAS_BACKEND=tensorflow
2. 讀入IMDB資料庫
        設定num_words=10000出現前10000筆常見文字
   - 送入神經網路的輸入處理
   - 設定輸入文字長度的上限
   - 把文字弄成一樣長，太短則補0
        from keras.preprocessing import sequence
        x_train = sequence.pad_sequences(x_train, maxlen=100)
        x_test = sequence.pad_sequences(x_test, maxlen=100)
3. 打造神經網路
   - 決定神經網路架構並讀入相關套件
        from keras.models import Sequential
        from keras.layers import Dense, Embedding, LSTM
   - 將10000組文字壓到128維
   - 最後用150個LSTM
   - 產出用sigmoid產生0~1之間的值
   - 建構神經網路
4. 檢視成果
   - model.summary()
5. 訓練神經網路
   - 將x_train, y_train丟入模型中訓練 
        model.fit(x_train,y_train,batch_size=32,epochs=15)
   - batch_size 每次訓練的資料量
   - epochs 訓練次數
6. 試用成果
   - predict = model.predict_classes(x_test)
7. 將訓練好的神經網路分別存下來
   - 存本體 
        model_json = model.to_json()
        open('XXXX.json','w').write(model.json)
   - 存權重 
        model.save_weights('XXXX.h5')
   - 快速儲存方式
        model.save('XXXX.h5')
