import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
# Định nghĩa các trường :
"""
    id: id người dùng
    school: trường học
    name_course: tên khoá học đăng ký
    level_course: cấp độ khoá học
    language_course: ngôn ngữ lập trình (công nghệ) sử dụng trong khoá học
    average_score_course: điểm bài tập trung bình của người dùng tại khoá học
    register_course_time: thời gian người dùng đăng ký khoá học
    love_language: ngôn ngữ lập trình yêu thích của người dùng
    distance_time: khoảng cách thời gian giữa hiện tại và thời gian đăng ký khoá học của người dùng
    suitable_job_course: lĩnh vực phù hợp của khoá học => Nhãn
"""
 # Các field có kiểu chuỗi
LIST_STRING_COLUMNS = ['school', 'name_course','language_course','suitable_job_course', 'love_language', "level_course"]
# Các thuộc tính để huấn luyện
LIST_FEATURES = ['school', 'name_course','language_course', 'love_language' , 'average_score_course', "level_course", 'distance_time']
# Nhãn
LABEL = ['suitable_job_course']
model_file = r'model.keras'

def run_predict(inputData,myAvgScores):
    df = pd.DataFrame(inputData)
    # Tiền xử lý dữ liệu
    # Tạo các LabelEncoder cho các features chuỗi
    label_encoders = {}
    for column in LIST_STRING_COLUMNS:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    # Convert dữ liệu thời gian
    now = datetime.now()
    df['register_course_time'] = pd.to_datetime(df['register_course_time']).dt.tz_localize(None)
    df['distance_time'] = (now - df['register_course_time']).dt.days
    # Chuẩn bị FEATURES và LABEL
    X = df[LIST_FEATURES].astype('float32')
    Y = df[LABEL]

    # Convert labels to one-hot encoding: convert value label to binary vector => Good for model
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

    # Build the LSTM model
    if os.path.exists(model_file):
        model = load_model(model_file, compile=True)
    else:
        model = Sequential()
        model.add(LSTM(100,input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(Dense(Y.shape[1], activation='softmax'))  # Y.shape[1] = number of classes of the label column
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, shuffle=True, validation_data=(X_test, y_test))
        model.save(model_file)
       
    try:
       loss, accuracy = model.evaluate(X_test, y_test)
    except:
        # Exception xảy ra khi số chiều của vector Y (Y.shape[1]) - số lớp của cột Y khác với tham số của model lưu vào file .keras trước đó
        model = Sequential()
        model.add(LSTM(100,input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(Dense(Y.shape[1], activation='softmax'))  # Y.shape[1] = number of classes of the label column
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, shuffle=True, validation_data=(X_test, y_test))
    
    #Summary model
    model.summary()  
    y_pred = model.predict(X_test)
    
    #  Đánh giá mô hình
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    # Chuyển label từ One-hot coding về dạng số: Lấy index có giá trị lớn nhất của từng vector con trong ma trận
    max_y_pred = np.argmax(y_pred, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    # Tính các độ đo precision, recall và F1-score
    precision = precision_score(max_y_test, max_y_pred, average='weighted', zero_division=0)
    recall = recall_score(max_y_test, max_y_pred, average='weighted', zero_division=0)
    f1 = f1_score(max_y_test, max_y_pred, average='weighted')
    print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    # === User's data ===
    dfMyAvgScores = pd.DataFrame(myAvgScores)
    dfMyAvgScores['register_course_time'] = pd.to_datetime(dfMyAvgScores ['register_course_time']).dt.tz_localize(None)
    dfMyAvgScores['distance_time'] = (now - dfMyAvgScores['register_course_time']).dt.days
     # Chuẩn bị FEATURES 
    unseen_values = {}
    for column in LIST_STRING_COLUMNS:
        label_encoder = label_encoders[column]
        # Lấy ra danh sách giá trị chưa được mã hoá ở tập dataset mã hoá
        unseen_vals = dfMyAvgScores[~dfMyAvgScores[column].isin(label_encoder.classes_)][column].unique()
        unseen_values[column] = unseen_vals.tolist()
        if unseen_vals.size > 0:
            # Lấy danh sách các lớp hiện tại và thêm dữ liệu unseen vào
            existing_classes = label_encoder.classes_.tolist()
            existing_classes.extend(unseen_vals)
            existing_classes = sorted(set(existing_classes))
            label_encoder.classes_ = np.array(existing_classes)
            
        dfMyAvgScores[column] = label_encoder.transform(dfMyAvgScores[column])
      
    myFeatures = dfMyAvgScores[LIST_FEATURES].astype('float32')
    # myLabels = dfMyAvgScores[LABEL]
    predicted_jobs = []
    for index,row in myFeatures.iterrows():
        decoded_data={}
        for column in LIST_FEATURES:
            if column in label_encoders:
                label_encoder = label_encoders[column]
                value = row[column]
                decoded_value = label_encoder.inverse_transform([int(value)])[0]
                decoded_data[column] = decoded_value
            else:
                decoded_data[column] = row[column]
        prediction = model.predict([row])
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = label_encoders['suitable_job_course'].inverse_transform([predicted_class])[0]
        predicted_jobs.append(result)

    return predicted_jobs


