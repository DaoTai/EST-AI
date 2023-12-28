import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
# Định nghĩa các trường trong file csv:
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
    df['register_course_time'] = pd.to_datetime(df ['register_course_time']).dt.tz_localize(None)
    df['distance_time'] = (now - df['register_course_time']).dt.days
    # Chuẩn bị FEATURES và LABEL
    X = df[LIST_FEATURES].astype('float32')
    Y = df[LABEL]

    # Convert labels to one-hot encoding
    Y = to_categorical(Y)
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True )

    # Xây dựng mô hình LSTM
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))  # Output has the same number of classes as the labels

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, shuffle=True, validation_data=(X_test, y_test))

    # #Predict record in test size
    y_pred = model.predict(X_test)
    #  Đánh giá mô hình
    loss, accuracy = model.evaluate(X_test, y_test)

    # print("\t*****Đánh giá mô hình*****")
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Chuyển đổi kết quả dự đoán thành nhãn được encode
    y_pred_encoded = np.argmax(y_pred, axis=1)
    y_test_encoded = np.argmax(y_test, axis=1)

    # Tính các độ đo precision, recall và F1-score
    precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')

    print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    # ====================Dữ liệu thực tế đầu vào
    dfMyAvgScores = pd.DataFrame(myAvgScores)
    dfMyAvgScores['register_course_time'] = pd.to_datetime(dfMyAvgScores ['register_course_time']).dt.tz_localize(None)
    dfMyAvgScores['distance_time'] = (now - dfMyAvgScores['register_course_time']).dt.days
    
     # Chuẩn bị FEATURES và LABEL
    unseen_values = {}
    for column in LIST_STRING_COLUMNS:
        label_encoder = label_encoders[column]
        # Lấy ra danh sách giá trị chưa được mã hoá ở tập dataset mã hoá
        unseen_vals = dfMyAvgScores[~dfMyAvgScores[column].isin(label_encoder.classes_)][column].unique()
        unseen_values[column] = unseen_vals.tolist()
        if unseen_vals.size > 0:
            # Lấy giá trị mã hoá lớn nhất tại column đang xét
            max_label = max(label_encoder.transform(label_encoder.classes_))
            # Lấy danh sách các lớp hiện tại
            existing_classes = label_encoder.classes_.tolist()
            # Mở rộng danh sách các lớp
            existing_classes.extend(unseen_vals)
            # Sắp xếp lại và loại bỏ các giá trị trùng lặp
            existing_classes = sorted(set(existing_classes))
            # Cập nhật lại danh sách các lớp trong encoder
            label_encoder.classes_ = np.array(existing_classes)
            # Mã hoá lại cột tương ứng trong dataframe
            dfMyAvgScores[column] = max_label + 1
            # print(f"Column {column}: ", dfMyAvgScores[column])
        else:
            dfMyAvgScores[column] = label_encoder.transform(dfMyAvgScores[column])
      
        
    myFeatures = dfMyAvgScores[LIST_FEATURES].astype('float32')
    myLabels = dfMyAvgScores[LABEL]
    predicted_jobs = []
    for index,row in myFeatures.iterrows():
        decoded_data={}
        # print("Encode_data: ", row)
        for column in LIST_FEATURES:
            if column in label_encoders:
                label_encoder = label_encoders[column]
                value = row[column]
                decoded_value = label_encoder.inverse_transform([int(value)])[0]
                decoded_data[column] = decoded_value
            else:
                decoded_data[column] = row[column]
        # print("decoded_data: ",decoded_data, flush=True)
        prediction = model.predict([row])
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = label_encoders['suitable_job_course'].inverse_transform([predicted_class])[0]
        # print("result: ",result,flush=True)
        predicted_jobs.append(result)

    return predicted_jobs


