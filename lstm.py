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

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)


# Tiền xử lý dữ liệu
# Tạo các LabelEncoder cho các features chuỗi
label_encoders = {}
for column in LIST_STRING_COLUMNS:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convert dữ liệu thời gian
now = datetime.now()
df['register_course_time'] = pd.to_datetime(data['register_course_time']).dt.tz_localize(None)
df['distance_time'] = (now - df['register_course_time']).dt.days
max_value = df['distance_time'].max()
# Chuẩn bị FEATURES và LABEL
X = df[LIST_FEATURES]
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

#Predict record in test size
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

# Lấy bản ghi random làm dữ lệu test
random_index = 2
input_data = X_test.iloc[[random_index]]

print("\t*****Input*****")
decoded_row = {}
for column in LIST_FEATURES:
    if column in label_encoders:
        label_encoder = label_encoders[column]
        value = X_test[column].iloc[random_index]
        decoded_value = label_encoder.inverse_transform([value])[0]
        decoded_row[column] = decoded_value
print(decoded_row)

real_label = y_test[[random_index]]
real_label_index = np.argmax(real_label, axis=1)[0]
real_result = label_encoders['suitable_job_course'].inverse_transform([real_label_index])[0]
print("Real label: ", real_result)

# Dự đoán 
predict_label = model.predict(input_data)
predict_label_index = np.argmax(predict_label, axis=1)[0] 
result = label_encoders['suitable_job_course'].inverse_transform([predict_label_index])[0]

print("Predicted label:", result)