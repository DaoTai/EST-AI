import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score


# Đọc dữ liệu từ file CSV
data = pd.read_csv('mock_data.csv')
df = pd.DataFrame(data)


# Tiền xử lý dữ liệu
label_encoders = {}
for column in ['school', 'name_course', 'average_score_course', 'language_course','suitbale_job_course']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convert dữ liệu thời gian
now = datetime.now()
df['register_course'] = pd.to_datetime(data['register_course'])
df['time'] = (now - df['register_course']).dt.days


# Chuẩn bị dữ liệu đầu vào và đầu ra
X = df[['school', 'name_course', 'average_score_course', 'language_course', 'time']]
Y = df['suitbale_job_course']

max_value = df['time'].max()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(Embedding(input_dim=max_value +1, output_dim=64))
 #input_dim: số lượng từ vựng trong dữ liệu văn bản đang sử dụng  xác định kích thước của ma trận trọng số trong lớp nhúng và sẽ bằng số từ vựng khác nhau trong dữ liệu. Vậy nếu thêm FIELD['time'] giá trị khoảng cách (ngày) sẽ có thể lớn hơn số lượng bản ghi nên tránh lỗi compile thì dùng giá trị lớn nhất của trường này (max_value +1)
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# # Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Dự đoán trên tập kiểm tra
predictions = model.predict(X_test)

# Chuyển đổi dự đoán thành nhãn dự đoán (ví dụ: argmax cho mô hình classification)
predicted_classes = np.argmax(predictions, axis=1)

# Tính toán precision, recall và F1-score
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')
f1 = f1_score(y_test, predicted_classes, average='weighted')

# ===================================
# Nhập dữ liệu mới từ bàn phím
school = input('Nhập trường học: ')
name_course = input('Nhập tên khoá học: ')
average_score_course = float(input('Nhập điểm trung bình: '))
language_course = input('Nhập ngôn ngữ lập trình: ')
time_delta = int(input('Nhập khoảng cách thời gian (ngày): '))  # Nếu thêm trường 'time'

# Mã hoá dữ liệu đầu vào từ bàn phím
encoded_school = label_encoders['school'].transform([school])[0]
encoded_name_course = label_encoders['name_course'].transform([name_course])[0]
encoded_language_course = label_encoders['language_course'].transform([language_course])[0]

# Tạo DataFrame cho dữ liệu mới
new_data = {
    'school': [encoded_school],
    'name_course': [encoded_name_course],
    'average_score_course': [average_score_course],
    'language_course': [encoded_language_course],
    'time': [time_delta] if 'time' in df.columns else None  # Kiểm tra xem trường 'time' có trong DataFrame không
}

new_df = pd.DataFrame(new_data)

# Dự đoán với mô hình LSTM
new_prediction = model.predict(new_df)

# Lấy nhãn tương ứng với giá trị dự đoán
predicted_class = int(round(new_prediction[0][0]))  # Lấy giá trị dự đoán (có thể cần làm tròn)

# Chuyển ngược từ nhãn đã dự đoán về tên ngành
predicted_course = label_encoders['suitbale_job_course'].inverse_transform([predicted_class])[0]

# In tên ngành dự đoán
print(f'Tên ngành phù hợp: {predicted_course}')
