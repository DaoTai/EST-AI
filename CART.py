
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import pandas as pd
from datetime import datetime, timedelta


LIST_STRING_COLUMNS = ['school', 'name_course','language_course','suitable_job_course', 'love_language',"level_course"]
LIST_FEATURES = ['school', 'name_course','language_course', 'love_language' , 'average_score_course', 'distance_time',"level_course"]
LABEL = ['suitable_job_course']

data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

#Tiền xử lý
# Tạo các LabelEncoder cho các cột chuỗi
label_encoders = {}
for column in LIST_STRING_COLUMNS:
    label_encoders[column] = preprocessing.LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convert dữ liệu thời gian
now = datetime.now()
df['register_course_time'] = pd.to_datetime(data['register_course_time']).dt.tz_localize(None)
df['distance_time'] = (now - df['register_course_time']).dt.days    

# Chuẩn bị features và label
X = df[LIST_FEATURES]
Y = df[LABEL]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True )

# Xây dựng model
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

# Lấy bản ghi bất kì trong tập test
index_of_sample = 3
input_data = X_test.iloc[[index_of_sample]]
print("\t*****Input*****")
decoded_row = {}
for column in LIST_FEATURES:
    if column in label_encoders:
        label_encoder = label_encoders[column]
        value = X_test[column].iloc[index_of_sample]
        decoded_value = label_encoder.inverse_transform([value])[0]
        decoded_row[column] = decoded_value
        
# print(decoded_row)

real_label = y_test.iloc[[index_of_sample]]
decoded_real_label = label_encoders['suitable_job_course'].inverse_transform(real_label.iloc[0])[0]
# print("Thực tế: ",decoded_real_label)


# Dự đoán
prediction = model.predict(input_data)
decoded_prediction = label_encoders['suitable_job_course'].inverse_transform(prediction)

print(f"Dự đoán: {decoded_prediction[0]}")

# Đánh giá mô hình

print("\t*****Đánh giá mô hình*****")
y_pred = model.predict(X_test)
precision = metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = metrics.recall_score(y_test, y_pred,average='macro', zero_division=0)
print("Precision:", precision)
print("Recall:", recall)
F1 = metrics.f1_score(y_test, y_pred,average='macro')
print("F1: ", F1)




# Dự đoán
# Nhập dữ liệu từ người dùng để dự đoán
# try:
#     school_input = input("Nhập trường học: ")
#     name_course_input = input("Nhập tên khóa học: ")
#     language_course_input = input("Nhập ngôn ngữ khóa học: ")
#     love_language_input = input("Nhập ngôn ngữ lập trình yêu thích: ")
#     average_score_input = float(input("Nhập điểm trung bình khóa học: "))
#     register_date_input = input("Nhập ngày đăng ký khóa học : ")
# except:
#     print("Invalid input data")

# # Mã hoá dữ liệu đầu vào từ người dùng
# encoded_school = label_encoders['school'].transform([school_input])[0]
# encoded_name_course = label_encoders['name_course'].transform([name_course_input])[0]
# encoded_language_course = label_encoders['language_course'].transform([language_course_input])[0]
# encoded_love_language = label_encoders['love_language'].transform([love_language_input])[0]

# # Format time input
# input_datetime = datetime.strptime(register_date_input, "%m/%d/%Y")
# distance_time_input = (now - input_datetime).days
# # Tạo dữ liệu đầu vào mới để dự đoán
# new_data = {
#     'school': [encoded_school],
#     'name_course': [encoded_name_course],
#     'language_course': [encoded_language_course],
#     'love_language': [encoded_love_language],
#     'average_score_course': [average_score_input],
#     'distance_time': [distance_time_input]
# }


# # Tạo DataFrame từ dữ liệu mới
# new_df = pd.DataFrame(new_data)

# # Dự đoán bằng mô hình đã huấn luyện
# prediction = model.predict(new_df)
# decoded_prediction = label_encoders['suitable_job_course'].inverse_transform(prediction)
# print(f"Dự đoán về ngành nghề phù hợp: {decoded_prediction[0]}")




    
