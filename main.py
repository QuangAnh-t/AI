import numpy as np
import pandas as pd
from sklearn  import datasets


# tải tập dữ liệu iris từ thừ viện sklearn 
iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data , columns=iris.feature_names)
df['target'] = iris.target

# Hiển thị thông tin cơ bản về dữ liệu
print(df.head())

# Phân tích thống kê cơ bản
print(df.describe())

# Kiểm tra sự phân phối của các lớp trong cột target
print(df['target'].value_counts())