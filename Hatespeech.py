import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jieba
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import pandas
import numpy
import jieba
import pandas as pd
from sklearn.preprocessing import LabelEncoder


train_file = 'train.json'
test_file = 'test1.json'

df_train = pd.read_json(train_file)
df_test = pd.read_json(test_file)


hate = 'hate'
Nhate = 'non-hate'

# 定义解析函数：分词+存储分词结果
def parse_output(output):
    quadruples = output.split('[SEP]')  # 使用字符串 '[SEP]' 作为分隔符
    results = []
    for quad in quadruples:
        quad = quad.strip()  # 去除前后空格
        if quad.endswith('[END]'):  # 检查是否以 [END] 结尾
            quad = quad[:-5].strip()  # 去除 [END]
        parts = quad.split('|')  # 使用 | 分割四元组
        if len(parts) == 4:  # 确保四元组有四个部分
            target, argument, group, hateful = [part.strip() for part in parts]
            results.append({
                'Target': target,
                'Argument': argument,
                'Targeted Group': group,
                'Hateful': hateful
            })
    return results

solved_train_output = list(map(parse_output, df_train['output']))


print(solved_train_output[0])

# 构建结构化数据集
def build_structured_data(df_train):
    structured_data = []
    for idx, row in df_train.iterrows():
        content = row['content']
        output = row['output']
        quadruples = parse_output(output)
        for quad in quadruples:
            structured_data.append({
                'id': row['id'],
                'content': content,
                'Target': quad['Target'],
                'Argument': quad['Argument'],
                'Targeted Group': quad['Targeted Group'],
                'Hateful': quad['Hateful']
            })
    return pd.DataFrame(structured_data)

# 示例
train_structured_df = build_structured_data(df_train)

print(train_structured_df.columns)


def is_hate(output_data):
    hate_count = 0
    Nhate_count = 0
    for hateful_value in output_data['Hateful']:  # 遍历 'Hateful' 列
        if hateful_value == 'hate':
            hate_count += 1
        elif hateful_value == 'non-hate':
            Nhate_count += 1
    return [hate_count, Nhate_count]

# 示例
print('hate_count:', is_hate(train_structured_df)[0])
print('non-hate_count:', is_hate(train_structured_df)[1])


#数据编码
label_encoder_group = LabelEncoder()
label_encoder_hateful = LabelEncoder()

train_structured_df['Targeted Group Encoded'] = label_encoder_group.fit_transform(train_structured_df['Targeted Group'])
train_structured_df['Hateful Encoded'] = label_encoder_hateful.fit_transform(train_structured_df['Hateful'])



print(train_structured_df[['Targeted Group', 'Targeted Group Encoded']].drop_duplicates())
print(train_structured_df[['Hateful', 'Hateful Encoded']].drop_duplicates())


print(train_structured_df.isnull().sum())
print(train_structured_df.duplicated().sum())

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import jieba

# 多标签编码
mlb = MultiLabelBinarizer()
targeted_group_encoded = mlb.fit_transform(train_structured_df['Targeted Group'].str.split(', '))
targeted_group_df = pd.DataFrame(targeted_group_encoded, columns=mlb.classes_)
train_structured_df = pd.concat([train_structured_df, targeted_group_df], axis=1)

# 分词
train_structured_df['content_tokenized'] = train_structured_df['content'].apply(lambda x: ' '.join(jieba.lcut(x)))
train_structured_df['Target_tokenized'] = train_structured_df['Target'].apply(lambda x: ' '.join(jieba.lcut(x)))
train_structured_df['Argument_tokenized'] = train_structured_df['Argument'].apply(lambda x: ' '.join(jieba.lcut(x)))

# 打印结果
print(train_structured_df.head())


# 计算每条样本分词后的 token 数量
train_structured_df['content_length'] = train_structured_df['content_tokenized'].apply(lambda x: len(x.split()))

# 绘制箱线图，直观展示数据分布
plt.figure(figsize=(8,4))
plt.boxplot(train_structured_df['content_length'])
plt.title("Content Tokenized Length Boxplot")
plt.ylabel("Token Count")
plt.show()

# 使用3σ法则检测异常值：计算均值和标准差
mean_length = train_structured_df['content_length'].mean()
std_length = train_structured_df['content_length'].std()
lower_bound = mean_length - 3 * std_length
upper_bound = mean_length + 3 * std_length
print("Mean content length:", mean_length)
print("Std content length:", std_length)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

# 标记出异常值（低于下界或高于上界的样本）
outliers = train_structured_df[(train_structured_df['content_length'] < lower_bound) | (train_structured_df['content_length'] > upper_bound)]
print("Number of outliers detected:", len(outliers))

# 选择特征和标签（使用经过分词的文本）
X = train_structured_df['content_tokenized']
y = train_structured_df['Hateful Encoded']

# 使用 TfidfVectorizer 转换文本数据，限制最大特征数以便解释
vectorizer = TfidfVectorizer(max_features=500)
X_vec = vectorizer.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 验证集准确率
accuracy = model.score(X_val, y_val)
print("Validation Accuracy:", accuracy)

# 使用 SHAP 评估特征重要性
# 对于线性模型可用 LinearExplainer，注意这里使用了“interventional”方式进行特征扰动
explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_val)

# 可视化 SHAP 值，总结最重要的特征（最多显示10个特征）
shap.summary_plot(shap_values, X_val, feature_names=vectorizer.get_feature_names_out(), max_display=10)


#原始数据
X_raw = train_structured_df['content']  # 原始文本
vectorizer_raw = TfidfVectorizer(max_features=500)
X_raw_vec = vectorizer_raw.fit_transform(X_raw)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_raw_vec, y, test_size=0.2, random_state=42)
model_raw = LogisticRegression(max_iter=1000)
model_raw.fit(X_train_raw, y_train_raw)
accuracy_raw = model_raw.score(X_val_raw, y_val_raw)
print("Validation Accuracy (Raw Data):", accuracy_raw)

#清理后的数据
X_clean = train_structured_df['content_tokenized']
vectorizer_clean = TfidfVectorizer(max_features=500)
X_clean_vec = vectorizer_clean.fit_transform(X_clean)
X_train_clean, X_val_clean, y_train_clean, y_val_clean = train_test_split(X_clean_vec, y, test_size=0.2, random_state=42)
model_clean = LogisticRegression(max_iter=1000)
model_clean.fit(X_train_clean, y_train_clean)
accuracy_clean = model_clean.score(X_val_clean, y_val_clean)
print("Validation Accuracy (Cleaned Data):", accuracy_clean)


print("Accuracy Improvement:", accuracy_clean - accuracy_raw)
