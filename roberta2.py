import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re


train_file = 'train_limit.json'
test_file = 'test_limit.json'
model_save_path = 'roberta_hate_model.pt'
output_file = 'predictions777.txt'

df_train = pd.read_json(train_file)
df_test = pd.read_json(test_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 定义解析函数：解析四元组输出
def parse_output(output):
    quadruples = output.split('[SEP]')
    results = []
    for quad in quadruples:
        quad = quad.strip()
        if quad.endswith('[END]'):
            quad = quad[:-5].strip()
        parts = quad.split('|')
        if len(parts) == 4:
            target, argument, group, hateful = [part.strip() for part in parts]
            results.append({
                'Target': target,
                'Argument': argument,
                'Targeted Group': group,
                'Hateful': hateful
            })
    return results


# 构建训练数据并提取所有四元组信息
def build_training_data():
    all_data = []

    for idx, row in df_train.iterrows():
        content = row['content']
        output = row['output']

        # 解析四元组
        quadruples = parse_output(output)

        for quad in quadruples:
            target = quad['Target']
            argument = quad['Argument']
            group = quad['Targeted Group']
            hateful = quad['Hateful']

            # 构建样本
            all_data.append({
                'id': row['id'],
                'content': content,
                'Target': target,
                'Argument': argument,
                'Targeted Group': group,
                'Hateful': hateful
            })

    return pd.DataFrame(all_data)


# 处理训练集
train_data = build_training_data()
print(f"训练数据样本数量: {len(train_data)}")

# 关键词列表（用于Target-Argument对提取）
hate_keywords = {
    'Racism': ['黑人', '黑鬼', '老黑', 'nigger', 'negro', '黄种人', '白人', '种族', '肤色'],
    'Sexism': ['女权', '男权', '女性', '男性', '性别', '女人', '男人', '娘炮', '女拳', '女的'],
    'Region': ['东北', '北京', '上海', '广东', '河南', '地域', '口音', '乡下', '外地', '北方', '南方'],
    'LGBTQ': ['同性恋', '男同', '女同', '变性', '跨性别', '双性恋', '同志', 'LGBT', '娘炮'],
    'Others': ['残疾', '穷人', '富人', '农民', '城里人', '底层', '外国', '宗教', '信仰']
}

# 标签编码
target_group_encoder = LabelEncoder()
train_data['Targeted Group Encoded'] = target_group_encoder.fit_transform(train_data['Targeted Group'])
target_group_classes = target_group_encoder.classes_
print(f"目标群体类别: {target_group_classes}")

hateful_encoder = LabelEncoder()
train_data['Hateful Encoded'] = hateful_encoder.fit_transform(train_data['Hateful'])
hateful_classes = hateful_encoder.classes_
print(f"是否仇恨类别: {hateful_classes}")

# 加载预训练的中文RoBERTa tokenizer
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')


# 构建数据集
class HateDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        content = row['content']
        target = row['Target']
        argument = row['Argument']

        # 构建输入文本
        # 特殊格式：[CLS] 内容 [SEP] 目标: {target} 论点: {argument} [SEP]
        input_text = content
        if target != 'NULL':
            # 高亮文本中的Target和Argument
            for word in [target, argument]:
                if word in content:
                    # 确保我们不会替换已经高亮的文本
                    pattern = r'(?<!\[)\b' + re.escape(word) + r'\b(?!\])'
                    input_text = re.sub(pattern, f"<{word}>", input_text)

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target_group': torch.tensor(row['Targeted Group Encoded']),
            'hateful': torch.tensor(row['Hateful Encoded']),
            'content': content,
            'target': target,
            'argument': argument
        }


# 训练集和验证集分割
train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)
train_dataset = HateDataset(train_df, tokenizer)
val_dataset = HateDataset(val_df, tokenizer)

# 创建数据加载器
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# 定义多任务模型
class MultiTaskHateModel(torch.nn.Module):
    def __init__(self, num_group_labels, num_hateful_labels):
        super(MultiTaskHateModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'hfl/chinese-roberta-wwm-ext',
            num_labels=num_group_labels  # 暂时设置为group标签数
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.group_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_group_labels)
        self.hateful_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_hateful_labels)

    def forward(self, input_ids, attention_mask, group_labels=None, hateful_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1][:, 0]  # 使用[CLS]token的最后一层隐藏状态
        pooled_output = self.dropout(pooled_output)

        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)

        loss = None
        if group_labels is not None and hateful_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            group_loss = loss_fct(group_logits, group_labels)
            hateful_loss = loss_fct(hateful_logits, hateful_labels)
            loss = group_loss + hateful_loss

        return {
            'loss': loss,
            'group_logits': group_logits,
            'hateful_logits': hateful_logits
        }


# 初始化模型
model = MultiTaskHateModel(
    num_group_labels=len(target_group_classes),
    num_hateful_labels=len(hateful_classes)
)
model.to(device)

# 优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_dataloader) * 5  # 5个epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# 训练函数
def train_model(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        group_labels = batch['target_group'].to(device)
        hateful_labels = batch['hateful'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            group_labels=group_labels,
            hateful_labels=hateful_labels
        )

        loss = outputs['loss']
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    all_group_preds = []
    all_group_labels = []
    all_hateful_preds = []
    all_hateful_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            group_labels = batch['target_group'].to(device)
            hateful_labels = batch['hateful'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                group_labels=group_labels,
                hateful_labels=hateful_labels
            )

            loss = outputs['loss']
            total_loss += loss.item()

            # 预测
            group_preds = torch.argmax(outputs['group_logits'], dim=1).cpu().numpy()
            hateful_preds = torch.argmax(outputs['hateful_logits'], dim=1).cpu().numpy()

            all_group_preds.extend(group_preds)
            all_group_labels.extend(group_labels.cpu().numpy())
            all_hateful_preds.extend(hateful_preds)
            all_hateful_labels.extend(hateful_labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    group_accuracy = accuracy_score(all_group_labels, all_group_preds)
    group_f1 = f1_score(all_group_labels, all_group_preds, average='macro')
    hateful_accuracy = accuracy_score(all_hateful_labels, all_hateful_preds)
    hateful_f1 = f1_score(all_hateful_labels, all_hateful_preds, average='macro')

    return {
        'loss': total_loss / len(dataloader),
        'group_accuracy': group_accuracy,
        'group_f1': group_f1,
        'hateful_accuracy': hateful_accuracy,
        'hateful_f1': hateful_f1
    }


# 训练循环
print("开始训练模型...")
epochs = 5
best_f1 = 0
best_model_state = None

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # 训练
    train_loss = train_model(model, train_dataloader, optimizer, scheduler)

    # 评估
    eval_results = evaluate_model(model, val_dataloader)

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {eval_results['loss']:.4f}")
    print(f"Group Accuracy: {eval_results['group_accuracy']:.4f}, F1: {eval_results['group_f1']:.4f}")
    print(f"Hateful Accuracy: {eval_results['hateful_accuracy']:.4f}, F1: {eval_results['hateful_f1']:.4f}")

    # 保存最佳模型
    avg_f1 = (eval_results['group_f1'] + eval_results['hateful_f1']) / 2
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved with average F1: {best_f1:.4f}")

    print("-" * 50)

# 加载最佳模型
model.load_state_dict(best_model_state)
print("最佳模型已加载")


# 改进的Target-Argument提取函数
def extract_target_argument_pairs(text):
    """
    使用改进的方法提取Target-Argument对
    """
    import jieba.posseg as pseg

    # 分词并标注词性
    words = pseg.cut(text)
    words_pos = [(word, pos) for word, pos in words]

    # 候选目标(名词、代词)和论点(动词、形容词)
    targets = []
    arguments = []

    for word, pos in words_pos:
        if pos.startswith('n') or pos == 'r':  # 名词或代词
            targets.append(word)
        elif pos.startswith('v') or pos.startswith('a'):  # 动词或形容词
            arguments.append(word)

    # 为关键词添加优先级
    high_priority_targets = []
    for word in targets:
        # 检查是否包含仇恨关键词
        for category, keywords in hate_keywords.items():
            if any(keyword in word for keyword in keywords):
                high_priority_targets.append(word)
                break

    # 如果有高优先级目标，优先使用
    if high_priority_targets:
        targets = high_priority_targets

    # 如果没有找到目标或论点
    if not targets:
        targets = ["NULL"]
    if not arguments and text:
        arguments = [text]

    # 限制最多产生2个目标-论点对，优先考虑高优先级目标
    pairs = []
    used_targets = set()
    used_arguments = set()

    # 首先尝试直接从文本中找到目标-论点对
    for target in targets:
        if target == "NULL" or len(pairs) >= 2:
            continue

        if target in text:
            # 找到目标前后最近的论点
            target_idx = text.index(target)

            closest_arg = None
            min_distance = float('inf')

            for arg in arguments:
                if arg in text and arg not in used_arguments:
                    arg_idx = text.index(arg)
                    distance = abs(arg_idx - target_idx)

                    if distance < min_distance:
                        min_distance = distance
                        closest_arg = arg

            if closest_arg and min_distance < len(text) / 2:  # 确保距离不是太远
                pairs.append((target, closest_arg))
                used_targets.add(target)
                used_arguments.add(closest_arg)

    # 如果没有找到任何对，尝试使用最可能的目标和论点
    if not pairs and targets and arguments:
        # 使用第一个目标和论点
        pairs.append((targets[0], arguments[0]))

    # 如果仍然没有任何对，使用NULL和整个文本
    if not pairs:
        pairs.append(("NULL", text))

    # 确保最多返回2个对
    return pairs[:2]


# 判断文本是否包含特定群体相关内容
def identify_targeted_group(text, target, argument):
    # 检查文本、目标和论点中是否包含关键词
    for category, keywords in hate_keywords.items():
        for keyword in keywords:
            if keyword in text or keyword in target or keyword in argument:
                return category

    return "non-hate"  # 默认为non-hate


# 判断是否包含仇恨言论
def is_hateful_content(text, target, argument):
    # 定义可能表示仇恨的词汇
    hate_indicators = [
        '讨厌', '恨', '滚', '垃圾', '废物', '愚蠢', '傻', '蠢', '贱',
        '恶心', '去死', '该死', '灭绝', '下贱', '垃圾', '渣', '混蛋'
    ]

    # 检查是否包含种族歧视词汇(特别处理)
    racist_terms = ['nigger', 'negro', 'niggeг', '黑鬼']
    for term in racist_terms:
        if term.lower() in text.lower():
            return True

    # 检查文本中是否包含仇恨指示词
    for indicator in hate_indicators:
        # 如果目标和仇恨指示词在同一句话中，可能是仇恨言论
        if indicator in text:
            # 检查仇恨指示词是否与目标相关联
            indicator_pos = text.find(indicator)
            target_pos = text.find(target) if target != "NULL" else -1

            # 如果目标和指示词相距较近或指示词是论点的一部分，则更可能是仇恨言论
            if target_pos != -1 and abs(indicator_pos - target_pos) < 10:
                return True
            if indicator in argument:
                return True

    return False


# 预测测试样本
def predict_test_sample(text, model, tokenizer):
    # 提取Target-Argument对
    pairs = extract_target_argument_pairs(text)

    results = []

    for target, argument in pairs:
        # 先使用规则进行初步判断
        targeted_group = identify_targeted_group(text, target, argument)
        is_hateful = is_hateful_content(text, target, argument)

        # 构建输入
        input_text = text
        if target != 'NULL':
            # 高亮文本中的Target和Argument
            for word in [target, argument]:
                if word in input_text:
                    pattern = r'(?<!\[)\b' + re.escape(word) + r'\b(?!\])'
                    input_text = re.sub(pattern, f"<{word}>", input_text)

        encoding = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # 使用模型预测
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            group_pred = torch.argmax(outputs['group_logits'], dim=1).item()
            hateful_pred = torch.argmax(outputs['hateful_logits'], dim=1).item()

            group = target_group_encoder.inverse_transform([group_pred])[0]
            hateful = hateful_encoder.inverse_transform([hateful_pred])[0]

        # 特殊情况处理

        # 1. 检查是否包含明显的种族歧视词汇
        for term in ['nigger', 'negro', 'niggeг', '黑鬼']:
            if term.lower() in text.lower():
                group = "Racism"
                hateful = "hate"

        # 2. 对于地域黑相关文本，如果包含地域名称，考虑为地域歧视
        if '广黑' in text and '地域' in group:
            # 但如果只是询问"广黑吧没了吗"这类中性问题，应该是non-hate
            if '吗' in text or '？' in text or '?' in text:
                hateful = "non-hate"

        # 3. 处理男权/女权议题
        if '男权' in text or '女权' in text:
            group = "Sexism"
            # 但如果文本是在倡导平权，不是仇恨言论
            if '平权' in text:
                hateful = "non-hate"

        results.append({
            'Target': target,
            'Argument': argument,
            'Targeted Group': group,
            'Hateful': hateful
        })

    return results


# 格式化输出结果
def format_output(results):
    output_parts = []
    for result in results:
        target = result['Target']
        argument = result['Argument']
        group = result['Targeted Group']
        hateful = result['Hateful']

        # 格式化为四元组
        quad = f"{target} | {argument} | {group} | {hateful}"
        output_parts.append(quad)

    # 添加分隔符和结束符
    return " [SEP] ".join(output_parts) + " [END]"


# 处理测试集
print("开始处理测试集...")
test_predictions = []

for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="预测测试集"):
    text = row['content']

    # 获取预测结果
    predictions = predict_test_sample(text, model, tokenizer)

    # 格式化输出
    formatted_output = format_output(predictions)

    test_predictions.append(formatted_output)

# 保存预测结果
with open(output_file, 'w', encoding='utf-8') as f:
    for pred in test_predictions:
        f.write(pred + '\n')

print(f"预测结果已保存到 {output_file}")

# 测试集预测示例
print("\n测试集预测示例:")
for i in range(min(5, len(df_test))):
    print(f"ID: {df_test.iloc[i]['id']}")
    print(f"内容: {df_test.iloc[i]['content']}")
    print(f"预测: {test_predictions[i]}")
    print("-------------------------")