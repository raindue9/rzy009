import os
import json
import time
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_scheduler
)
from tqdm.auto import tqdm

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# 配置参数
class Config:
    def __init__(self):
        self.model_name = "uer/t5-base-chinese-cluecorpussmall"  # 中文T5模型
        self.data_dir = "./data"  # 数据集路径
        self.output_dir = "./results"  # 结果保存路径
        self.max_length = 128  # 最大序列长度
        self.batch_size = 8  # 批次大小
        self.learning_rate = 5e-5  # 学习率
        self.num_epochs = 5  # 训练轮数
        self.train_mode = True  # 是否训练
        self.predict_mode = True  # 是否预测


# 数据集类
class HateSpeechDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length, is_training=True, limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.input_examples = []
        self.output_examples = []

        # 读取数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if limit:
            data = data[:limit]  # 限制数据大小（测试用）

        for item in data:
            self.input_examples.append(item["content"])
            if is_training and "output" in item:
                self.output_examples.append(item["output"])

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.input_examples[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
        }

        if self.is_training and self.output_examples:
            outputs = self.tokenizer(
                self.output_examples[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item["labels"] = outputs.input_ids.squeeze()

        return item


# 训练函数
def train(model, train_dataloader, optimizer, lr_scheduler, device, config):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(train_dataloader)


# 预测函数
def predict(model, tokenizer, test_dataloader, device, config):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_length,
                num_beams=4,
                early_stopping=True
            )

            decoded_preds = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            predictions.extend(decoded_preds)
    return predictions


# 保存预测结果
def save_predictions(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    logger.info(f"Predictions saved to {output_file}")


# 主函数
def main():
    # Create configuration
    config = Config()

    # Create output directory if it doesn't exist
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    model.to(device)

    # Check if files exist
    train_file = os.path.join(config.data_dir, "train.json")
    test_file = os.path.join(config.data_dir, "test1.json")

    # Log the paths of the data files
    logger.info(f"Training data path: {train_file}")
    logger.info(f"Test data path: {test_file}")

    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        return

    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
    # 训练
    if config.train_mode:
        logger.info("Starting training...")
        train_dataset = HateSpeechDataset(train_file, tokenizer, config.max_length, is_training=True, limit=1000)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        num_training_steps = config.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                     num_training_steps=num_training_steps)

        for epoch in range(config.num_epochs):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            train_loss = train(model, train_dataloader, optimizer, lr_scheduler, device, config)
            logger.info(f"Training loss: {train_loss:.4f}")
            epoch_end = time.time()
            logger.info(f"Epoch {epoch + 1} Time: {epoch_end - epoch_start:.2f} seconds")

        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        logger.info(f"Final model saved to {config.output_dir}")

    # 预测
    if config.predict_mode:
        logger.info("Starting prediction...")
        test_dataset = HateSpeechDataset(test_file, tokenizer, config.max_length, is_training=False, limit=500)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

        predictions = predict(model, tokenizer, test_dataloader, device, config)
        output_file = os.path.join(config.output_dir, "predictions.txt")
        save_predictions(predictions, output_file)

    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.2f} seconds ({(end_time - start_time) / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
