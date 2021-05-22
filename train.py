import os
import pandas as pd
import pickle as pickle

import torch

from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments

from load_data import *


import wandb
wandb.init(project='Stage2-KLUE', name='koelectra two sentence')

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def train():
    #MODEL_NAME = "bert-base-multilingual-cased"
    MODEL_NAME = "monologg/koelectra-base-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    train_label = train_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.to(device)

    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=3,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=10, #4              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    #evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    #eval_steps = 500,            # evaluation step.
    )
    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    #eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()


def main():
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')


    main()
