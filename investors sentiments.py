from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification

# 建立情绪分类Bert模型
tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm')
model = BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm', return_dict=True)
metric = load_metric('glue', 'sst2')

raw_datasets = load_dataset('csv', data_files={'train': '/Users/huanggm/Desktop/副本2014-12-31.csv',
                                               'test': '/Users/huanggm/Desktop/result/2015-01-01.csv'})


def tokenize_function(examples):
    return tokenizer(examples['title'], truncation=True, padding='max_length', max_length=512)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets
