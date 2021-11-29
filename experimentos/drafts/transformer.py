import os
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoConfig, BertConfig, AutoModelForSequenceClassification



def main():
    model_name = models[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer)
    config = AutoConfig.from_pretrained(model_name,num_labels=5,classifier_dropout=0.2)
    print(config.classifier_dropout,config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
    print(model)

if __name__ == "__main__":
    main()