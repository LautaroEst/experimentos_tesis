import os
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoConfig, BertConfig, AutoModelForSequenceClassification,EncoderDecoderConfig, EncoderDecoderModel

def main():
    model_src = "dccuchile/bert-base-spanish-wwm-uncased"
    # Initializing a BERT bert-base-uncased style configuration
    config_encoder = BertConfig()
    config_decoder = BertConfig()

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    # Initializing a Bert2Bert model from the bert-base-uncased style configurations
    model = EncoderDecoderModel(config=config)

    # Accessing the model configuration
    config_encoder = model.config.encoder
    config_decoder  = model.config.decoder
    # set decoder config to causal lm
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True

    # Saving the model, including its configuration
    model.save_pretrained('my-model')

    # loading model and config from pretrained folder
    encoder_decoder_config = EncoderDecoderConfig.from_pretrained(model_src)
    model = EncoderDecoderModel.from_pretrained(model_src, config=encoder_decoder_config)
    print(model)

# def main():
    # model_name = models[0]
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(tokenizer)
    # config = AutoConfig.from_pretrained(model_name,num_labels=5,classifier_dropout=0.2)
    # print(config.classifier_dropout,config)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
    # print(model)

    

if __name__ == "__main__":
    main()