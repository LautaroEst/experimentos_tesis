import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel


def init_bert2bert(num_layers):
    path = "dccuchile/bert-base-spanish-wwm-cased"
    config = AutoConfig.from_pretrained(path, num_hidden_layers=num_layers)
    encoder = AutoModel.from_pretrained(path,config=config)
    decoder = None




if __name__ == "__main__":
    path = "dccuchile/bert-base-spanish-wwm-cased"
    # path = "DeepESP/gpt2-spanish"
    config = AutoConfig.from_pretrained(path,bos_token="[BOS]",eos_token="[EOS]",num_hidden_layers=4)
    encoder = BertGenerationEncoder.from_pretrained(path,config=config)
    decoder = BertGenerationDecoder.from_pretrained(path,config=config)
    model = EncoderDecoderModel(encoder=encoder,decoder=decoder)
    

    tokenizer = AutoTokenizer.from_pretrained(path,config=config)
    encoded_inputs = tokenizer("Esta es una frase de prueba para que sea resumida por beto",return_tensors='pt')
    decoder_inputs = tokenizer("",return_tensors='pt')
    outputs = model.generate(
        input_ids=encoded_inputs['input_ids'],
        attention_mask=encoded_inputs['attention_mask']
        # decoder_input_ids=decoder_inputs['input_ids'],
        # decoder_attention_mask=decoder_inputs['attention_mask']
    )
    print(outputs)
    # print(encoded_inputs['input_ids'].size())
    # print(outputs.last_hidden_state.size())
