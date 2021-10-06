from flask import Flask, render_template,request
from summarizer import Summarizer,TransformerSummarizer
from newspaper import fulltext
import requests
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/handle_url', methods=['POST'])
def handle_url():
    url = request.form['text']
    select = request.form['model']

    body = fulltext(requests.get(url).text)

    if select == 'xlnet':
        select = 'XLNet'
        model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
        full = ''.join(model(body, min_length=60))
    elif select == 'gpt':
        select = 'GPT-2'
        GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
        full = ''.join(GPT2_model(body, min_length=60))
    elif select == 'bert':
        select = 'BERT'
        bert_model = Summarizer()
        full = ''.join(bert_model(body, min_length=60))

    else :
        select = 'T5'
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')

        preprocess_text = body.strip().replace("\n","")
        t5_prepared_Text = preprocess_text

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=100,
                                    max_length=500,
                                    early_stopping=True)

        full = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template('text_display.html',bert = full,select = select)


@app.route('/handle_data', methods=['POST'])
def handle_data():
    body = request.form['text']
    select = request.form['model']

    if select == 'xlnet':
        select = 'XLNet'
        model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
        full = ''.join(model(body, min_length=60))
    elif select == 'gpt':
        select = 'GPT-2'
        GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
        full = ''.join(GPT2_model(body, min_length=60))
    elif select == 'bert':
        select = 'BERT'
        bert_model = Summarizer()
        full = ''.join(bert_model(body, min_length=60))

    else :
        select = 'T5'
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')

        preprocess_text = body.strip().replace("\n","")
        t5_prepared_Text = preprocess_text

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=100,
                                    max_length=300,
                                    early_stopping=True)

        full = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return render_template('text_display.html',bert = full,select = select)

if __name__ == '__main__':
    app.run()