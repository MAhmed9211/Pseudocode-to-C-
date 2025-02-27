import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os

# Load Dataset
df = pd.read_csv('spoc-train.tsv', sep='\t', header=None, dtype=str, keep_default_na=False)
df = df.iloc[:, :2].dropna()
df.columns = ['text', 'code']

# Tokenizer
class Tokenizer:
    def __init__(self, texts, min_freq=1):
        self.word_counts = Counter()
        for text in texts:
            self.word_counts.update(text.split())
        self.vocab = {word: idx+4 for idx, (word, count) in enumerate(self.word_counts.items()) if count >= min_freq}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.vocab['<SOS>'] = 2
        self.vocab['<EOS>'] = 3
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(word, 1) for word in text.split()] + [self.vocab['<EOS>']]

    def decode(self, tokens):
        words = [self.inv_vocab.get(token, '<UNK>') for token in tokens if token > 3]
        formatted_code = []
        for word in words:
            formatted_code.append(word)
            if word in [';', '{', '}']:
                formatted_code.append('\n')
        return ' '.join(formatted_code)

pseudo_tokenizer = Tokenizer(df['text'])
cpp_tokenizer = Tokenizer(df['code'])

# Model Definition
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)
    
    def forward(self, src, tgt):
        encoder_embedded = self.encoder_embedding(src)
        _, (hidden, cell) = self.encoder_lstm(encoder_embedded)
        decoder_embedded = self.decoder_embedding(tgt)
        decoder_output, _ = self.decoder_lstm(decoder_embedded, (hidden, cell))
        return self.fc_out(decoder_output)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_vocab_size = len(pseudo_tokenizer.vocab)
output_vocab_size = len(cpp_tokenizer.vocab)
model = Seq2SeqLSTM(input_vocab_size, output_vocab_size).to(device)

MODEL_PATH = "best_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

# Inference Function
def generate_code(pseudo_text):
    model.eval()
    input_tensor = torch.tensor(pseudo_tokenizer.encode(pseudo_text), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        hidden, cell = model.encoder_lstm(model.encoder_embedding(input_tensor))[1]
        tgt_tensor = torch.tensor([[cpp_tokenizer.vocab['<SOS>']]], dtype=torch.long).to(device)
        output_tokens = []
        
        for _ in range(50):
            decoder_embedded = model.decoder_embedding(tgt_tensor)
            output, (hidden, cell) = model.decoder_lstm(decoder_embedded, (hidden, cell))
            next_token = output.argmax(dim=-1)[:, -1].item()
            
            if next_token == cpp_tokenizer.vocab['<EOS>'] or next_token == 0:
                break
            
            output_tokens.append(next_token)
            tgt_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)

    return cpp_tokenizer.decode(output_tokens)

# Streamlit UI
st.title("Pseudocode to C++ Code Generator")

pseudo_input = st.text_area("Enter your pseudocode:")

if st.button("Generate C++ Code"):
    if pseudo_input.strip():
        generated_code = generate_code(pseudo_input)
        st.subheader("Generated C++ Code:")
        st.code(generated_code, language='cpp')
    else:
        st.warning("Please enter valid pseudocode.")
