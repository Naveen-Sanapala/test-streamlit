import random
import json

import numpy as np



import nltk
nltk('punkt')
import streamlit as st

st.title("Chat bot")

import torch
import torch.nn as nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):

    return nltk.word_tokenize(sentence)


def stem(word):

    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):

    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "bot"

st.title("Room Booking chat bot")
#print("Let's chat! (type 'quit' to exit)")
st.header(""""Let us chat! type quit to exit""")
while True:
    # sentence = "do you use credit cards?"
    #sentence = input("You: ")
    sentence = st.text_input("enter your text here")
    st.write("You:",sentence)
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                answer=random.choice(intent['responses'])
                st.write("Bot: ",answer)


    else:
        #print(f"{bot_name}: I do not understand...")
        st.write("Bot: ", "I do not understand")
