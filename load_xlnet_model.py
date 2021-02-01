# import torch

from keras.preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import xlnet tokenizer and classifier
from transformers import XLNetTokenizer, XLNetForSequenceClassification

label_dict = {'anger': 2,
              'disgust': 4,
              'fear': 1,
              'guilt': 6,
              'happiness': 7,
              'joy': 0,
              'sadness': 3,
              'shame': 5}

# load the model from disk
import torch

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

# creating object for xlnet tokinzer and tokinizing sentecne list

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#xlnet_model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

xlnet_model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=8, dropout=0.25)
xlnet_model.load_state_dict(torch.load('xlnet_model_89.pt',map_location=map_location))

import pandas as pd

# Load the dataset into a pandas dataframe.
text=input('Enter the text: ')
new_sentence = pd.Series(text)

# manually adding spl tokens to sentences
sentences1 = []
for sentence in new_sentence.values:
    sentence = sentence + "[SEP] [CLS]"
    sentences1.append(sentence)
y = []
y.append(0)
tokenized_text1 = [tokenizer.tokenize(sent) for sent in sentences1]
# manually creating the tokens to ids for elnet vocabulary
ids1 = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text1]
# adding padding index to sentences to make equal lenght input
input_padded1 = pad_sequences(ids1, maxlen=92, dtype="long", truncating="post", padding="post")
Xtest = torch.tensor(input_padded1)
y = torch.tensor(y)
test_data = TensorDataset(Xtest, y)
test_loader = DataLoader(test_data, batch_size=64)

xlnet_model.eval()  # Testing our Model
predictions = []

for inp, y in test_loader:
    inp.to(map_location)
    # lab1.to(device)
    # t+=lab1.size(0)
    inp = inp.type(torch.LongTensor)
    outp1 = xlnet_model(inp.to(map_location))

print('emotions.')
import torch
import torch.nn.functional as F

# caluclate the softmax for logits from bert model
# becasue we have 7 different calsses otherwise we use sigmoid for binary classification
softmax_val = F.softmax(outp1[0],dim=-1).tolist()
softmax_val = softmax_val[0]
# sorting the list of softmax values to get max 2 probabilities
sorted_integers = sorted(softmax_val, reverse=True)

largest_prob = sorted_integers[0]

# Here we are extracting the max probability value index from the list and get the key(label) from label dictionary

# max_value = max((softmax_val))
max_index1 = softmax_val.index(largest_prob)

key_list = list(label_dict.keys())
val_list = list(label_dict.values())

emotion1 = (key_list[val_list.index(max_index1)])

print(emotion1)

