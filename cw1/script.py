import yaml
with open("data/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.Loader)

# Reproducing same results
SEED = 2023
import random
import numpy as np
# Torch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else"cpu" )
def seed_everything(seed = 2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# Read train dataset
class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, filename, target ,transform = None):
        with open(filename, 'r', encoding='iso-8859-1') as f:
            self.sentences = f.readlines()
            self.labels = []
            for sentence in self.sentences:
                label, text = sentence.strip().split(" ", 1)
                coarse = label.split(':')[0]
                if target == 'coarse':
                  self.labels.append(coarse)
                elif target == 'fine':
                  self.labels.append(label)
        self.transform = transform
    
    def __getitem__(self, index):
        # Get the sentence and label at the given index
        sentence = self.sentences[index].strip().split(' ', 1)[1]
        sentence = sentence.strip().lower().split()
              
        return sentence, self.labels[index]
        
    def __len__(self):
        return len(self.sentences)

def text_pipeline(sentence):
    indexed_sentence = []
    for w in sentence:
        if w in train_vocab:
            indexed_sentence.append(train_vocab[w])
        else:
            indexed_sentence.append(train_vocab['<unk>'])
    return indexed_sentence

# Load Glove
## Create a dic of glove {word:vector}
import numpy as np
glove_file = 'data/glove.small.txt'
glove_dict = {}
with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_dict[word] = coefs
glove_embed_size = np.stack(glove_dict.values()).shape[1]

# Bag of Words
import torch.nn as nn
import torch.nn.functional as F
class QuestionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, use_glove, fine_tune_embed):
        super().__init__()
        self.use_glove = use_glove
        self.fine_tune_embed = fine_tune_embed

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True) # Bag-of-Words
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # random initialised embedding
        if self.use_glove == False:
          self.embedding.weight.data.uniform_(-initrange, initrange) 
        # Glove embedding
        else:
          self.embedding.weight = nn.Parameter(torch.tensor(glove_weights_matrix, dtype=torch.float32))
        
        # If freeze the embedding weight
        if self.fine_tune_embed == False: 
          self.embedding.requires_grad_(False)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc(embedded)
        return x

# BiLSTM
from torch.nn.utils.rnn import pad_sequence

class BiLSTM3(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, num_class, use_glove, fine_tune_embed):
        super(BiLSTM3, self).__init__()
        self.use_glove = use_glove
        self.fine_tune_embed = fine_tune_embed

        self.hidden_size = 64
        drp = 0.1
        # n_classes = len(le.classes_)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # random initialised embedding
        if self.use_glove == False:
          self.embedding.weight.data.uniform_(-initrange, initrange) 
        # Glove embedding
        else:
          self.embedding.weight = nn.Parameter(torch.tensor(glove_weights_matrix, dtype=torch.float32))
        
        # If freeze the embedding weight
        if self.fine_tune_embed == False: 
          self.embedding.weight.requires_grad_(False)
        '''self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()'''

    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

# Train, Test, Val for Bag of Words
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

class TrainValTestModel():
  def __init__(self, model, collate_fn, n_epochs, batch_size, min_valid_loss) -> None:
     self.model = model
     self.collate_fn = collate_fn
     self.n_epochs = n_epochs
     self.batch_size = batch_size
     self.min_valid_loss = min_valid_loss

     # CrossEntropyLoss already contains Softmax function inside
     self.criterion = torch.nn.CrossEntropyLoss().to(device)
     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

  def split_into_train_valid(self, train_dataset, train_ratio = 0.90):
    train_len = int(len(train_dataset) * train_ratio)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len],generator=torch.Generator().manual_seed(42))
    return sub_train_, sub_valid_

  def train(self, sub_train):
    # Set the model to train mode
    self.model.train()
    # Train model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train, batch_size=self.batch_size, shuffle=True,
                      collate_fn=self.collate_fn)
    for i, (text, offsets, cls) in enumerate(data):
        self.optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        # print(f'text: {text}')
        # print(f'offsets: {offsets}')
        # print(f'cls: {cls}')
        output = self.model(text, offsets)
        # print(f'output: {output}')
        loss = self.criterion(output, cls)
        # print(f'loss: {loss}')
        train_loss += loss.item()
        loss.backward()
        self.optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    # Update learning rate
    self.scheduler.step()

    return train_loss / len(sub_train), train_acc / len(sub_train)

  def validate(self, sub_valid):
    self.model.eval()
    loss = 0
    acc = 0
    valid_preds = []
    valid_labels = []
    data = DataLoader(sub_valid, batch_size=self.batch_size, collate_fn=self.collate_fn)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = self.model(text, offsets)
            loss += self.criterion(output, cls).item()
            acc += (output.argmax(1) == cls).sum().item()
            valid_preds += output.argmax(1).tolist()
            valid_labels += cls.tolist()

    return loss / len(sub_valid), acc / len(sub_valid), valid_preds, valid_labels

  def f1_score_macro(self, y_true, y_pred):
    num_labels = len(np.unique(y_true))
    f1_scores = []
    for label in range(num_labels):
        tp = np.sum(np.logical_and(y_pred == label, y_true == label))
        fp = np.sum(np.logical_and(y_pred == label, y_true != label))
        fn = np.sum(np.logical_and(y_pred != label, y_true == label))
        eps = 1e-7  # epsilon to avoid division by zero
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    macro_f1_score = np.mean(f1_scores)
    return macro_f1_score
  

  def train_and_evaluate(self, train_dataset, train_ratio=0.90):
    sub_train, sub_valid = self.split_into_train_valid(train_dataset, train_ratio=train_ratio)
    print('[Train & Validation]')

    train_loss_list = []
    valid_loss_list = []
    valid_preds_list = []
    valid_labels_list = []
    for epoch in range(self.n_epochs):
        start_time = time.time()
        train_loss, train_acc = self.train(sub_train)
        valid_loss, valid_acc, valid_preds, valid_labels = self.validate(sub_valid)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        self.train_loss_list = train_loss_list
        self.valid_loss_list = valid_loss_list
        valid_preds_list.append(valid_preds)
        valid_labels_list.append(valid_labels)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    # Calculate macro F1 score
    valid_preds_all = np.concatenate(valid_preds_list)
    valid_labels_all = np.concatenate(valid_labels_list)
    macro_f1_score = self.f1_score_macro(valid_labels_all, valid_preds_all)
    print('Macro F1 score: %.4f' % macro_f1_score)
  
  def loss_plot(self):
    fig = plt.figure(figsize=(6, 6))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(self.n_epochs)+1), self.train_loss_list, label='train')
    plt.plot(list(np.arange(self.n_epochs)+1), self.valid_loss_list, label='valid')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

  def test(self, test_dataset):
    test_loss, test_acc, test_preds, test_labels = self.validate(test_dataset)
    print('[Test]')
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

# Train, Test, Val for BiLSTM
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

class TrainValTestBiLSTMModel():
  def __init__(self, model, collate_fn, n_epochs, batch_size, min_valid_loss) -> None:
     self.model = model
     self.collate_fn = collate_fn
     self.n_epochs = n_epochs
     self.batch_size = batch_size
     self.min_valid_loss = min_valid_loss

     # CrossEntropyLoss already contains Softmax function inside
     self.criterion = torch.nn.CrossEntropyLoss().to(device)
     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

  def split_into_train_valid(self, train_dataset, train_ratio = 0.90):
    train_len = int(len(train_dataset) * train_ratio)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len],generator=torch.Generator().manual_seed(42))
    return sub_train_, sub_valid_

  def train(self, sub_train):
    # Set the model to train mode
    self.model.train()
    # Train model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train, batch_size=self.batch_size, shuffle=True,
                      collate_fn=self.collate_fn)
    for i, (text, offsets, cls) in enumerate(data):
        self.optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        # print(f'text: {text}')
        # print(f'offsets: {offsets}')
        # print(f'cls: {cls}')
        output = self.model(text)
        # print(f'output: {output}')
        loss = self.criterion(output, cls)
        # print(f'loss: {loss}')
        train_loss += loss.item()
        loss.backward()
        self.optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    # Update learning rate
    self.scheduler.step()

    return train_loss / len(sub_train), train_acc / len(sub_train)

  def validate(self, sub_valid):
    self.model.eval()
    loss = 0
    acc = 0
    data = DataLoader(sub_valid, batch_size=self.batch_size, collate_fn=self.collate_fn)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = self.model(text)
            loss = self.criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(sub_valid), acc / len(sub_valid)

  def train_and_evaluate(self, train_dataset, train_ratio = 0.90):
    sub_train, sub_valid =  self.split_into_train_valid(train_dataset, train_ratio = train_ratio)
    print('[Train & Validation]')

    train_loss_list = []
    valid_loss_list = []
    for epoch in range(self.n_epochs):
      start_time = time.time()
      train_loss, train_acc = self.train(sub_train)
      valid_loss, valid_acc = self.validate(sub_valid)

      train_loss_list.append(train_loss)
      valid_loss_list.append(valid_loss)
      self.train_loss_list = train_loss_list
      self.valid_loss_list = valid_loss_list

      secs = int(time.time() - start_time)
      mins = secs / 60
      secs = secs % 60
      
      print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
      print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
      print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
  
  def loss_plot(self):
    fig = plt.figure(figsize=(6, 6))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(self.n_epochs)+1), self.train_loss_list, label='train')
    plt.plot(list(np.arange(self.n_epochs)+1), self.valid_loss_list, label='valid')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

  def test(self, test_dataset):
    test_loss, test_acc = self.validate(test_dataset)
    print('[Test]')
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

if cfg['classification'] == 'coarse':
    # Read Coarse dataset
    coarse_train_dataset = ReadDataset(cfg['path_train'], target = "coarse")
    # Load test Coarse dataet
    coarse_test_dataset = ReadDataset(cfg['path_test'], target = "coarse")

    # Create vocab
    train_vocab = {}
    for sentence, _ in coarse_train_dataset:
        for w in sentence:
            if w not in train_vocab:
                train_vocab[w] = len(train_vocab)

    train_vocab['<unk>'] = len(train_vocab)

    # Define Collation Function for Coarse labels
    def coarse_label_pipeline(label):
        if label == "ABBR":
            return 0
        elif label == "DESC":
            return 1
        elif label == "ENTY":
            return 2
        elif label == "HUM":
            return 3
        elif label == "LOC":
            return 4
        elif label == "NUM":
            return 5

    # Collate without padding for BoW: Coarse dataloader
    from torch.utils.data import DataLoader
    def collate_batch_coarse(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(coarse_label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, offsets, label_list
    
    # Collate with padding for BiLSTM: Coarse dataloader
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    def collate_batch_padding2_coarse(batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in sorted_batch:
            label_list.append(coarse_label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        # Get each sequence and pad it
        sequences = [x for x in text_list]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
  
        return sequences_padded, lengths, label_list
    
    # Create a weight matrix using glove_dict & vocab of train dataset
    matrix_len = len(train_vocab)

    glove_weights_matrix = np.zeros((matrix_len, glove_embed_size))
    words_found = 0

    for i, word in enumerate(train_vocab):
        try: 
            glove_weights_matrix[i] = glove_dict[word]
            words_found += 1
        except KeyError:
            glove_weights_matrix[i] = np.random.normal(scale=0.6, size=(glove_embed_size, ))

    # Coarse Model
    VOCAB_SIZE = len(train_vocab)
    EMBED_DIM = glove_embed_size # 300
    NUM_CLASS_coarse = len(set(coarse_train_dataset.labels))
    print(f'num class: {NUM_CLASS_coarse}')

    # lr = 1e-4
    dropout_prob = 0.5
    # max_document_length = 100  # each sentence has until 100 words
    # max_size = 5000 # maximum vocabulary size
    num_hidden_nodes = 93
    hidden_dim2 = 128
    num_layers = 2  # LSTM layers
    bi_directional = True 

    if cfg['model'] == 'bow' and cfg['embedding'] == 'random' and cfg['setting'] == 'finetune':
        model1 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_coarse, use_glove=False, fine_tune_embed=True).to(device)
        model1_evaluator = TrainValTestModel(model = model1, collate_fn = collate_batch_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model1_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
        model1_evaluator.test(test_dataset=coarse_test_dataset)
        #model1_evaluator.loss_plot()

    elif cfg['model'] == 'bow' and cfg['embedding'] == 'glove' and cfg['setting'] == 'freeze':
        model2 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_coarse, use_glove=True, fine_tune_embed=False).to(device)
        model2_evaluator = TrainValTestModel(model = model2, collate_fn = collate_batch_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model2_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
        model2_evaluator.test(test_dataset=coarse_test_dataset)
        #model2_evaluator.loss_plot()

    elif cfg['model'] == 'bow' and cfg['embedding'] == 'glove' and cfg['setting'] == 'finetune':
        model3 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_coarse, use_glove=True, fine_tune_embed=True).to(device)
        model3_evaluator = TrainValTestModel(model = model3, collate_fn = collate_batch_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model3_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
        model3_evaluator.test(test_dataset=coarse_test_dataset)
        #model3_evaluator.loss_plot()

    elif cfg['model'] == 'bilstm' and cfg['embedding'] == 'random' and cfg['setting'] == 'finetune':
        model4 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_coarse, use_glove=False, fine_tune_embed=True).to(device)
        model4_evaluator = TrainValTestBiLSTMModel(model = model4, collate_fn = collate_batch_padding2_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model4_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
        model4_evaluator.test(test_dataset=coarse_test_dataset)
        #model4_evaluator.loss_plot()
    
    elif cfg['model'] == 'bilstm' and cfg['embedding'] == 'glove' and cfg['setting'] == 'freeze':
        model5 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_coarse, use_glove=True, fine_tune_embed=False).to(device)
        model5_evaluator = TrainValTestBiLSTMModel(model = model5, collate_fn = collate_batch_padding2_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model5_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
        model5_evaluator.test(test_dataset=coarse_test_dataset)
        #model5_evaluator.loss_plot()

    elif cfg['model'] == 'bilstm' and cfg['embedding'] == 'glove' and cfg['setting'] == 'finetune':
        model6 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_coarse, use_glove=True, fine_tune_embed=True).to(device)
        model6_evaluator = TrainValTestBiLSTMModel(model = model6, collate_fn = collate_batch_padding2_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model6_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
        model6_evaluator.test(test_dataset=coarse_test_dataset)
        #model6_evaluator.loss_plot()

elif cfg['classification'] == 'fine':
    # Read fine dataset
    fine_train_dataset = ReadDataset(cfg['path_train'], target = "fine")
    # Load test Fine dataet
    fine_test_dataset = ReadDataset(cfg['path_test'], target = "fine")
    # Create vocab
    train_vocab = {}
    for sentence, _ in fine_train_dataset:
        for w in sentence:
            if w not in train_vocab:
                train_vocab[w] = len(train_vocab)

    train_vocab['<unk>'] = len(train_vocab)


    # Define Collation Function for Fine labels
    def fine_label_pipeline(label):
        if label == "ABBR:abb":
            return 0
        elif label == "ABBR:exp":
            return 1
        elif label == "ENTY:animal":
            return 2
        elif label == "ENTY:body":
            return 3
        elif label == "ENTY:color":
            return 4
        elif label == "ENTY:cremat":
            return 5
        elif label == "ENTY:currency":
            return 6
        elif label == "ENTY:dismed":
            return 7
        elif label == "ENTY:event":
            return 8
        elif label == "ENTY:food":
            return 9
        elif label == "ENTY:instru":
            return 10
        elif label == "ENTY:lang":
            return 11
        elif label == "ENTY:letter":
            return 12
        elif label == "ENTY:other":
            return 13
        elif label == "ENTY:plant":
            return 14
        elif label == "ENTY:product":
            return 15
        elif label == "ENTY:religion":
            return 16
        elif label == "ENTY:sport":
            return 17
        elif label == "ENTY:substance":
            return 18
        elif label == "ENTY:symbol":
            return 19
        elif label == "ENTY:techmeth":
            return 20
        elif label == "ENTY:termeq":
            return 21
        elif label == "ENTY:veh":
            return 22
        elif label == "ENTY:word":
            return 23
        elif label == "DESC:def":
            return 24
        elif label == "DESC:desc":
            return 25
        elif label == "DESC:manner":
            return 26
        elif label == "DESC:reason":
            return 27
        elif label == "HUM:gr":
            return 28
        elif label == "HUM:ind":
            return 29
        elif label == "HUM:title":
            return 30
        elif label == "HUM:desc":
            return 31
        elif label == "LOC:city":
            return 32
        elif label == "LOC:country":
            return 33
        elif label == "LOC:mount":
            return 34
        elif label == "LOC:other":
            return 35
        elif label == "LOC:state":
            return 36
        elif label == "NUM:code":
            return 37
        elif label == "NUM:count":
            return 38
        elif label == "NUM:date":
            return 39
        elif label == "NUM:dist":
            return 40
        elif label == "NUM:money":
            return 41
        elif label == "NUM:ord":
            return 42
        elif label == "NUM:other":
            return 43
        elif label == "NUM:period":
            return 44
        elif label == "NUM:perc":
            return 45
        elif label == "NUM:speed":
            return 46
        elif label == "NUM:temp":
            return 47
        elif label == "NUM:volsize":
            return 48
        elif label == "NUM:weight":
            return 49 

    # Collate without padding for BoW: Fine dataloader
    from torch.utils.data import DataLoader
    def collate_batch_fine(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(fine_label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, offsets, label_list

    # Collate with padding for BiLSTM: Fine dataloader
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    def collate_batch_padding2_fine(batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in sorted_batch:
            label_list.append(fine_label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        # Get each sequence and pad it
        sequences = [x for x in text_list]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
    
        return sequences_padded, lengths, label_list

    # Create a weight matrix using glove_dict & vocab of train dataset

    matrix_len = len(train_vocab)

    glove_weights_matrix = np.zeros((matrix_len, glove_embed_size))
    words_found = 0

    for i, word in enumerate(train_vocab):
        try: 
            glove_weights_matrix[i] = glove_dict[word]
            words_found += 1
        except KeyError:
            glove_weights_matrix[i] = np.random.normal(scale=0.6, size=(glove_embed_size, ))

    # Fine Model
    VOCAB_SIZE = len(train_vocab)
    EMBED_DIM = glove_embed_size # 300
    NUM_CLASS_fine = len(set(fine_train_dataset.labels))

    # lr = 1e-4
    dropout_prob = 0.5
    # max_document_length = 100  # each sentence has until 100 words
    # max_size = 5000 # maximum vocabulary size
    num_hidden_nodes = 93
    hidden_dim2 = 128
    num_layers = 2  # LSTM layers
    bi_directional = True 

    if cfg['model'] == 'bow' and cfg['embedding'] == 'random' and cfg['setting'] == 'finetune':
        model7 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_fine, use_glove=False, fine_tune_embed=True).to(device)
        model7_evaluator = TrainValTestModel(model = model7, collate_fn = collate_batch_fine, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model7_evaluator.train_and_evaluate(train_dataset=fine_train_dataset)
        model7_evaluator.test(test_dataset=fine_test_dataset)
        #model7_evaluator.loss_plot()    

    if cfg['model'] == 'bow' and cfg['embedding'] == 'glove' and cfg['setting'] == 'freeze':
        model8 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_fine, use_glove=True, fine_tune_embed=False).to(device)
        model8_evaluator = TrainValTestModel(model = model8, collate_fn = collate_batch_fine, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model8_evaluator.train_and_evaluate(train_dataset=fine_train_dataset)
        model8_evaluator.test(test_dataset=fine_test_dataset)
        #model8_evaluator.loss_plot()

    if cfg['model'] == 'bow' and cfg['embedding'] == 'glove' and cfg['setting'] == 'finetune':
        model9 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_fine, use_glove=True, fine_tune_embed=True).to(device)
        model9_evaluator = TrainValTestModel(model = model9, collate_fn = collate_batch_fine, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model9_evaluator.train_and_evaluate(train_dataset=fine_train_dataset)
        model9_evaluator.test(test_dataset=fine_test_dataset)
        #model9_evaluator.loss_plot()

    if cfg['model'] == 'bilstm' and cfg['embedding'] == 'random' and cfg['setting'] == 'finetune':
        model10 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_fine, use_glove=False, fine_tune_embed=True).to(device)
        model10_evaluator = TrainValTestBiLSTMModel(model = model10, collate_fn = collate_batch_padding2_fine, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model10_evaluator.train_and_evaluate(train_dataset=fine_train_dataset)
        model10_evaluator.test(test_dataset=fine_test_dataset)
        #model10_evaluator.loss_plot()

    if cfg['model'] == 'bilstm' and cfg['embedding'] == 'glove' and cfg['setting'] == 'freeze':
        model11 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_fine, use_glove=True, fine_tune_embed=False).to(device)
        model11_evaluator = TrainValTestBiLSTMModel(model = model11, collate_fn = collate_batch_padding2_fine, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model11_evaluator.train_and_evaluate(train_dataset=fine_train_dataset)
        model11_evaluator.test(test_dataset=fine_test_dataset)
        #model11_evaluator.loss_plot()

    if cfg['model'] == 'bilstm' and cfg['embedding'] == 'glove' and cfg['setting'] == 'finetune':
        model12 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_fine, use_glove=True, fine_tune_embed=True).to(device)
        model12_evaluator = TrainValTestBiLSTMModel(model = model12, collate_fn = collate_batch_padding2_fine, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
        model12_evaluator.train_and_evaluate(train_dataset=fine_train_dataset)
        model12_evaluator.test(test_dataset=fine_test_dataset)
        #model12_evaluator.loss_plot()