import sys, os, datasets, time, ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
))

import OpenAttack
from attackers import get_attackers

def tokenized_dict(alphabet):
    return dict((c, i) for i, c in enumerate(alphabet))

def lineToTensor(line,max_len,n_letters=2):
    tensor = torch.zeros(1,max_len,n_letters)
    startIndex = max_len-len(line)
    for li, letter in enumerate(line):
        tensor[0][li+startIndex][letter] = 1
    return tensor

def seq_to_tokens(word, lookup_dict: dict):
    return [lookup_dict[letter] for letter in word] if isinstance(word, (list, tuple)) else lookup_dict[
        word] if word is not None else []

class RNNModel(nn.Module,OpenAttack.Classifier):
    def __init__(self,input_dim,hidden_dim, output_dim,num_layers):
        super(RNNModel, self).__init__()
        # Number of hidden dimension
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        self.input_dim=input_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True,bidirectional=False, nonlinearity='relu')
        #self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        char_dict = tokenized_dict(["0","1"])
        tokenized_x = seq_to_tokens(list(x), char_dict)
        maxL = 30
        padded_x =lineToTensor(tokenized_x,max_len=maxL)
        
        x_tens = torch.tensor(padded_x)
        h0 = Variable(torch.zeros(self.num_layers, x_tens.size(0), self.hidden_dim))
        
        # One time step
        #print("padded_x size: " + str(padded_x.size()))
        out, hn = self.rnn(x_tens, h0)
        temp = self.fc(out[:, -1, :])
        out = self.softmax(temp)
        return out

    def initHidden(self):
        return torch.zeros(1,self.hidden_dim)

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = []
        for sent in input_: 
            thing =  self.forward(sent)
            ret.append(thing)        
        # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
        return ret

    def get_pred(self, input_):
        thing = self.get_prob(input_)
        return torch.max(thing[0],1)[1]

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)    

    x_test = torch.load('../Adversarial-Attack-on-recurrent-neural-network/x_test_tomita6_sent.pt')
    y_test = torch.load('../Adversarial-Attack-on-recurrent-neural-network/y_test_tomita6_sent.pt')
    x_train = torch.load('../Adversarial-Attack-on-recurrent-neural-network/x_train_tomita6_sent.pt')
    y_train = torch.load('../Adversarial-Attack-on-recurrent-neural-network/y_train_tomita6_sent.pt')
    
    dataset = []
   # for i in range(len(x_test)):
    for i in range(1):
        this_dict = {'x': x_test[i],'y': int(y_test[i])}
        dataset.append(this_dict)
    
    # dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)

    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = RNNModel(2,100,2,1)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    model = load_checkpoint('./rnn model tomita 6.pth')

    attackers = get_attackers(dataset, model)
    print("attackers: " + str(attackers))
    for attacker in attackers:
        print(attacker.__class__.__name__)
        try:
            print(
                OpenAttack.AttackEval(attacker, model).eval(dataset, progress_bar=True),
            )
        except Exception as e:
            raise e
            print(e)
            print("\n")

if __name__ == "__main__":
    main()
