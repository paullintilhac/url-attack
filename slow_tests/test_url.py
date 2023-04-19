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

A = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
"0","1","2","3","4","5","6","7","8","9",
"/","?", ":", "@", "&", "=", "+", "$","-", "_", ".", "!", "~", "*", "'", "(", ")", "#"]
input_dim = len(A)
def tokenized_dict(alphabet):
    return dict((c, i) for i, c in enumerate(alphabet))

def lineToTensor(line,max_len,n_letters=input_dim):
    tensor = torch.zeros(1,max_len,n_letters)
    startIndex = max_len-len(line)
    for li, letter in enumerate(line):
        tensor[0][li+startIndex][letter] = 1
    return tensor

def seq_to_tokens(word, lookup_dict: dict):
    return [lookup_dict[letter] for letter in word] if isinstance(word, (list, tuple)) else lookup_dict[
        word] if word is not None else []

class RNNModel(nn.Module,OpenAttack.Classifier):
    def __init__(self,input_dim,hidden_dim, output_dim,num_layers,maxL):
        super(RNNModel, self).__init__()
        # Number of hidden dimension
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        self.input_dim=input_dim
        self.maxL = maxL
        # RNN
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        # Readout layer
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        char_dict = tokenized_dict(A)
        tokenized_x = seq_to_tokens(list(x), char_dict)
        padded_x =lineToTensor(tokenized_x,max_len=self.maxL)
        
        x_tens = torch.tensor(padded_x)
        h0 = Variable(torch.zeros(2*self.num_layers, x_tens.size(0), self.hidden_dim))
        
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
def main():
    args = sys.argv[1:]
    path = args[0]
    print("args: " + str(args))
    import multiprocessing
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)    
    x_test = torch.load(path + '/x_train_url_sent.pt')
    y_test = torch.load(path + '/y_train_url_sent.pt')
    maxL = 0
    for x in (x_test):
        if len(x)>maxL:
            maxL = len(x)

    print("x_test[0]: " + str(x_test[0]))

    
    dataset = []
    for i in range(6000):
    # for i in range(1):
        this_dict = {'x': x_test[i],'y': int(y_test[i])}
        dataset.append(this_dict)
    
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath,map_location='cpu')
        model = RNNModel(input_dim,100,2,2,maxL)

        print(model)

        model.load_state_dict(checkpoint)

        return model
    
    model = load_checkpoint(path + "/rnn model url vs adversarial w gpu.pth")

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
