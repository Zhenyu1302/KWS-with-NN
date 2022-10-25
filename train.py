import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from features import feature

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    # Concatenate label indexes along a new dimension
    targets = torch.stack(targets)
    return tensors, targets

def train(MODELS_list, epoch, log_interval,train_loader,labels_list, device):
    global labels
    labels = labels_list
    for model in MODELS_list:
        model.train()      
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        
        # apply features and model on whole batch directly on device
        data = torch.squeeze(data)
        fbank_features = feature(data)
        for model in MODELS_list:
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
            output = model(fbank_features)
            # Loss and backward propagation
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print training stats
            if batch_idx % log_interval == 0:
                print(f"Train {type(model).__name__} Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
              
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()
  
def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(MODELS_list, epoch,test_loader,labels_list,device):
    for model in MODELS_list:
        model.eval()    
    correct = np.zeros((len(MODELS_list),epoch))
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = torch.squeeze(data)
        fbank_features = feature(data)
        i = 0
        for model in MODELS_list:
            output = model(fbank_features)
            pred = get_likely_index(output)
            correct[i,epoch] += number_of_correct(pred, target)
            i += 1
    
    for i in range(len(MODELS_list)):
        print(f"\nTest {type(model).__name__} Epoch: {epoch}\tAccuracy: {correct[i]}/{len(test_loader.dataset)} ({100. * correct[i] / len(test_loader.dataset):.0f}%)\n")

    error_rate = 1-correct/len(test_loader.dataset)

    return error_rate
