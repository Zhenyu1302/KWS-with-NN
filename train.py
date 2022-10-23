import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def train(MODELS_list, epoch, log_interval):
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
              print(f"Train Epoch {type(model).__name__}: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
              
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()
  
def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(MODELS_list, epoch):
    for model in MODELS_list:
        model.eval()    
    correct = [0,0] # should be determined by the MODELS_lists length
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
            correct[i] += number_of_correct(pred, target)
            i += 1
            print(f"\nTest Epoch {type(model).__name__}: {epoch}\tAccuracy: {correct[0]}/{len(test_loader.dataset)} ({100. * correct[0] / len(test_loader.dataset):.0f}%)\n")

    error_rate = 1-correct/len(test_loader.dataset)

    return error_rate
