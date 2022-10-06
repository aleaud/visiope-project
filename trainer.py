import torch
import time
from metrics import mIoU


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, device, epochs=1):
  torch.cuda.empty_cache()
  model.to(device)
  start = time.time()

  train_loss, valid_loss = [], []
  train_acc, valid_acc = [], []
  train_iou, valid_iou = [], []
  #best_acc = 0.0
  
  for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    print('-' * 10)

    for phase in ['train','valid']:
      if phase == 'train':
        model.train()
        dataloader = train_dl
      else:
        model.eval()
        dataloader = valid_dl
    
      running_loss = 0.0
      running_acc = 0.0
      running_iou = 0.0

      for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.squeeze(1).to(device)
        iou = 0.0

        #forward pass
        if phase == 'train':
          optimizer.zero_grad()
          predictions = model(x)
          #print("prediction shape: {}\ntarget shape: {}".format(predictions.shape, y.shape))
          loss = loss_fn(predictions, y)
          iou += mIoU(predictions, y)
          #the backward pass frees the graph memory, so there is no need for torch.no_grad in this training pass
          loss.backward()
          optimizer.step()
        else:
          with torch.no_grad():
            predictions = model(x)
            loss = loss_fn(predictions, y)
            iou += mIoU(predictions, y)
      

        #compute stats
        accuracy = acc_fn(predictions, y)
        running_acc += accuracy * dataloader.batch_size
        running_loss += loss.item() * dataloader.batch_size
        running_iou += iou * dataloader.batch_size

        #print summary every 10 steps
        if batch_idx % 10 == 0:
          print('Current step: {}  Loss: {}  Acc: {} IoU: {} AllocMem (Mb): {}'.format(batch_idx, loss, accuracy, iou, torch.cuda.memory_allocated()/1024/1024))
        
      epoch_loss = running_loss / len(dataloader.dataset)
      epoch_acc = running_acc / len(dataloader.dataset)
      epoch_iou = running_iou / len(dataloader.dataset)
      print('{} Loss: {:.4f} Acc: {} IoU: {}'.format(phase, epoch_loss, epoch_acc, epoch_iou))
      
      #store history of losses and accuracy
      train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
      train_acc.append(epoch_acc) if phase == 'train' else valid_acc.append(epoch_acc)
      train_iou.append(epoch_iou) if phase == 'train' else valid_iou.append(epoch_iou)

  time_elapsed = time.time() - start
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  '''
  best_train_acc = np.amax(train_acc)
  best_train_acc_epoch = np.where(train_acc==best_train_acc)[0]       #also np.argmax could be used
  best_valid_acc = np.amax(valid_acc)
  best_valid_acc_epoch = np.where(valid_acc==best_valid_acc)[0]
  print('For early stopping: best accuracy during training was {} at epoch {}, while the best valid accuracy was {} at epoch {}',
                best_train_acc, best_train_acc_epoch, best_valid_acc, best_valid_acc_epoch)
  '''
  
  #group everything and return values
  history = {'train_loss': train_loss, 'valid_loss': valid_loss, 
              'train_acc': train_acc, 'valid_acc': valid_acc,
              'train_iou': train_iou, 'valid_iou': valid_iou}
  return history 

