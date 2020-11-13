#!/usr/bin/env python3
import time
import torch
from models.mltn import MLTN 
from models.mps import MPS
from torchvision import transforms, datasets
import pdb
from data.lidc_dataset import LIDC, LIDCSeg
from utils.tools import *
from models.Densenet import *
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from carbontracker.tracker import CarbonTracker
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sys
from sklearn.metrics import balanced_accuracy_score as bal_acc

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPS = 1e-6
# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bin_acc(target,preds):
    preds = (preds > 0.5).astype(float)
    return np.sum(preds == target).astype(float)/len(target)

def evaluate(loader):
    ### Evaluation funcntion for validation/testing

    vl_acc = 0.
    vl_loss = 0.
    labelsNp = [] 
    predsNp = [] 
    model.eval()

    for i, (inputs, labels) in enumerate(loader):
        b = inputs.shape[0]

        labelsNp = labelsNp + labels.numpy().tolist()
        if args.tenetx or args.mlp:
            inputs = inputs.view(b,1,-1)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Inference
        scores = (model(inputs))
        scores[scores.isnan()] = 0
        scores = torch.sigmoid(scores)
        preds = scores.clone()

        loss = loss_fun(scores.view(labels.shape),labels)

        predsNp = predsNp + preds.cpu().numpy().tolist()
        vl_loss += loss.item()
    # Compute AUC over the full (valid/test) set

    vl_loss = vl_loss/len(loader)

    labelsNp, predsNp = np.array(labelsNp), np.array(predsNp)
    vl_acc = roc_auc_score(labelsNp.reshape(-1), predsNp.reshape(-1))
    return vl_acc, vl_loss

# Miscellaneous initialization
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--fold', type=int, default=0, help='Fold to use for testing')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--l2', type=float, default=0, help='L2 regularisation')
parser.add_argument('--p', type=float, default=0.5, help='Augmentation probability')
parser.add_argument('--aug', action='store_true', default=False, help='Use data augmentation')
parser.add_argument('--bn', action='store_true', default=False, help='Do batch norm after each MPS layer')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping. No clipping if 0')
parser.add_argument('--lidc', action='store_true', default=False, help='Use the 2d LIDC data.')
parser.add_argument('--data_path', type=str, default='lidc/',help='Path to data.')
parser.add_argument('--bond_dim', type=int, default=5, help='MPS Bond dimension')
parser.add_argument('--kernel', type=int, default=2, help='Stride of squeeze kernel')
parser.add_argument('--nChannel', type=int, default=1, help='Number of input channels')
parser.add_argument('--tenetx', action='store_true', 
            default=False, help='Use Tenet-x baseline')
parser.add_argument('--densenet', action='store_true', 
            default=False, help='Use densenet baseline')
parser.add_argument('--mlp', action='store_true', 
            default=False, help='Use MLP baseline')
parser.add_argument('--mltn', action='store_true', 
            default=False, help='Use MLTN')

args = parser.parse_args()


batch_size = args.batch_size

# LoTeNet parameters
adaptive_mode = False 
periodic_bc   = False

kernel = args.kernel # Stride along spatial dimensions
output_dim = 1
feature_dim = 2

logFile = time.strftime("%Y%m%d_%H_%M")+'.txt'
makeLogFile(logFile)

normTensor = 0.5*torch.ones(args.nChannel)
### Data processing and loading....
trans_valid = transforms.Compose([transforms.Normalize(mean=normTensor,std=normTensor,inplace=True)])
if args.aug:
    trans_train = transforms.Compose([transforms.ToPILImage(),
              transforms.RandomHorizontalFlip(p=args.p),
              transforms.RandomVerticalFlip(p=args.p),
              transforms.RandomRotation(20),
              transforms.ToTensor(),
              transforms.Normalize(mean=normTensor,std=normTensor,inplace=True),
              transforms.RandomErasing(p=args.p,inplace=True)])
    print("Using Augmentation with p=%.2f"%args.p)
else:
#       trans_valid = None
    trans_train = trans_valid
    print("No augmentation....")

# Load processed LIDC data 
print("Using LIDC dataset")
print("Using Fold: %d"%args.fold)

dataset_train = LIDC(split='Train', data_dir=args.data_path,
            transform=trans_train,rater=4,fold=args.fold)
dataset_valid = LIDC(split='Valid', data_dir=args.data_path, 
                transform=trans_valid,rater=4,fold=args.fold)
dataset_test = LIDC(split='Test', data_dir=args.data_path, 
                transform=trans_valid,rater=4,fold=args.fold)

# Initiliaze input dimensions
dim = torch.ShortTensor(list(dataset_train[0][0].shape[1:]))
nCh = int(dataset_train[0][0].shape[0])
eval_bSize = batch_size

# Initiliaze input dimensions
num_train = len(dataset_train)
num_valid = len(dataset_valid)
num_test = len(dataset_test)
print("Num. train = %d, Num. val = %d, Num. test = %d"%(num_train,num_valid,num_test))
if not args.densenet:
    evalSize = num_valid // 2
else:
    evalSize = batch_size
loader_train = DataLoader(dataset = dataset_train, drop_last=False,num_workers=1, 
              batch_size=batch_size, shuffle=True,pin_memory=True)
loader_valid = DataLoader(dataset = dataset_valid, drop_last=True,num_workers=1,
              batch_size=evalSize, shuffle=False,pin_memory=True)
loader_test = DataLoader(dataset = dataset_test, drop_last=True,num_workers=1,
             batch_size=(evalSize), shuffle=False,pin_memory=True)

# Initialize the models
if args.mltn:
    print("Using MLTN")
    model = MLTN(input_dim=dim, output_dim=output_dim, 
      nCh=nCh, kernel=kernel, bn = args.bn,dropout=args.dropout, 
      bond_dim=args.bond_dim, feature_dim=feature_dim, 
      adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, virtual_dim=1)
elif args.tenetx:
    print("Tensornet-X baseline")
#       pdb.set_trace()
    model = MPS(input_dim=torch.prod(dim), output_dim=output_dim,
    bond_dim=args.bond_dim,feature_dim=feature_dim,
    adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,tenetx=args.tenetx)
elif args.densenet:
    print("Densenet Baseline!")
    model = DenseNet(depth=40, growthRate=12, 
            reduction=0.5,bottleneck=True,nClasses=output_dim)
elif args.mlp:
    print("MLP Baseline!")
    model = BaselineMLP(inCh=torch.prod(dim),nhid=1,nClasses=output_dim)
else:
    print("Choose a model!")
    sys.exit()
    # Choose loss function and optimizer

loss_fun = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                 weight_decay=args.l2)

nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%d"%(nParam))
print(f"Maximum MPS bond dimension = {args.bond_dim}")

print(f"Using Adam w/ learning rate = {args.lr:.1e}")
print("Local feature map dim: %d, nCh: %d, B:%d"%(feature_dim,nCh,batch_size))
with open(logFile,"a") as f:
    print("Bond dim: %d"%(args.bond_dim),file=f)
    print("Number of parameters:%d"%(nParam),file=f)
    print(f"Using Adam w/ learning rate = {args.lr:.1e}",file=f)
    print("Local feature map dim: %d, nCh: %d, B:%d"%(feature_dim,nCh,batch_size),file=f)

model = model.to(device)
nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

maxAuc = 0
minLoss = 1e3
convCheck = 20
convIter = 0
tracker = CarbonTracker(epochs=args.num_epochs,
        log_dir='carbontracker/',monitor_epochs=-1)

# Let's start training!
for epoch in range(args.num_epochs):
    tracker.epoch_start()
    running_loss = 0.
    running_acc = 0.
    t = time.time()
    model.train()
    predsNp = [] 
    labelsNp = []
    bNum = 0
    for i, (inputs, labels) in enumerate(loader_train):
#           optimizer.zero_grad()
        for p in model.parameters():
            p.grad = None
        bNum += 1
        b = inputs.shape[0]
        if args.tenetx or args.mlp:
            inputs = inputs.view(b,1,-1)
        labelsNp = labelsNp + (labels.numpy()).tolist()

        inputs = inputs.to(device)
        labels = labels.to(device)
        scores = (model(inputs))
        scores = torch.sigmoid(scores)

        loss = loss_fun(scores.view(labels.shape),labels)

        # Backpropagate and update parameters
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        with torch.no_grad():
            preds = scores.clone()
            predsNp = predsNp + (preds.data.cpu().numpy()).tolist()
            running_loss += loss
            
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
               .format(epoch+1, args.num_epochs, i+1, nTrain, loss.item()))
    
    labelsNp, predsNp = np.array(labelsNp), np.array(predsNp)
    accuracy = roc_auc_score(labelsNp.reshape(-1),predsNp.reshape(-1))

    # Evaluate on Validation set 
    with torch.no_grad():
    
        vl_acc, vl_loss = evaluate(loader_valid)
        if vl_acc > maxAuc or vl_loss < minLoss:
            convIter = 0
            ### Predict on test set
            if (vl_acc > maxAuc) or (vl_acc >= maxAuc and vl_loss < minLoss):
                maxAuc = vl_acc
                print('New Best: %.4f'%np.abs(maxAuc))
                ts_acc, ts_loss = evaluate(loader=loader_test)
                print('Test Set Loss:%.4f       Acc:%.4f'%(ts_loss, ts_acc))
                with open(logFile,"a") as f:
                    print('Test Set Loss:%.4f       Acc:%.4f'%(ts_loss, ts_acc),file=f)
                convEpoch = epoch
                if vl_loss < minLoss:
                    minLoss = vl_loss
        else:
            convIter += 1
    if convIter == convCheck:
        print("Converged at epoch:%d with AUC:%.4f"%(convEpoch+1,maxAuc))
        break
    writeLog(logFile, epoch, running_loss/bNum, accuracy,
        vl_loss, np.abs(vl_acc), time.time()-t)
    tracker.epoch_end()
tracker.stop()
