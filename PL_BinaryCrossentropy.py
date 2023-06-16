import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.optim as optim
import pytorch_lightning as pl
import timm
from torchvision import datasets, models, transforms
import os
from torch.optim import lr_scheduler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb
from torchmetrics.classification import BinaryAccuracy
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import argparse
import yaml


class LitAutoEncoder(pl.LightningModule):
    def __init__(self,model_name, threshold, lr =2e-4):
        super().__init__()
        self.lr = lr
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy(threshold=threshold)
    def forward(self, x):
        outputs = self.backbone(x)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.unsqueeze(1).float())
        acc = self.accuracy(outputs, labels.unsqueeze(1).float())
        
        self.log('train_loss', loss ,on_step = False, on_epoch = True, logger = True);
        self.log('train_acc', acc ,on_step = False, on_epoch = True, logger = True);
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.unsqueeze(1).float())
        acc = self.accuracy(outputs, labels.unsqueeze(1).float())
        self.log('val_loss', loss ,on_step = False, on_epoch = True, logger = True);
        self.log('val_acc', acc ,on_step = False, on_epoch = True, logger = True);
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.unsqueeze(1).float())
        acc = self.accuracy(outputs, labels.unsqueeze(1).float())
        self.log('test_loss', loss ,prog_bar = True);
        self.log('test_acc', acc ,prog_bar = True);
        
        return loss
    
def run(args):
    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    wandb.login(key='6e1fe556323d6ff4de8ebd2b657707a17715f7e8')
    wandb_logger = WandbLogger(project='PETDIS_Binary',# group runs in "MNIST" project
                           name=f"{args.model_name}_{os.path.basename(args.root)}_{args.batch_size}_{args.learning_rate}",
                           log_model='all') 
    # Data transformations
    transform = {
        'train':transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48772067, 0.45476404, 0.41735706), (0.22629902, 0.22098567, 0.22134961))]),
        'val':transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.49372694, 0.45860133, 0.41793576), (0.22488698, 0.22060247, 0.22134696))]),
        'test':transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48728254, 0.45438185, 0.41734886), (0.22419675, 0.22046398, 0.22168836))
        ])
        
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.root, x),
                                              transform[x])
                      for x in ['train', 'val','test']}

    # Create data loaders
    train_loader = DataLoader(image_datasets['train'], batch_size=args.batch_size, num_workers=12 , shuffle=True)
    val_loader = DataLoader(image_datasets['val'], batch_size=args.batch_size, num_workers=12)
    test_loader = DataLoader(image_datasets['test'], batch_size=args.batch_size, num_workers=12)
    # Create the model
    model = LitAutoEncoder(args.model_name,args.Threshold)
    # Create a PyTorch Lightning trainer
    trainer = Trainer(logger=wandb_logger, devices=args.devices, max_epochs= args.epochs,
                     callbacks=[EarlyStopping(monitor="val_loss", mode="min"),ModelCheckpoint(dirpath = args.save_model_path, filename = f"{args.model_name}_best")] )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders = test_loader)
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = '/home/ubuntu/workspace/kwanwoo/PETDIS/dataset/cat&dog', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 128, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'resnet34.a1_in1k', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 3, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 3e-5, help = "Learning rate value")
    parser.add_argument("-T", "--Threshold", type = float, default = 0.5, help = "Binary T classification Threshold")
    parser.add_argument("-e", "--epochs", type = int, default = 20, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sr", "--save_results_path", type = str, default = 'results', help = "Path to the directory to save the train results")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)