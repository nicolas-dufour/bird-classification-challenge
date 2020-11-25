import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
import timm


nclasses = 20 

class Net(ppl.LightningModule):
    '''
    This class is a wrapper for the rest of our models
    '''
    def __init__(self,lr,momentum):
        super(Net, self).__init__()
        self.model = Linear()
        self.lr = lr
        self.momentum = momentum

    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self,batch, batch_idx):
        data,target = batch
        output = self(data)
        pred = output.data.max(1, keepdim=True)[1]
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        self.log("training_loss_step", loss)
        n_correct_pred = pred.eq(target.data.view_as(pred)).sum().detach()
        return {'loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(target)}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        self.logger.experiment.add_scalars("Losses_epoch", {"train": avg_loss},global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"train": train_acc},global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        data,target = batch
        output = self(data)
        pred = output.data.max(1, keepdim=True)[1]
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        self.log("val_loss", loss)
        n_correct_pred = pred.eq(target.data.view_as(pred)).sum().detach()
        return {'val_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(target)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        self.logger.experiment.add_scalars("Losses_epoch", {"val": avg_loss},global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"val": val_acc},global_step=self.current_epoch)
        
    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.009, max_lr=0.07,step_size_up=275,base_momentum=0.8 , max_momentum=0.99, scale_mode='exp_range', gamma=0.94)
        return [optimizer]#,[scheduler]


class InceptionNet(Net):
    def __init__(self,lr,momentum):
        super(Net, self).__init__(lr,momentum)
        self.inception = models.inception_v3(pretrained=True)
        # for child in list(self.inception.children())[:-2]:
        #     for param in child.parameters():
        #         param.requires_grad = False
        self.inception.aux_logits = False
        num_ftrs = self.inception.fc.in_features
        self.inception.fc= nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        x = self.inception(x)
        return x
        
    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.009, max_lr=0.07,step_size_up=275,base_momentum=0.8 , max_momentum=0.99, scale_mode='exp_range', gamma=0.94)
        return [optimizer]#,[scheduler]



class EfficientNet(Net):
    def __init__(self,lr,momentum):
        super(Net, self).__init__(lr,momentum)
        self.efficientnet = timm.create_model('tf_efficientnet_b4', pretrained=True)
        for child in list(self.efficientnet.children())[:-6]:
            for param in child.parameters():
                param.requires_grad = False
        num_ftrs = self.efficientnet.classifier.in_features
        self.efficientnet.classifier= nn.Linear(num_ftrs, 512)
        self.fc = nn.Linear(512,nclasses)

    def forward(self, x):
        x = F.relu(self.efficientnet(x))
        x = self.fc(x)
        return x
        
    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.0001,step_size_up=100, cycle_momentum = False,scale_mode='exp_range', gamma=0.94)
        return [optimizer],[scheduler]

class VitNet(Net):
    def __init__(self,lr,momentum):
        super(Net, self).__init__(lr,momentum)
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        num_ftrs = list(self.vit.children())[-1].in_features
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(num_ftrs,nclasses)

    def forward(self, x):
        x = F.relu(self.vit(x))
        x = self.fc(x)
        return x
    
    def embedding(self,x):
        return self.vit(x) 
        
    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        return [optimizer],[scheduler]

class VitNetAug(Net):
    def __init__(self,lr,momentum):
        super(Net, self).__init__(lr,momentum)
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        num_ftrs = list(self.vit.children())[-1].in_features
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(num_ftrs,nclasses)

    def forward(self, x):
        x = F.relu(self.vit(x))
        x = self.fc(x)
        return x
    
    def embedding(self,x):
        return self.vit(x)

    def training_step(self,batch, batch_idx):
        data,target = batch

        output_orig = self(data[0])
        output_augmix_1 = self(data[1])
        output_augmix_2 = self(data[2])

        pred = output_orig.data.max(1, keepdim=True)[1]

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss_CE = criterion(output_orig, target)
        loss = loss_CE
        p_clean, p_aug1, p_aug2 = F.softmax(
          output_orig, dim=1), F.softmax(
              output_augmix_1, dim=1), F.softmax(
                  output_augmix_2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        
        self.log("training_loss_step", loss)
        n_correct_pred = pred.eq(target.data.view_as(pred)).sum().detach()
        return {'loss': loss,'loss_CE': loss_CE, "n_correct_pred": n_correct_pred, "n_pred": len(target)}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss_CE'] for x in outputs]).mean()
        train_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        self.logger.experiment.add_scalars("Losses_epoch", {"train": avg_loss},global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"train": train_acc},global_step=self.current_epoch)
        
    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        return [optimizer],[scheduler]