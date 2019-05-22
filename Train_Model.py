import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import time
import datetime
import json
import logging
import importlib
import shutil

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

# TODO: import net 
from yolo_model import yoloModel
from PASCAL_Dataset import create_split_loaders
from YOLOLoss import YoloLoss

def train(config):
    config['global_step'] = config.get('start_step', 0)
    is_training = False if config.get('export_onnx') else True

    # TODO: Load and initialize network
    net = yoloModel(config)

    # Define the optimizer and learning rate
    optimizer = obtain_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
        step_size=config['lr']['decay_step'],
        gamma=config['lr']['decay_gamma'])

    # Use pretrained model
    if config['pretrain_snapshot']:
        logging.info('Load pretrained weights from {}'.format(config['pretrain_snapshot']))
        state_dict = torch.load(config['pretrain_snapshot'])
        net.load_state_dict(state_dict)

    # Use all 3 scales for computing YOLO loss
    YOLO_losses = []
    for i in range(3):
        YOLO_losses.append(YoloLoss(config['yolo']['classes'], (config['img_w'], config['img_h']),
                            config['yolo']['anchors'][i])

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 3, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    
    # Load in data
    root_dir = os.getcwd()
    imgs_dir = './VOC2012/JPEGImages/'
    labels_dir = './VOC2012/Labels/'
    train_dataloader, val_loader, test_loader = create_split_loaders(imgs_dir, labels_dir, config['batch_size'])
    
    # Instantiate model to run on the GPU or CPU based on CUDA support
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)
    
    # Begin training loop
    print("Start training:")
    for epoch in range(config['epochs']):
        for minibatch, (images, labels) in enumerate(train_dataloader):
            start_time = time.time()
            config['global_step'] += 1

            # Forward and backward
            optimizer.zero_grad()
            outputs = net(images)
            loss_names = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for z in range(len(loss_names)):
                losses.append([])
            for i in range(3):
                loss_item = YOLO_losses[i](outputs[i], labels)
                for j, l in enumerate(loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            loss.backward()
            optimizer.step()

            if minibatch > 0 and minibatch % 10 == 0:
                _loss = loss.item()
                lr = optimizer.param_groups[0]['lr']
                print('Epoch [%.3d] Minibatch = %d Loss = %.2f lr = %.5f '%
                    (epoch, minibatch, _loss, lr)
                )
                config['tensorboard_writer'].add_scalar("lr",
                                                        lr,
                                                        config['global_step'])
                for i, name in enumerate(loss_names):
                    value = _loss if i == 0 else losses[i]
                    config['tensorboard_writer'].add_scalar(name,
                                                            value,
                                                            config['global_step'])

            if minbatch > 0 and minibatch % 1000 == 0:
                save_checkpoint(net.state_dict(), config)

        lr_scheduler.step()

    _save_checkpoint(net.state_dict(), config)
    print('Training Complete')

    
def save_checkpoint(state_dict, config, evaluate_func=None):
        
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    print("Model checkpoint saved to %s" % checkpoint_path)


def obtain_optimizer(config, net):
    optimizer = None

    # Assign different learning rate for each layer
    parameters = None
    base_parameters = list(
        map(id, net.backbone.parameters())
    )
    logits_parameters = filter(lambda p: id(p) not in base_parameters, net.parameters())

    if not config['lr']['freeze_backbone']:
        parameters = [
            {"parameters": logits_parameters, "lr": config['lr']['other_lr']},
            {"parameters": net.backbone.parameters(), "lr": config['lr']['backbone_lr']},
        ]
    else:
        print("Freezing backbone parameters")
        for p in net.backbone.parameters():
            p.requires_grad = False
        parameters = [
            {"params": logits_parameters, "lr": config['lr']['other_lr']},
        ]

    # Initialize optimizer class
    if config['optimizer']['type'] == "adam":
        optimizer = optim.Adam(parameters, weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == "amsgrad":
        optimizer = optim.Adam(parameters, weight_decay=config['optimizer']['weight_decay'], amsgrad=True)
    elif config['optimizer']['type'] == "rmsprop":
        optimizer = optim.RMSprop(parameters, weight_decay=config['optimizer']['weight_decay'])
    else:
        optimizer = optim.SGD(params, momentum=0.9, weight_decay=config['optimizer']['weight_decay'],
                              nesterov=(config['optimizer']['type'] == "nesterov"))

    return optimizer

def main():
    
    # Initialize hyperparameters/variables
    config = {}
    config['backbone_name'] = "darknet_53"
    config['backbone_pretrained'] = "../weights/darknet53_weights_pytorch.pth" # set empty to disable
    
    config['anchors'] = [[[116, 90], [156, 198], [373, 326]],
                                [[30, 61], [62, 45], [59, 119]],
                                [[10, 13], [16, 30], [33, 23]]]
    config['classes'] = 20
    
    config['backbone_lr'] = 0.001
    config['other_lr'] = 0.01
    config['freeze_backbone'] = False   #  freeze backbone wegiths to finetune
    config['decay_gamma'] = 0.1
    config['decay_step'] = 20          #  decay lr in every ? epochs
    
    config['optimizer_type'] = "sgd"
    config['optimizer_weight_decay'] =  4e-05
    
    config['batch_size'] = 16  # Number of training samples per batch to be passed to network
    config['epochs'] = 50  # Number of epochs to train the model
    config['img_h'] = config['img_w'] = 416,
    config['seed'] = np.random.seed()
    config['working_dir'] = "./states"     #  replace with your working dir
    config['pretrain_snapshot'] = ""       #  load checkpoint
    config['try'] = 0,
   
    
    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)
                                     

    # Create tf_summary writer
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    train(config)

if __name__ == "__main__":
    main()