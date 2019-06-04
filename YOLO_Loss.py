import numpy as np
import math
import torch
import torch.nn as nn

from bbox import bbox_iou


class YoloLoss(nn.Module):
    def __init__(self, num_classes, img_size, anchors):
        
        # Initalize input variables
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.bbox_attribs = num_classes + 5
        self.img_size = img_size
        self.anchors = anchors
        self.num_anchors = len(anchors)
        
        # Define hyperparameters
        self.threshold = 0.5
        self.lambda_coord = 2.5
        self.lambda_conf = 1.0
        self.lambda_class = 1.0

        # Define loss functions
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss()

    def forward(self, inputs, targets=None):
        # Initialize input variables
        batch_size = np.size(inputs,0)
        grid_w = np.size(inputs,2)
        grid_h = np.size(inputs,3)
        str_w = self.img_size[0] / grid_w
        str_h = self.img_size[1] / grid_h
        anchors_scaled = [(anch_w / str_w, anch_h / str_h) for anch_w, anch_h in self.anchors]

        pred = inputs.view(batch_size, self.num_anchors,
                                self.bbox_attribs, grid_w, grid_h).permute(0, 1, 3, 4, 2).contiguous()

        # Get the outputs from the inputs
        x = torch.sigmoid(pred[...,0])          
        y = torch.sigmoid(pred[...,1])          
        w = pred[...,2]                         
        h = pred[...,3]                         
        conf = torch.sigmoid(pred[...,4])       
        pred_class = torch.sigmoid(pred[...,5:])  

        if targets is not None:
            # Parse variables from targets
            mask, noobj_mask, t_x, t_y, t_w, t_h, t_conf, t_class = self.parse_targets(targets, anchors_scaled,
                                                                               grid_w, grid_h,
                                                                               self.threshold)
            # Move variables to CUDA device
            mask, noobj_mask, t_x, t_y, t_w, t_h, t_conf, t_class = mask.cuda(), noobj_mask.cuda(), \
                        t_x.cuda(), t_y.cuda(), t_w.cuda(), t_h.cuda(), t_conf.cuda(), t_class.cuda()

            # Calculate the losses between the network outputs and targets
            loss_x = self.bceloss(x*mask, t_x*mask)
            loss_y = self.bceloss(y*mask, t_y*mask)
            loss_w = self.mseloss(w*mask, t_w*mask)
            loss_h = self.mseloss(h*mask, t_h*mask)
            loss_conf = self.bceloss(conf*mask, mask) + \
                0.5*self.bceloss(conf*noobj_mask, noobj_mask*0.0)
            loss_class = self.bceloss(pred_class[mask == 1], t_class[mask == 1])
            loss = (loss_x + loss_y + loss_w + loss_h)*self.lambda_coord + \
               loss_conf*self.lambda_conf + loss_class*self.lambda_class

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                loss_h.item(), loss_conf.item(), loss_class.item()
        
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, grid_w-1, grid_w).repeat(grid_w, 1).repeat(
                batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, grid_h-1, grid_h).repeat(grid_h, 1).t().repeat(
                batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            # Calculate anchor w, h
            anch_w = FloatTensor(anchors_scaled).index_select(1, LongTensor([0]))
            anch_h = FloatTensor(anchors_scaled).index_select(1, LongTensor([1]))
            anch_w = anch_w.repeat(batch_size, 1).repeat(1, 1, grid_h * grid_w).view(w.shape)
            anch_h = anch_h.repeat(batch_size, 1).repeat(1, 1, grid_h * grid_w).view(h.shape)
            # Add offset and scale with anchors
            pred_boxes = FloatTensor(pred[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anch_w
            pred_boxes[..., 3] = torch.exp(h.data) * anch_h
            # Results
            _scale = torch.Tensor([str_w, str_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                                conf.view(batch_size, -1, 1), pred_class.view(batch_size, -1, self.num_classes)), -1)
            return output.data
        
    def parse_targets(self, targets, anchors, grid_w, grid_h, threshold):
        
        # Initalize variables
        batch_size = targets.size(0)
        mask = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        noobj_mask = torch.ones(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        t_x = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        t_y = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        t_w = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        t_h = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        t_conf = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, requires_grad=False)
        t_class = torch.zeros(batch_size, self.num_anchors, grid_w, grid_h, self.num_classes, requires_grad=False)
        
        # Calculate values
        for b in range(batch_size):
            for t in range(targets.shape[1]): 
                if targets[b,t].sum() == 0:
                    continue

                #Convert positions to make them relative to box
                g_x = targets[b,t,1]*grid_w
                g_y = targets[b,t,2]*grid_h
                g_w = targets[b,t,3]*grid_w
                g_h = targets[b,t,4]*grid_h

                # Get the indices of the grid box
                g_i = int(g_x)
                g_j = int(g_y)

                # Get shape of ground truth box
                ground_truth_box = torch.FloatTensor(np.array([0, 0, g_w, g_h])).unsqueeze(0)

                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))

                # Calculate the IoU between gt and anchor shapes
                anchor_ious = bbox_iou(ground_truth_box,anchor_shapes)

                # Set mask to zero where the overlap is larger than the threshold
                noobj_mask[b,anchor_ious > threshold,g_j,g_i] = 0

                # Find the best matching anchor box
                n_best = np.argmax(anchor_ious)

                # TODO: add 1st dimension b back in
                mask[b, n_best, g_j, g_i] = 1
                t_x[b, n_best, g_j, g_i] = g_x - g_i
                t_y[b, n_best, g_j, g_i] = g_y - g_j
                t_w[b, n_best, g_j, g_i] = math.log(g_w/anchors[n_best][0] + 1e-16)
                t_h[b, n_best, g_j, g_i] = math.log(g_h/anchors[n_best][1] + 1e-16)
                t_conf[b, n_best, g_j, g_i] = 1
                t_class[b, n_best, g_j, g_i, int(targets[b, t, 0])] = 1
            
        return mask, noobj_mask, t_x, t_y, t_w, t_h, t_conf, t_class