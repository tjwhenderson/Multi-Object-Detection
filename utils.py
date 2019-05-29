from boundbox import IOU

def compute_AP(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision.
    """
    ## TODO!
    return 0

def NMS(prediction, num_classes, t_conf=0.5, t_nms=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # convert center_x, center_y, width, height to x_min, x_max, y_min, y_max
    bbox = prediction.new(prediction.shape)
    bbox[:,:,0] = prediction[:,:,0] - prediction[:,:,2] / 2
    bbox[:,:,1] = prediction[:,:,1] - prediction[:,:,3] / 2
    bbox[:,:,2] = prediction[:,:,0] - prediction[:,:,2] / 2
    bbox[:,:,3] = prediction[:,:,1] - prediction[:,:,3] / 2

    prediction[:,:,4] = bbox[:,:,4]
    output = [None for ii in range(len(prediction))]

    for img, pred in enumerate(prediction):
        # Filter confidence scores below threshold
        conf_mask = (pred[:,4] >= t_conf).squeeze()
        pred = pred[conf_mask]

        if not pred.size(0):
            continue

        # gets score and class with highest confidence
        class_conf, class_pred = torch.max(pred[:, :5], class_conf.float(), class_pred.float()), 1)

        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = IOU(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output
