import torch
from boundbox import IOU

def compute_AP(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Arguments:
    ----------
        recall:    The recall curve (list).
        precision: The precision curve (list).

    Returns:
    --------
        The average precision.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return 0

def NMS(prediction, num_classes=20, conf_thresh=0.5, nms_thresh=0.4):
    """
    Removes detections with lower object confidence score than threshold and
    performs Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)

    Arguments:
    ----------
        prediction:  Label in form (x, y, w, h, objectness_score, class_scores) X B
        num_classes: Defaulted to 20 for PascalVOC2012 dataset
        conf_thresh: threshold for object confidence score
        nms_thresh:  theshold for suppression of objects

    Returns:
    --------

    """

    ### Conversion  of bounding box abs coordinates is done in boundbox.py ###

    bbox = prediction[:,:,:4]
    obj_score = prediction[:,:,4]
    cls_scores = prediction[:,:,5:]

    output = [None for _ in range(len(prediction))]

    for ii, pred in enumerate(prediction):
        # removes confidence scores below threshold
        conf_mask = (pred[:,4] >= conf_thresh).squeeze()
        pred = pred[conf_mask]

        # there are no objects that meet the threshold, so go to the next prediction
        if not pred.size(0):
            continue

        # gets score and class with highest confidence
        class_conf, class_pred = torch.max(pred[:,5:], 1, keepdim=True)
        detection = torch.cat((pred[:, :5], class_conf.float(), class_pred.float()), 1)

        unique_labels = detection[:, -1].unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for l in unique_labels:
            # gets detections of a particular class and sorts it by max object confidence score
            class_detection = detection[detection[:, -1] == l]
            _, sorted_conf = torch.sort(class_detection[:,4], descending=True)
            class_detection = class_detection[sorted_conf]

            # non-maximum suppression
            max_detections = []
            while class_detection.size(0):
                # get detection with highest confidence
                max_detections.append(class_detection[0].unsqueeze(0))
                if len(class_detection) == 1:
                    break

                # get IOUs for all boxes with lower confidence
                ious = IOU(max_detections[-1], class_detection[1:])
                # Remove detections with IoU >= NMS threshold
                class_detection = class_detection[1:][ious < nms_thresh]

            max_detections = torch.cat(max_detections).data

            # add max detections to outputs
            output[ii] = max_detections if output[ii] is None else torch.cat((output[ii], max_detections))


    return output
