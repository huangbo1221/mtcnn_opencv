import numpy as np

def NMS(boxes,thresh = 0.5, mode = 'u'):
    """
    :param boxes:
    boxes里每一行的形式：[xmin,ymin,xmax,ymax,xmin_reg,ymin_reg,xmax_reg,ymax_reg,score]
    :return:
    """
    boxes_num = boxes.shape[0]
    if boxes_num == 0:
        return None,None
    if boxes_num == 1:
        return boxes,np.array([0])
    tmp_idx = [0] * boxes_num#用来记录已经被删除的box
    selected_idx = 0
    while selected_idx < boxes_num:
        xmin,ymin,xmax,ymax = boxes[selected_idx,:4]
        score = boxes[selected_idx,-1]
        area1 = (xmax - xmin + 1) * (ymax - ymin + 1)
        selected_idx += 1
        if selected_idx == boxes_num:
            break
        for cmp_idx in range(selected_idx,boxes_num):
            if tmp_idx[cmp_idx] == 1:
                continue
            x = max(xmin,boxes[cmp_idx,0])
            y = max(ymin,boxes[cmp_idx,1])
            w = min(xmax,boxes[cmp_idx,2]) - x + 1
            h = min(ymax,boxes[cmp_idx,3]) - y + 1
            if w < 0 or h < 0:
                continue
            area2 = (boxes[cmp_idx,2] - boxes[cmp_idx,0] + 1) * (boxes[cmp_idx,3] - boxes[cmp_idx,1] + 1)
            area3 = w * h
            if mode == 'u':
                if area3 / (area1 + area2 - area3) > thresh:
                    tmp_idx[cmp_idx] = 1
                    continue
            if mode == 'm':
                if area3 / min(area1,area2) > thresh:
                    tmp_idx[cmp_idx] = 1
                    continue
    tmp_idx = np.array(tmp_idx)
    nms_boxes = boxes[tmp_idx < 1]
    return nms_boxes,tmp_idx

def BoxRegression(boxes):
    nums_box = boxes.shape[0]
    if nums_box == 0:
        return None
    w = boxes[:,2] - boxes[:,0] + 1
    h = boxes[:,3] - boxes[:,1] + 1
    boxes[:, 0] += boxes[:,4] * w
    boxes[:, 1] += boxes[:, 5] * h
    boxes[:, 2] += boxes[:, 6] * w
    boxes[:, 3] += boxes[:, 7] * h
    return boxes

def BBoxPadSquare(boxes,origin_w,origin_h):
    origin_w = np.array([origin_w] * boxes.shape[0])
    origin_h = np.array([origin_h] * boxes.shape[0])
    w,h = boxes[:,2] - boxes[:,0],boxes[:,3] - boxes[:,1]
    side = np.max(np.concatenate((w,h),axis=0).reshape(2,-1),axis=0)
    tmp = boxes[:,0] + (w - side) * 0.5
    tmp[tmp < 0] = 0
    boxes[:,0] = np.round(tmp)

    tmp = boxes[:, 1] + (h - side) * 0.5
    tmp[tmp < 0] = 0
    boxes[:, 1] = np.round(tmp)

    boxes[:,2] = np.round(np.min(np.concatenate((boxes[:, 0] + side - 1.0, origin_w - 1.0),axis=0).reshape(2,-1),axis=0))
    boxes[:,3] = np.round(np.min(np.concatenate((boxes[:, 1] + side - 1.0, origin_h - 1.0),axis=0).reshape(2, -1),axis=0))
    return boxes

def BBoxPad(boxes,origin_w,origin_h):
    if sum(boxes[:,0] < 0)> 0:
        boxes[:,0][boxes[:, 0] < 0] = 0
        boxes[:,0] = np.round(boxes[:,0])
    if sum(boxes[:,1] < 0 ) > 0:
        boxes[:, 1][boxes[:, 1] < 0] = 0
        boxes[:, 1] = np.round(boxes[:, 1])

    origin_w = np.array([origin_w] * boxes.shape[0]) - 1
    origin_h = np.array([origin_h] * boxes.shape[0]) - 1
    boxes[:,2] = np.round(np.min(np.concatenate((boxes[:,2],origin_w),axis=0).reshape(2,-1),axis=0))
    boxes[:,3] = np.round(np.min(np.concatenate((boxes[:,3], origin_h), axis=0).reshape(2,-1), axis=0))
    return boxes
if __name__ == '__main__':
    NMS()