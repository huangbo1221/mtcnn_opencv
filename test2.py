import cv2
import numpy as np
from utils import NMS
from utils import BBoxPadSquare
from utils import BoxRegression
from utils import BBoxPad

cap_img = False
class MTCNN(object):
    def __init__(self,pnet_prototxt_file,pnet_caffemodel,rnet_prototxt_file,rnet_caffemodel,onet_prototxt_file,onet_caffemodel,
                 min_face = 48,thresh=[0.699,0.699,0.699],scale_factor=0.709,batch_size=128):
        self.pnet = cv2.dnn.readNetFromCaffe(pnet_prototxt_file,pnet_caffemodel)
        self.rnet = cv2.dnn.readNetFromCaffe(rnet_prototxt_file,rnet_caffemodel)
        self.onet = cv2.dnn.readNetFromCaffe(onet_prototxt_file,onet_caffemodel)
        self.min_face = min_face
        self.thresh = thresh
        self.scale_factor = scale_factor
        self.batch_size = 128

    def generate_scales(self,img):
        scales = []
        h,w = img.shape[:2]
        scale = 12.0 / self.min_face
        minWH = min(h,w) * scale
        while minWH > 12:
            scales.append(scale)
            minWH *= self.scale_factor
            scale *= self.scale_factor
        return scales

    def pnet_detect(self,img,thresh):
        scales = self.generate_scales(img)
        h,w = img.shape[:2]
        pnet_boxes = []
        for idx in range(len(scales)):
            ws = int(np.ceil(w * scales[idx]))
            hs = int(np.ceil(h * scales[idx]))
            resized_img = cv2.resize(img,(ws,hs),0,0,cv2.INTER_LINEAR)
            blob = cv2.dnn.blobFromImage(resized_img, 1.0 / 255.0, None, (0, 0, 0), False)

            self.pnet.setInput(blob)
            detections = self.pnet.forward(["conv4-2", "prob1"])
            reg = np.squeeze(detections[0])
            score = np.squeeze(detections[1][:, 1, :, :])

            score_h, score_w = score.shape
            total_boxes = []
            for i in range(score_h):
                for j in range(score_w):
                    if score[i, j] < 1 - 0.6999:
                        tmp = []
                        xmin = j * 2 / scales[idx]
                        ymin = i * 2 / scales[idx]
                        xmax = (j * 2 + 12 - 1) / scales[idx]
                        ymax = (i * 2 + 12 - 1) / scales[idx]
                        tmp.extend([xmin, ymin, xmax, ymax])
                        tmp.extend(reg[:, i, j])
                        tmp.append(score[i, j])
                        total_boxes.append(tmp)
            if len(pnet_boxes) == 0 and len(total_boxes) > 0:
                pnet_boxes = np.array(total_boxes)
            elif len(pnet_boxes) > 0 and len(total_boxes) > 0:
                pnet_boxes = np.concatenate((pnet_boxes,np.array(total_boxes)),axis=0)
        if len(pnet_boxes) == 0:
            return None
        pnet_sorted_total_boxes = pnet_boxes[np.lexsort(-pnet_boxes.T)]  # 按最后一列排序
        pnet_nms_boxes,_ = NMS(pnet_sorted_total_boxes)
        pnet_reg_boxes = BoxRegression(pnet_nms_boxes)
        pnet_pad_boxes = BBoxPadSquare(pnet_reg_boxes, w, h)
        return pnet_pad_boxes

    def rnet_detect(self,image ,boxes,thresh):
        batchs = boxes.shape[0]
        inputs = []
        for i in range(batchs):
            roi = image[int(boxes[i, 1]):int(boxes[i, 3]), int(boxes[i, 0]):int(boxes[i, 2])]
            roi = cv2.resize(roi, (24, 24))
            inputs.append(roi)
        inputs = np.array(inputs)
        blobs = cv2.dnn.blobFromImages(inputs, 0.0078125, None, (127.5, 127.5, 127.5), False)
        self.rnet.setInput(blobs)
        rnet_detections = self.rnet.forward(["conv5-2", "prob1"])

        rnet_scores = rnet_detections[1][:, 1]
        rnet_boxes = rnet_detections[0]

        boxes = boxes[rnet_scores > thresh]
        if len(boxes) == 0:
            return []
        rnet_boxes = rnet_boxes[rnet_scores > thresh]
        boxes[:, 4:8] = rnet_boxes
        boxes[:, -1] = rnet_scores[rnet_scores > thresh]
        return boxes

    def onet_detect(self,image,boxes,thresh):
        batchs = boxes.shape[0]
        inputs = []
        for i in range(batchs):
            roi = image[int(boxes[i, 1]):int(boxes[i, 3]), int(boxes[i, 0]):int(boxes[i, 2])]
            roi = cv2.resize(roi, (48, 48))
            inputs.append(roi)
        inputs = np.array(inputs)
        blobs = cv2.dnn.blobFromImages(inputs, 0.0078125, None, (127.5, 127.5, 127.5), False)
        self.onet.setInput(blobs)
        onet_detections = self.onet.forward(["conv6-2", "conv6-3", "prob1"])

        onet_scores = onet_detections[2][:, 1]
        onet_landmarks = onet_detections[1]
        onet_boxes = onet_detections[0]

        boxes = boxes[onet_scores > thresh]
        if len(boxes) == 0:
            return [],[]
        onet_boxes = onet_boxes[onet_scores > thresh]
        boxes[:, 4:8] = onet_boxes
        boxes[:, -1] = onet_scores[onet_scores > thresh]
        onet_landmarks = onet_landmarks[onet_scores > thresh]
        w_ = boxes[:, 2] - boxes[:, 0] + 1
        h_ = boxes[:, 3] - boxes[:, 1] + 1
        for i in range(5):
            onet_landmarks[:, 2 * i] = onet_landmarks[:, 2 * i] * w_ + boxes[:, 0]
            onet_landmarks[:, 2 * i + 1] = onet_landmarks[:, 2 * i + 1] * h_ + boxes[:, 1]
        return boxes,onet_landmarks


    def detect_face(self,img):
        pnet_boxes = self.pnet_detect(img,self.thresh[0])
        if pnet_boxes is None:
            return None,None
        pnet_num_boxes = pnet_boxes.shape[0]
        iters = int(np.ceil(pnet_num_boxes / self.batch_size))
        total_rnet_boxes = []
        for i in range(iters):
            start = i * self.batch_size
            end = min(start + self.batch_size, pnet_num_boxes)
            tmp = self.rnet_detect(img, pnet_boxes[start:end], self.thresh[1])
            if len(total_rnet_boxes) == 0 and len(tmp) > 0:
                total_rnet_boxes = tmp
            elif len(total_rnet_boxes) > 0 and len(tmp) > 0:
                total_rnet_boxes = np.concatenate((total_rnet_boxes, tmp),axis=0)
        if len(total_rnet_boxes) == 0:
            return None,None
        rnet_sorted_total_boxes = total_rnet_boxes[np.lexsort(-total_rnet_boxes.T)]  # 按最后一列排序
        rnet_nms_boxes,_ = NMS(rnet_sorted_total_boxes, 0.4, 'm')
        rnet_reg_boxes = BoxRegression(rnet_nms_boxes)
        rnet_pad_boxes = BBoxPadSquare(rnet_reg_boxes, img.shape[1], img.shape[0])

        rnet_num_boxes = rnet_pad_boxes.shape[0]
        iters = int(np.ceil(rnet_num_boxes / self.batch_size))
        total_onet_boxes = []
        total_landmarks = []
        for i in range(iters):
            start = i * self.batch_size
            end = min(start + self.batch_size, rnet_num_boxes)
            tmp_box,tmp_landmarks = self.onet_detect(img, rnet_pad_boxes[start:end], self.thresh[2])
            if len(total_onet_boxes) == 0 and len(tmp_box) > 0:
                total_onet_boxes = tmp_box
                total_landmarks = tmp_landmarks
            elif len(total_onet_boxes) > 0 and len(tmp_box) > 0:
                total_onet_boxes = np.concatenate((total_onet_boxes, tmp_box), axis=0)
                total_landmarks = np.concatenate((total_landmarks, tmp_landmarks), axis=0)
        if len(total_onet_boxes) == 0:
            return None,None
        onet_reg_boxes = BoxRegression(total_onet_boxes)
        onet_nms_boxes,valid_index = NMS(onet_reg_boxes,0.4, 'm')
        total_landmarks = total_landmarks[valid_index < 1]
        onet_pad_boxes = BBoxPad(onet_nms_boxes, img.shape[1], img.shape[0])
        return onet_pad_boxes,total_landmarks


if __name__ =="__main__":
    mtcnn = MTCNN('det1_.prototxt','det1_.caffemodel','det2.prototxt','det2_half.caffemodel','det3-half.prototxt','det3-half.caffemodel',
                  24, [0.699, 0.699, 0.699], 0.709, 128)
    if cap_img:
        cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
        while True:
            _,frame = cap.read()
            onet_pad_boxes,landmarks = mtcnn.detect_face(frame)
            if onet_pad_boxes is not None:
                for i in range(onet_pad_boxes.shape[0]):
                    xmin = int(onet_pad_boxes[i, 0])
                    ymin = int(onet_pad_boxes[i, 1])
                    xmax = int(onet_pad_boxes[i, 2])
                    ymax = int(onet_pad_boxes[i, 3])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
                    for j in range(5):
                        cv2.circle(frame,(int(landmarks[i,2*j]),int(landmarks[i,2*j+1])),2,(0,255,0),1)
            cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        img = cv2.imread("oscar.jpg")
        onet_pad_boxes,landmarks = mtcnn.detect_face(img)
        for i in range(onet_pad_boxes.shape[0]):
            xmin = int(onet_pad_boxes[i,0])
            ymin = int(onet_pad_boxes[i, 1])
            xmax = int(onet_pad_boxes[i, 2])
            ymax = int(onet_pad_boxes[i, 3])
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0))
            for j in range(5):
                cv2.circle(img, (int(landmarks[i, 2 * j]), int(landmarks[i, 2 * j + 1])), 2, (0, 255, 0), 1)
        cv2.imshow('test',img)
        cv2.imwrite('bobo3.jpg',img)
        cv2.waitKey(0)