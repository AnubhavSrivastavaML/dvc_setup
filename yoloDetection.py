import numpy as np
import cv2
import os
import glob
import tempfile
import time


class DETECTOR:

    def __init__(self,cfgPath,weightPath,labelPath,blob=(512,512),coordformat='default'):
        self.cfgfile = cfgPath
        self.weightfile = weightPath
        self.vehicle_net = cv2.dnn.readNetFromDarknet(self.cfgfile, self.weightfile)
        self.ln = [self.vehicle_net.getLayerNames()[(i[0] - 1)] for i in self.vehicle_net.getUnconnectedOutLayers()]
        with open(labelPath,'r+') as file:
        	self.labels = [label.strip() for label in file.readlines()]
        self.format = coordformat
        self.blobSize = blob
        print("Inferencing with blob shape {}\nlabels {} \n".format(self.blobSize,self.labels))
        print("Call detect(detect(image , score_threshold = 0.25,filterLabel=None)) for getting inference with parameters \n")

        if coordformat == 'default' : 
            print("Output are in the formats [[x, y, w, h,label,confidence]] ")
        else:
            print("Output are in the formats [[xmin, ymin, xmax, ymax,label,confidence]]")      


    
    def detect(self, image , score_threshold = 0.25,filterLabel=None):
        H, W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image,0.003,self.blobSize, swapRB=True, crop=False)
        self.vehicle_net.setInput(blob)
        t=time.time()
        layerOutputs = self.vehicle_net.forward(self.ln)
        print("Time taken for feed formard : ",time.time()-t)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence >= score_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype('int')
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)
        info = []
        scores = []
        if len(idxs) > 0:
            for i in idxs.flatten():
            	if filterLabel:
            		v_type = self.labels[classIDs[i]]
            		if v_type not in filterLabel:
            			continue
            	
            	else:
            		v_type = self.labels[classIDs[i]]
            				
            	x, y = max(0,boxes[i][0]), max(0,boxes[i][1])
            	w, h = boxes[i][2], boxes[i][3]
            	conf = confidences[i]
                
            	if self.format == 'max':                	
                	info.append([x, y, x+w, y+h,v_type,conf])
            	else:
                	info.append([x, y, w, h,v_type,conf])
                	 

		#v_type = self.labels[classIDs[i]]
		#info.append([x, y, w, h])
		#scores.append(conf)

        return info




