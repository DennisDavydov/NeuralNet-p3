import random
import pickle
import numpy as np
import Network
import cv2
import os
import imutils

 

erazing = False
drawing = False # true if mouse is pressed
ix,iy = -1,-1
 
 
# mouse callback function
def draw_mode(event,x,y,flags,param):
    global ix,iy,drawing, erazing

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),15,(255,255,255),-1)
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix,iy), (x,y), (255,255,255), 30)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),15, (255,255,255),-1)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),50,(0,0,0),-1)
        erazing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if erazing == True:
            cv2.line(img, (ix,iy), (x,y), (0,0,0), 100)
            ix, iy = x, y
    elif event == cv2.EVENT_RBUTTONUP:
        erazing = False
        cv2.circle(img,(x,y),50, (0,0,0),-1)
        

#initialize the network and load weights and biases
filepath = os.path.dirname(__file__)+'\w_b'
with open(filepath, 'rb') as file:
    weights, biases = pickle.load(file)

net = Network.Network(None, weights, biases)
# Create a white image, a window and bind the function to window
img = np.zeros((560,560,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_mode)
cv2.imshow('image',img)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('e'):
        img2 = cv2.blur(img,(20, 20))
        
        img2 = cv2.resize(img2, (28, 28))
        
        cv2.imshow('im2', img2)
        img2 = np.reshape(img2[:,:,1], (1, 784))
        img2 = img2.transpose()
        #print(img2)
        #print(net.feedforward(img2).shape)
        a = net.feedforward(img2/255)
        print(a[np.argmax(a)])
        print(np.argmax(a))
    elif k == ord('r'):
        img = imutils.rotate(img, 10)
        
        
        
cv2.destroyAllWindows()

