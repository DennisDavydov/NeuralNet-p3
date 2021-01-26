import cv2
import numpy as np

'''['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON',
 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN',
 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP',
 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK',
 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']'''
erazing = False
drawing = False # true if mouse is pressed
ix,iy = -1,-1
 
 
# mouse callback function
def draw_mode(event,x,y,flags,param):
    global ix,iy,drawing, erazing

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),15,(0,0,0),-1)
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            x2, y2 = x, y
            leash = np.sqrt(np.square(ix-x) + np.square(iy-y)) 
            print(leash)
            if leash >= 10:
                x2, y2 = int(ix + (0.1*(x-ix))), int(iy + (0.1*(y-iy)))
                cv2.line(img, (ix,iy), (x2,y2), (0,0,0), 30)
                ix, iy = x2, y2
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),15, (0,0,0),-1)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),50,(255,255,255),-1)
        erazing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if erazing == True:
            cv2.line(img, (ix,iy), (x,y), (255,255,255), 100)
            ix, iy = x, y
    elif event == cv2.EVENT_RBUTTONUP:
        erazing = False
        cv2.circle(img,(x,y),50, (255,255,255),-1)
        

# Create a white image, a window and bind the function to window
img = np.ones((560,560,1), np.uint8) * 255
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_mode)
cv2.imshow('image',img)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('e'):
        img2 = cv2.resize(img, (28, 28))
        cv2.imshow('resize', img2)
cv2.destroyAllWindows()