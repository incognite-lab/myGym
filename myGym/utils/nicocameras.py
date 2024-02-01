import threading
import time
import numpy as np
import cv2
import os
import requests
import zipfile
import io
import subprocess

def download(path,url):
    if os.path.exists(path):
        return
    print("downloading",path)
    response = requests.get(url)
    if response.ok:
        file_like_object = io.BytesIO(response.content)
        zipfile_object = zipfile.ZipFile(file_like_object)    
        zipfile_object.extractall(".")

def initializeCameraControls():
    download("v4l2-ctl.exe","https://www.agentspace.org/download/v4l2-ctl.zip")  

def getCameraControls(id,controls):
    command = [ "v4l2-ctl", "-d", f"/dev/video{id}" ]
    for control in controls:
        command += [ "-C", f"{control}" ]
    data = subprocess.check_output(command)
    output = data.decode()
    values = output.split('\n')
    #return [ int(value) for value in values[:-1] ]
    return [ 0, 0, 0]
def setCameraControls(id,controls):
    command = [ "v4l2-ctl", "-d", f"/dev/video{id}" ]

    for control in controls.keys():
        value = controls[control]
        command += [ "-c", f"{control}={value}" ]
        _ = subprocess.check_output(command)

def getCameraDevices(name):
    command = [ "v4l2-ctl", "--list-devices" ]
    data = subprocess.check_output(command)
    output = data.decode()
    names = output.split('\r\n')
    ids = [ id for id in range(len(names)) if name in names[id] ]
    return ids

class NicoCameras:

    def __init__(self):
        self.stopped = False
        initializeCameraControls()
        self.ids = [2,4]
        if len(self.ids) == 0:
            self.ids = [0,0]
        elif len(self.ids) == 1:
            id = self.ids[0]
            self.ids = [id,id]
        print('camera ids:',self.ids)
        self.frames = {}
        self.fpss = {}
        for id in self.ids:
            self.frames[id] = None
            self.fpss[id] = 0
        time.sleep(1)
        print('starting camera threads')
        self.threads = []
        launched=[]
        for i in range(len(self.ids)):
            if self.ids[i] in launched:
                continue
            thread = threading.Thread(name="camera"+str(i), target=self.grabbing, args=(self.ids[i],))
            thread.start()
            self.threads.append(thread)
            launched.append(self.ids[i])
        self.zooms = {}
        self.tilts = {}
        self.pans = {}
        for id in self.ids:
            zoom, tilt, pan = getCameraControls(id,['zoom_absolute','tilt_absolute','pan_absolute'])
            self.zooms[id] = zoom
            self.tilts[id] = tilt
            self.pans[id] = pan
        self.tocheck = True

    def grabbing(self,id):
        print(f'grabbing thread {id} started')
        #camera = cv2.VideoCapture(id,cv2.CAP_MSMF)
        camera = cv2.VideoCapture(id,cv2.CAP_DSHOW)
        fps = 30 
        camera.set(cv2.CAP_PROP_FPS,fps)
        fps = 0
        t0 = time.time()
        while True:
            hasFrame, self.frames[id] = camera.read()
            if not hasFrame or self.stopped:
                break
            t1 = time.time()
            if int(t1) != int(t0):
                self.fpss[id] = fps
                #print('camera',id,fps,'fps',self.frames[id].shape if self.frames[id] is not None else 'none')
                fps = 0
                t0 = t1
            fps += 1
            cv2.waitKey(1)

    def read(self):
        frames = tuple([ self.frames[id] for id in self.ids ])
        if self.tocheck:
            if len(frames) == 2 and frames[0] is not None and frames[1] is not None and frames[0].shape == frames[1].shape:
                if self.check(frames[0],frames[1]):
                    self.ids.reverse()
                    frames = list(frames)
                    frames.reverse()
                    frames = tuple(frames)    
                self.tocheck = False
        return frames

    def fps(self):
        return ( self.fpss[id] for id in self.ids )
        
    def getZoom(self):
        return ( self.zooms[id] for id in self.ids )
        
    def getTilt(self):
        return ( self.tilts[id] for id in self.ids )

    def getPan(self):
        return ( self.pans[id] for id in self.ids )

    def setZoom(self,i,zoom):
        self.zooms[self.ids[i]] = zoom
        setCameraControls(self.ids[i],{'zoom_absolute':zoom})
        
    def setTilt(self,i,tilt):
        self.tilts[self.ids[i]] = tilt
        setCameraControls(self.ids[i],{'tilt_absolute':tilt})

    def setPan(self,i,pan):
        self.pans[self.ids[i]] = pan
        setCameraControls(self.ids[i],{'pan_absolute':pan})
        
    def close(self):
        self.stopped = True

    def check(self,left,right): #TBD: binarize by 'good features to follow'
        return False # means OK
        #left_gray = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
        #right_gray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
        #left_src = 1.0 - np.float32(left_gray)/255.0
        #right_src = 1.0 - np.float32(right_gray)/255.0
        #ret, confidence = cv2.phaseCorrelate(left_src,right_src)
        #return confidence > 0.1 and ret[0] < -1.0


"""
def image_shift_xy(left,right):
    left_gray = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    left_src = 1.0 - np.float32(left_gray)/255.0
    right_src = 1.0 - np.float32(right_gray)/255.0
    ret, confidence = cv2.phaseCorrelate(left_src,right_src)
    return ret
"""

def getTranslationX(affineTransform):
    return affineTransform[0,2]

def getTranslationY(affineTransform):
    return affineTransform[1,2]

#first = True    
def image_shift_xy(left,right):
    """
    global first
    if first:
        cv2.imwrite('left.png',left)
        cv2.imwrite('right.png',right)
        first = False
    """
    
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    left_points = cv2.goodFeaturesToTrack(left_gray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
    
    right_points, status, err = cv2.calcOpticalFlowPyrLK(left_gray, right_gray, left_points, None) 

    indices = np.where(status==1)[0]
    warp_matrix, _ = cv2.estimateAffinePartial2D(left_points[indices], right_points[indices], method=cv2.LMEDS)
    
    """
    left_crop = left[left.shape[0]//4:3*left.shape[0]//4,left.shape[1]//4:3*left.shape[1]//4]
    right_crop = right[right.shape[0]//4:3*right.shape[0]//4,right.shape[1]//4:3*right.shape[1]//4]
    left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)    
    warp_matrix = np.array([[1,0,0],[0,1,0]],np.float32)
    try:
        # use ECC
        warp_mode = cv2.MOTION_AFFINE
        number_of_iterations = 200
        termination_eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        (cc, warp_matrix) = cv2.findTransformECC(left_gray, left_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)
    except Exception as ee:
        # print('ECC diverged:', ee)
        pass
    """
    dx = getTranslationX(warp_matrix)
    dy = getTranslationY(warp_matrix)

    return (dx,dy)
    
