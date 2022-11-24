import cv2
import time
import threading
import numpy as np
from facenet_pytorch import MTCNN
import torch
import cv2
import time
from PIL import Image, ImageDraw, ImageStat
from pynput.mouse import Controller

class Tracker():
    def __init__(self,input_path = 0,output_path = "./output.mp4"):

        # Setting up vision model for eye-tracking
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MTCNN(keep_all=True, device=device)
        self.v_cap = cv2.VideoCapture(input_path)
        self.fresh = FreshestFrame(self.v_cap)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        v_res = (int(self.v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        v_fps = self.v_cap.get(cv2.CAP_PROP_FPS)
        self.output_video = cv2.VideoWriter(output_path, fourcc, v_fps, v_res)

        # Setting up mouse tracking
        self.mouse = Controller()

    def capture(self):
        cnt = 0
        # The fresh frame helps stop race conditions and assures we are collecting the frames and mouse coords that correspond to each other
        ret, frame = self.fresh.read(seqnumber=cnt+1)
        # get mouse position:
        p = self.mouse.position
        t =time.time()
        e = (None,None)
        bounding_box = self.model.detect(frame,landmarks=True)
        if bounding_box[2] is not None:
            right_eye_coors = (bounding_box[2][0][0][0], bounding_box[2][0][0][1])
            left_eye_coors = (bounding_box[2][0][1][0], bounding_box[2][0][1][1])
            e = (right_eye_coors, left_eye_coors)
        return p, e, t


        # print(bounding_box[2][0])
        # right_eye_coors = [int(bounding_box[2][0][0][0]),int(bounding_box[2][0][0][1])]
        # left_eye_coors = [int(bounding_box[2][0][1][0]),int(bounding_box[2][0][1][1])]
        # #testing to see if i got the right coordinates
        # cv2.circle(frame, tuple(right_eye_coors), 20,(255,0,0))
        # cv2.circle(frame, tuple(left_eye_coors), 20,(255,0,0))
        # cv2.imwrite("test.jpg",frame)

    def stop_camera(self):
        self.fresh.release()
        self.v_cap.release()


class FreshestFrame(threading.Thread):
    def __init__(self, capture, name="FreshestFrame"):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(
                    lambda: self.latestnum >= seqnumber, timeout=timeout
                )
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)
