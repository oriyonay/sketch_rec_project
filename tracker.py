import cv2
import time
import threading
import numpy as np
import torch
import cv2
import time
from PIL import Image, ImageDraw, ImageStat
from pynput.mouse import Controller
import mediapipe as mp



class Tracker():
    def __init__(self,input_path = 0,output_path = "./output.mp4"):

        # Setting up vision model for eye-tracking
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #setting up mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh

        # setting up recording
        self.v_cap = cv2.VideoCapture(input_path, cv2.CAP_DSHOW)

        # self.v_cap.set(3, 1280)
        # self.v_cap.set(4, 720)

        self.fresh = FreshestFrame(self.v_cap)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        v_res = (int(self.v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        v_fps = self.v_cap.get(cv2.CAP_PROP_FPS)
        print("resolution:",v_res)
        self.output_video = cv2.VideoWriter(output_path, fourcc, v_fps, v_res)
        # Setting up mouse tracking
        self.mouse = Controller()

    def capture(self):
        cnt = 0
        # The fresh frame helps stop race conditions and assures we are collecting the frames and mouse coords that correspond to each other
        ret, frame = self.fresh.read(seqnumber=cnt+1)

        # get mouse position:
        p = self.mouse.position
        p = [p[0],p[1]]
        t =time.time()
        e = self.pupil_coords(frame)

        return p, e, t


    def stop_camera(self):
        self.fresh.release()
        self.v_cap.release()
        self.output_video.release()

    def getLandmarks(self,image):
        face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.7,
                                          refine_landmarks=True, max_num_faces=1)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
        # print(results.multi_face_landmarks[0])
        if results.multi_face_landmarks is not None:
            landmarks = results.multi_face_landmarks[0].landmark
            return landmarks, results
        return None, None

    def pupil_coords(self,frame):

        landmarks,results = self.getLandmarks(frame)
        if landmarks is not None:
            #get just right and left center pupil coordinates
            iris_landmarks = [landmarks[473], landmarks[468]]
            coords = []
            for landmark in iris_landmarks:
                shape = frame.shape
                x = landmark.x
                y = landmark.y
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                frame = cv2.circle(frame, (relative_x, relative_y), radius=1, color=(0, 0, 255), thickness=-1)

                self.output_video.write(frame)
                # cv2.imshow('frame',frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                coords.append([relative_x,relative_y])
            return coords
        return [None,None]

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
