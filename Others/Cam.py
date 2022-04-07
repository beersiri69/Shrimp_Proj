import cv2
import time
class Loadcam: # load2end use around 1.2 sec
	def __init__(self):
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.cap = cv2.VideoCapture(0)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360.0)
		self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = max(self.cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 60.0  # 30 FPS fallback    
		#LOGGER.info(f"Success frames {self.w}x{self.h} at {self.fps:.2f} FPS")
		
	def Getframe(self):
		return self.cap.read()  # guarantee first frame

	def ShowVideo(self):
		pframe = 0 
		nframe = 0
		while True: 
			ret,frame = self.cap.read()

			#Get FPS
			nframe = time.time()
			fps = 1/(nframe-pframe)
			pframe = nframe
			fps = int(fps)
			resolution = str(self.w) + " x " + str(self.h)
			txtshow = resolution + " :" + str(fps)

			cv2.putText(frame, txtshow, (7, 70), self.font, 1, (100, 255, 0), 1, cv2.LINE_AA)

			cv2.imshow('ShowVideo', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		self.Endcam()



	def Endcam(self):
		self.cap.release()
		cv2.destroyAllWindows()

	def Snap(self):
		print('Snapshot...')
		t_end = time.time() + 2
		while time.time() < t_end:
			_,frame = self.cap.read()
		
		# cv2.imwrite('/home/pi/Desktop/Gtamp/Yolo_module/TSnap.png',frame)
		cv2.imwrite('./Yolo_module/Snap.png',frame)
		#LOGGER.info(f"Snap shot at {save}")  # newline

	def Imgshow(x):
		img = cv2.imread('Snap.png')
		cv2.imshow('frame1',img)

