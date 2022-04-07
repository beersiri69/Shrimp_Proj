import cv2
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


class Loadcam: # load2end use around 1.2 sec
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = max(self.cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 60.0  # 30 FPS fallback    
		#LOGGER.info(f"Success frames {self.w}x{self.h} at {self.fps:.2f} FPS")
		
	def Getvideo(self):
		return self.cap.read()  # guarantee first frame
		
	def Endcam(self):
		self.cap.release()
		cv2.destroyAllWindows()

	def Snap(self):
		print('Snapshot...')
		t_end = time.time() + 2
		while time.time() < t_end:
			_,frame = self.cap.read()
		
		cv2.imwrite('Snap.png',frame)

		#LOGGER.info(f"Snap shot at {save}")  # newline

	def Imgshow():
		img = cv2.imread('Snap.png',3)
		cv2.imshow('frame1',img)

def Detect(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'Snap.png',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.4,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'Re_S',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference):
		):

	M_start_time = time.time()

	source = str(source)	
	save_img = not nosave and not source.endswith('.txt')  # save inference images
	save_dir = increment_path(Path(project)/ name, exist_ok=exist_ok)  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	
	device = select_device(device)
	model = DetectMultiBackend(weights, device=device, dnn=dnn)
	
	stride, names, pt = model.stride, model.names, model.pt
	dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

	model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
	dt, seen = [0.0, 0.0, 0.0], 0
	for path, im, im0s, vid_cap, s in dataset:
		start_time = time.time()
		t1 = time_sync()
		im = torch.from_numpy(im).to(device)
		im = im.half() if half else im.float()  # uint8 to fp16/32
		im /= 255  # 0 - 255 to 0.0 - 1.0
		if len(im.shape) == 3:
			im = im[None]  # expand for batch dim
		t2 = time_sync()
		dt[0] += t2 - t1

        # Inference
		visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
		pred = model(im, augment=augment, visualize=visualize)
		t3 = time_sync()
		dt[1] += t3 - t2

        # NMS
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
		dt[2] += time_sync() - t3

		for i, det in enumerate(pred):  # per image
			seen += 1
			p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # im.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
			s += '%gx%g ' % im.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			imc = im0.copy() if save_crop else im0  # for save_crop
			annotator = Annotator(im0, line_width=line_thickness, example=str(names))
			if len(det):
           
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
				count_n = np.array([0])
				count = ''
				# Print results
				# Counting obj
				j = 0
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					count_n[j] = n
					count += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				LOGGER.info(f'')
				LOGGER.info(f'count by detect {count}')
				LOGGER.info(f'')
				# Write results
				for *xyxy, conf, cls in reversed(det):
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * len(line)).rstrip() % line + '\n')

					if save_img or save_crop or view_img:  # Add bbox to image
						c = int(cls)  # integer class
						label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
						annotator.box_label(xyxy, label, color=colors(c, True))
						if save_crop:
							save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
				im0 = annotator.result()
				if view_img:

					# add fps
					fps = 1.0/(time.time() - start_time)
					cv2.putText(im0,str(fps), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,255), 1, cv2.LINE_4)

					cv2.imshow('Yeah', im0)
					cv2.waitKey(1)  # 1 millisecond

				# Save results (image with detections)
				
				cv2.imwrite(save_path, im0)
					
		# Print results
		t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
		LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
		lap = time.time() - M_start_time
		LOGGER.info(f'Use {lap} sec')
		if save_txt or save_img:
			s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
			LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
		if update:
			strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
		
	return int(count_n)
		

def main():

		
		cam0 = Loadcam()
		m = Detect()
		
		while(True):
			x=0
			num = 0
			Number = input('Input number: ')

			
			if Number.isnumeric(): # Check for int
				Number = int(Number)
				if Number == 0: # exit main loop
					break
			else :
				print("input must be integer!")
				continue

			while(num<Number): # Snap soht loop
				x = input(f"z for snapshot ({x}): ")
				if x == 'z':
					x = 'a'
					Loadcam.Snap(cam0)
					# Loadcam.Imgshow()
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()
					num = num + Detect()

					print(f"Count {num}")


if __name__ == "__main__":
    main()
