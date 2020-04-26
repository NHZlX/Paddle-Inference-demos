##  Mask detection

The project mainly realized the mask detection function

1) cd models && bash model_downloads.sh
2) Run `python cam_video.py`, the program will start the camera on your machine, and then carry out face mask detection.

The project mainly contains the following files:


- config.py   
	```
	It contains configure information of the entire project, such as the path of the 	model and the model output threshold etc.
	```

- mask_pred.py  
	```
	Realized the process of mask prediction, which contains face detection and mask classification.
	``` 
- models/pd_model.py  
	```
	The file mainly shows how to use the paddle inference api interface to make inference, The face detect and mask classify models construction in the mask_pred.py file is based on this.
	```
- models/preprocess.py  
	```
	Contains the image pre-processing functions of the two models.
	```
- models/pyramidbox_lite.      
	```
	model of the face detection.
	```
- models/mask_detect.     
	```
	model of the face mask classification.
	```
- assets.      
	``` 
	Used to store test images or other files
	```
