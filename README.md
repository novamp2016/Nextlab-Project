# 변경사항
선행 영상과 후행 영상의 ReID를 위해 input과 output 영상의 매개변수를 리스트 형식으로 넣으셔야 합니다.
video masking도 동일하게 리스트 형식으로 넣여야 적용됩니다. masking이 불필요하다면 None으로 지정하시면 됩니다.

# 발표 슬라이드
https://docs.google.com/presentation/d/1vHx04eHgGDH3_WjalozfwaPxphd7UC_n/edit?usp=sharing&ouid=105425414364534859439&rtpof=true&sd=true

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights
테스트에는 yolo-416 모델을 사용하였습니다.
download file here: https://github.com/AlexeyAB/darknet/

# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ['./data/video/number_cctv.mp4', './data/video/area_cctv.mp4'] --output ['./outputs/number_cctv.mp4', './outputs/area_cctv.mp4']

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to outputs
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video
    (default: ['./data/video/number_cctv.mp4', './data/video/area_cctv.mp4'])
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: ['./outputs/number_cctv.mp4', './outputs/area_cctv.mp4'])
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'mp4v')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
  --video_mask: get video mask if needed
    (default: [None, './data/masking/mask.png'])
```

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
