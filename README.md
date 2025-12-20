# DropMOT
**Multi-object detection and tracking for microfluidic droplet videos using pre-trained YOLOv4 and tracking algorithms**

**This repository will continue to be updated as new methods, improvements, and bug fixes for droplet detection and tracking are developed.**



## Step 1
Download pre-trained weights (yolov4_1ob_best.weights) of the YOLOv4 model for droplet detection from https://zenodo.org/records/10938306

## Step 2
Create a virtual environment
```bash
conda env create -f environment.yml
conda activate yolo
```

## Step 3
```bash
python yolov4_trackpy.py
```

## Step 4
Deactivate the virtual environment
```bash
conda deactivate
```

