# Detect palm trees on large aerial image with Yolo and ImageAI
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/palm_tree.JPG" alt="Palm trees detected" width = "100%">

This exercise is to demonstrate object detection model training for Geospatial Image processing with YoloV3 and ImageAI module.

If you are new to objection detection I would recommend to read below articles of Moses Olafenwa.
 - ![Object Detection with 10 lines of code][https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606]
 - ![Train Object Detection AI with 6 lines of code written][https://medium.com/deepquestai/train-object-detection-ai-with-6-lines-of-code-6d087063f6ff]
 
For Applications of Deeplearning in Geospatial Field, please read ESRI's tutorial![Use deep learning to assess palm tree health][https://learn.arcgis.com/en/projects/use-deep-learning-to-assess-palm-tree-health/]

This exercise is combination of knowledge gained from these articles.

|![][imageai_10]     |![][imageai_6]      |
:-------------------:|:-------------------:
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/esri_ssd.PNG" alt="ESRI Tutorial" width = "100%">

### Training data preparation
A total of about 400 images of 448 x 448 size are prepared and label partly in ArcGIS Pro and LabelImg. I just used the output tiles left over after ESRI tutorial otherwise slicing the image into tiles could be done by Qtile in QGIS or with python. 
about 10% of the images are used for validation. Train and validation images are stored as follow as needed by ImageAI.

```
>> train    >> images       >> img_1.jpg  (shows Object_1)
            >> images       >> img_2.jpg  (shows Object_2)
            >> images       >> img_3.jpg  (shows Object_1, Object_3 and Object_n)
            >> annotations  >> img_1.xml  (describes Object_1)
            >> annotations  >> img_2.xml  (describes Object_2)
            >> annotations  >> img_3.xml  (describes Object_1, Object_3 and Object_n)
 
>> validation   >> images       >> img_151.jpg (shows Object_1, Object_3 and Object_n)
                >> images       >> img_152.jpg (shows Object_2)
                >> images       >> img_153.jpg (shows Object_1)
                >> annotations  >> img_151.xml (describes Object_1, Object_3 and Object_n)
                >> annotations  >> img_152.xml (describes Object_2)
                >> annotations  >> img_153.xml (describes Object_1)
```
|![Annotation with LabelImg application][labelImg]|![annotation data][xml]    |
:------------:|:-----------:

| ![tiles with annotation overlay][view_images]|
:-----------------:
#### ImageAI installation
It is straight forward as follow
```
!pip install imageai --upgrade
```

#### Model training
There are only 5 lines of code for model training. There is not much controls for hypyer-parameters except batch_size and epochs.
I trained 50 epochs and took about 4 hours.

```
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=data_directory)
trainer.setTrainConfig(object_names_array=object_names, batch_size=batch_size, num_experiments=epochs, train_from_pretrained_model=pretrained_model)
trainer.trainModel()
```

```
            Generating anchor boxes for training images and annotation...
            Average IOU for 9 anchors: 0.90
            Anchor Boxes generated.
            Detection configuration saved in  Palmtrees/json/detection_config.json
            Evaluating over 40 samples taken from Palmtrees/validation
            Training over 353 samples  given at Palmtrees/train
            Training on: 	['palm_tree']
            Training with Batch Size:  10
            Number of Training Samples:  353
            Number of Validation Samples:  40
            Number of Experiments:  5
            Training with transfer learning from pretrained Model
....
....

Epoch 1/5
288/288 [==============================] - 471s 2s/step - loss: 84.0621 - yolo_layer_3_loss: 19.2964 - yolo_layer_4_loss: 23.9211 - yolo_layer_5_loss: 29.2711 - val_loss: 90.1451 - val_yolo_layer_3_loss: 22.2672 - val_yolo_layer_4_loss: 19.9599 - val_yolo_layer_5_loss: 36.3750
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Epoch 2/5
288/288 [==============================] - 394s 1s/step - loss: 62.9945 - yolo_layer_3_loss: 10.2843 - yolo_layer_4_loss: 17.7230 - yolo_layer_5_loss: 23.5400 - val_loss: 71.0814 - val_yolo_layer_3_loss: 16.5870 - val_yolo_layer_4_loss: 19.2088 - val_yolo_layer_5_loss: 23.9582
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Epoch 3/5
288/288 [==============================] - 435s 2s/step - loss: 56.1398 - yolo_layer_3_loss: 10.2244 - yolo_layer_4_loss: 15.2990 - yolo_layer_5_loss: 19.4094 - val_loss: 68.5703 - val_yolo_layer_3_loss: 21.0795 - val_yolo_layer_4_loss: 15.9761 - val_yolo_layer_5_loss: 20.4472
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Epoch 4/5
288/288 [==============================] - 410s 1s/step - loss: 52.0709 - yolo_layer_3_loss: 8.6568 - yolo_layer_4_loss: 14.0305 - yolo_layer_5_loss: 18.4830 - val_loss: 58.9342 - val_yolo_layer_3_loss: 10.3460 - val_yolo_layer_4_loss: 16.1389 - val_yolo_layer_5_loss: 21.7304
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Epoch 5/5
288/288 [==============================] - 434s 2s/step - loss: 48.7902 - yolo_layer_3_loss: 8.4707 - yolo_layer_4_loss: 13.0336 - yolo_layer_5_loss: 16.7292 - val_loss: 53.1895 - val_yolo_layer_3_loss: 11.0294 - val_yolo_layer_4_loss: 12.7339 - val_yolo_layer_5_loss: 19.0254
....

```
#### Detection on UAV image
Although training tile images are created from areal imagery, there is a huge difference in size. While tiles are 448 x 448, original image is about 18,000 x 25,000. As a results, detection directly on the original image produce not output at all. Therefore original image is split into tiles during the detection process and results are stored in a csv file. Location of Bounding boxes are converted to GCS coordinates so that the results could be displacy on the map.
```
image = "Kolovai UAV4R Subset.tif"
chip_h = 448
chip_w = 448
prob_threshold = 25
csv_name = "detection_report.csv"

detect(detector,image,chip_h,chip_w,prob_threshold,csv_name)
```
```
            detection started: 2021-08-10 17:13:55.259247 

            number of object detected:  14292

            detection completed:  2021-08-10 17:21:27.751945 


            detection results are saved in detection_report.csv
```

#### View report with pandas
The csv file contains center coordinates, width, height, aspect_ratio, probibility information of each bounding box of detected palm trees.


| ![][csv_view]   |
:-----------------:

#### view on Map
For quick check, the csv file is viewed in a folium map in the notebook

| ![][folium_map] |
:-----------------:
### Keep your model and results
Once you get out of colab, your trained model, model definition json and deteion report (csv) will be wipe off. Therefore download them before you turn off the browser.
With the trained model and json file you can continue detection on your laptop.

### Try it now!
you can get a notebook with datasets from this link on my github. You open it on Google colab and read and run each cells.

### Credits: 
I would like to thank ImageAI and ESRI for sharing their articales,tutorials and Deeplearning frameworks. Without their sharing, I would not be able to learn Deeplearning application in Geospatial Image processing.

[imageai_10]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_10lines.PNG
[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG
[view_images]:https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/view_images.PNG          
[labelImg]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/labelImg.png
[xml]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/xml.PNG
[csv_view]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/csv_view.PNG

[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG
[folium_map]:https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/folium_map.PNG
