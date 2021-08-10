# Detect palm trees on large aerial image with Yolo and ImageAI
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/palm_tree.JPG" alt="Palm trees detected" width = "100%">

A demonstration of object detection in drone moasic with Yolo and ImageAI
Object detection in Geospatial Image Processing is never easier than now thanks to Deep learning models

|![][imageai_10]     |![][imageai_6]      |
:-------------------:|:-------------------:
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/esri_ssd.PNG" alt="ESRI Tutorial" width = "100%">

### train and validation data structure
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
Annotation is done in LabelImg application

|![][labelImg]|![][xml]    |
:------------:|:-----------:

| ![][view_images]|
:-----------------:

#### model training
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
#### detect on UAV image
```
            detection started: 2021-08-10 17:13:55.259247 

            number of object detected:  14292

            detection completed:  2021-08-10 17:21:27.751945 


            detection results are saved in detection_report.csv

```

#### view report

| ![][csv_view]   |
:-----------------:

#### view on Map

| ![][folium_map] |
:-----------------:

[imageai_10]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_10lines.PNG
[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG
[view_images]:https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/view_images.PNG          
[labelImg]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/labelImg.png
[xml]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/xml.PNG
[csv_view]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/csv_view.PNG

[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG
[folium_map]:https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/folium_map.PNG
