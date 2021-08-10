# Detect palm trees on large aerial image with Yolo and ImageAI
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/palm_tree.JPG" alt="Palm trees detected" width = "100%">

A demonstration of object detection in drone moasic with Yolo and ImageAI
Object detection in Geospatial Image Processing is never easier than now thanks to Deep learning models

|![][imageai_10]     |![][imageai_6]      |
:-------------------:|:-------------------:
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/esri_ssd.PNG" alt="ESRI Tutorial" width = "100%">

[imageai_10]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_10lines.PNG

[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG


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

![][view_images]

          
[labelImg]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/labelImg.png

[xml]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/xml.PNG

[view_images]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/view_images.PNG
