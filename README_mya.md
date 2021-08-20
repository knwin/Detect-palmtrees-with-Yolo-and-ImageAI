# လေ့လာဖွယ်ရာ Deeplearning နည်းပညာ - ‌ကောင်းကင်ဓါတ်ပုံ drone mosaic မှ အုံးပင်များ ကို Yolo နှင့် ImageAI သုံးပြီး ခွဲထုတ်ခြင်း
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/palm_tree.JPG" alt="Palm trees detected" width = "100%">

ယခု ဆောင်းပါးသည် Geospatial နယ်ပယ်တွင် YoloV3 နှင့် ImageAI deep learning နည်းပညာသုံးပြီး အရာဝတ္ထုများကို ရှာဖွေနိုင်ပုံကို တင်ပြခြင်း ဖြစ်ပါသည်။

Deep learning နည်းပညာသုံးပြီး အရာဝတ္ထုများကို ရှာဖွေနိုင် သည့်အကြောင်းအရာ နှင့် မရင်းနှီး အသုံးမချဖူးပါက Moses Olafenwa အောက်ပါ ၏ ဆောင်းပါများဖတ်ပြီး စမ်းသပ်ကြည့်ရန် အကြံပြုပါသည်။
 - ![Object Detection with 10 lines of code](https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606)
 - ![Train Object Detection AI with 6 lines of code written](https://medium.com/deepquestai/train-object-detection-ai-with-6-lines-of-code-6d087063f6ff)
 
Geospatial နယ်ပယ်တွင် Deeplearning နည်းပညာ အသုံးချမှုများကို လေ့လာရန် စိတ်ဝင်စားပါက ESRI's ၏ ![အုန်းပင်များ၏ကျန်းမာရေးကို Deeplearning နည်းပညာဖြင့်လေ့လာခြင်း](https://learn.arcgis.com/en/projects/use-deep-learning-to-assess-palm-tree-health/) သင်ခန်းစာ ကို ဖတ်ရှုကြည့်ပါ။ ယင်းသင်ခန်းစာ တွင် code တစ်ကြောင်းမျှမပါပဲ dialog box ဖြင့် အလုပ်လုပ်သွားသည်ကို တွေ့ရပါမည်။ သို့သော် ArcGIS Pro ကို အသုံးပြုထားသဖြင့် လူတိုင်း လိုက်လုပ်နိုင်မည်တော့ မဟုတ်ပါ။

ယခုတင်ပြချက်သည် အထက်ပါ သင်ခန်းစာ နှင့် ဆောင်းပါးများ အပြင် ယခင်ရှေ့က လေ့လာဖူးသော အခြားသူတို့မျှဝေပေးသည့် သင်ခန်းစာများနှင့် ဆောင်းပါးများ  မှ ရရှိလာသော အသိပညာ နှင့် ကျွမ်းကျင်မှုများကို ပြန်လည် အသုံးချထားခြင်း ဖြစ်ပါသည်။

|![][imageai_10]     |![][imageai_6]      |
:-------------------:|:-------------------:
<img src="https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/esri_ssd.PNG" alt="ESRI Tutorial" width = "100%">

### သင်ကြားရန်ပြင်ဆင်ခြင်း
၄၄၈ x ၄၄၈ အရွယ် ပုံပေါင်း၄၀၀ ခန့်ကို မူလ mosaic ကနေဖြတ်ထုတ်ပြီး labelImg app နဲ့ စာထိုးပါတယ်။ (မူလက ESRI turorial လုပ်တုံးက ကျန်ခဲ့တဲ့  image chip တွေကို လည်းဆက်သုံးခဲ့ပါသေးတယ်။) စာထိုးတယ်ဆိုတာ အုန်းပင် ကို ခြုံပြီး လေးထောင့်ကွက်ဆွဲပေးပြီး palm_tree လို့မှတ်ပေးတာပါ။ ပုံစိတ်တာ ကို tiling လုပ်တယ်လို့ ခေါ်ပါတယ်။ ပုံလေးတွေကို tile သို့မဟုတ် chip လို့လည်းခေါ်ပါတယ်။ ဒီပုံလေးတွေထဲက သတ်မှတ်ပေးထားတဲ့ အရာဝတ္ထုတွေကို စက်ကဖတ်ပြီး ဘယ်အရာဝတ္ထုက ဘယ်လို ဆိုတာ သူ့ဖာသာ နားလည်အောင် သင်ယူသွားမှာပါ။ 

ပုံတွေထဲက ၁၀ ရာခိုင်နှုံးကို သင်ပြီးတဲ့ model  က ဘယ်လောက်ကောင်းတယ်ဆိုတာ သိရဖို့ စစ်ဆေးဖို့ခွဲထားပါတယ်။ test/validation images လို့ခေါ်ပါတယ်။ သင်ယူရာမှာ သုံးမယ့် ၉၀ရာခိုင်နှုံးပုံ‌တွေ ကိုတော့ training images လို့ခေါ်ပါတယ်။ စာထိုးထားတဲ့ လေးထောင့်ကွက်တွေက xml file နဲ့သပ်သပ်သိမ်းထားလို့ xml ဖိုင် တွေကိုလည်း annotations ဆိုတဲ့ဖိုဒါတွေအောက်မှာ သိမ်းထားရပါတယ်။ အဲဒီနှစ်စုကို ဖိုဒါနှစ်ခု ခွဲပြီး ImageAI moduel ရဲ့လိုအပ်ချက် အရ အောက်ပါအတိုင်း စီစဉ်ပါတယ်။
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
### ImageAI moduel ထည့်ခြင်း
notebook ရဲ့ ဆဲကွက်ထဲ မှာ သို့မဟုတ် command prompt အောက်ပါအတိုင်း ရိုက်ပြီး ခိုင်းလိုက်ရင် အင်တာနက်က နေ ရယူထည့်သွင်းသွားမှာပါ။ မိမိစက်မှာ ဆို တစ်ခါတည်းလုပ်ရမှာဖြစ်ပေမယ့် Google Colab မှာဆိုရင်တော့ အကြိမ်တိုင်း (ထွက်ပြီးပြန်ဝင်တိုင်း) install လုပ်ဖို့လိုမှာပါ။
```
!pip install imageai --upgrade
```
ImageAI ထည့်စဉ်မှာ ပါလာတဲ့ အဖော် module တွေပါ အသက်ဝင်လာဖို့ runtime ကို ပြန်စ (restart) ဖို့လိုပါလိမ့်မယ်။

### Model ကိုသင်ကြားပေးခြင်း
ဒီနေရာမှာ model ကိုသင်ပေးတယ်ဆိုတာ က တကယ်တော့ ကကြီးရေ က ကနေစသင်တာ တော့မဟုတ်ပါဘူး။ အခြားအရာဝတ္ထုတွေကို ခွဲခြားဖို့သင်ပေးထားပြီးသား model တစ်ခုကနေ သူသိပြီးသား အချက်အလက်တွေကို ရယူသုံးပြီး သင်တာပါ။ အဲဒီမော်ဒယ်က အုံးပင်တွေခွဲခြားဖို့ မသင်ခဲ့ဖူးပါဘူး။ ဒီလို ရှေ့က model ဆီက weigth တွေကို ယူသုံးပြီးသင်တာကို  လက်ဆင့်ကမ်းသင်ခြင်း (transfer learning) လို့ခေါ်ပါတယ်။ ဒီလိုသင်ခြင်းအားဖြင့် အချိန်နည်းနည်းအကြီမ်ရေနည်းနည်း နဲ့ ရလာဒ်ကောင်းကောင်း သင်ပေးနိုင်ပါတယ်။ အစကနေသာ စ သင်မယ်ဆိုရင် ပုံပေါင်းသောင်းနဲ့ချီလိုပြီး နာရီ ပေါင်း များစွာ ( ၁ - ၇ ရက်) ထိကြာပါတယ်။

ယခု လုပ်ဆောင်ပြချက်မှာ တော့ ImageAI ရဲ့  နေ့စဉ်တွေ့နေကြအရာဝတ္ထု ၈၀ အတွက် ကျင့်ထားတဲ့ model တစ်ခုကို လက်ဆင့်ကမ်းသင်ဖို့ ယူသုံးသွားမှာဖြစ်ပါတယ်။ 

သင်တာကို လုပ်တဲ့ code က (၅) ကြောင်းပဲရှိပါတယ်။ သင်ကြားရာမှာ ထိန်းချုပ် တဲ့ hyper-parameter လို့ခေါ်တဲ့အသေးစိတ် အချက်အလက်လေးတွေ အများကြီးရှိပါတယ်။ ImageAI မှာ batch_size (တစ်ခါသင် ဘယ်နှစ်ပုံ) နဲ့ အကြိမ်ရေ epochs ဘယ်လောက် (ပုံအကုန်လုံးပြီးမှ တစ်ကြိမ်လိုသတ်မှတ်ပါတယ်) ဆိုတာ ပဲ ထည့်ပေးရပါတယ် ကျန်တာ တွေက တော့ သူ့ဘာသာ နောက်ကွယ်မှာ ကြည်ကျက်လုပ်သွားတယ်လို့ထင်ပါတယ် :D ။ အခုမှစ လေ့လာမယ့်သူတွေအတွက် တော့ ခေါင်းသိပ်မစားရ တော့ဘူးပေါ့။

Google colab မှာ ကျွန်တော် သင်တာ တော့ အကြိမ် ၅၀ အတွက် ၄ နာရီလောက်ကြာခဲ့ပါတယ်။ 

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
သင်ပြီးရင်လည်း model က ဘယ်လောက်ကောင်းတယ်မကောင်းတယ်ဆိုတာ လည်းကြည့်ရပါသေးတယ်။ မကောင်းရင် ဘာတွေလုပ်ဖို့လိုမလဲ စဉ်းစားရပါတယ်။ training lableတွေမှားလို့လား၊ training imageနည်းလို့လား၊ batch_size တိုးမလားလျော့မလား၊ learning rate တိုးမလားလျော့မလား စသည်ဖြင့်၊ လိုအပ်ရင် epoch အကြိမ်ရေတိုး ပြီး သင်ရပါတယ်။ မိတ်ဆက်သဘော မို့ ဒါတွေကိုအသေးစိတ်ဒီမှာ ရေးမပြတော့ပါ။

### ကဲ... အုံးပင်တွေရှာပေးပါ
ရလာတဲ့ model ကို ကဲ ဒီ mosaic ထဲမှာ အုံးပင်ရှာ ပါလည်းဆိုရော ရှာပါတော့တယ် ဒါပေမယ့် ပြီးသာ သွားတယ် ဘာ အုံးပင်တစ်ခုမှ ရှာလို့မတွေ့ဘူး။အကြောင်းက တော့ မူလက သင်ပေးတုံးက ပုံလေးတွေက ၄၄၈ x ၄၄၈ အရွယ်လေးတွေ၊ အုံးပင် ရှာခိုင်းတဲ့ပုံက ၁၈၀၀၀ x ၂၅၀၀၀ အရွယ်ကြီးဖြစ်နေလို့ ဖြစ်ရတာပါ။ ဒါနဲ့ မူလပုံအကြီးကို အစိတ်လေးတွေစိတ်ပြီး အစိတ်တစ်ခုခြင်းထဲမှာ ရှာခိုင်းလိုက်တော့မှ အဆင်ပြေသွားပါတယ်။ တွေ့တဲ့ အုံးပင် တည်နေရာ lat-lon, အရွယ်အစား, ဖြစ်တန်စွမ်း, စတာ တွေကို csv ဖိုင်နဲ့ သိမ်းခိုင်းထားလိုက်ပါတယ်။ csv  ရရင် image ပေါ်မှာ point တွေတင်လို့ရပြီ‌။


```
image = "Kolovai UAV4R Subset.tif"
chip_h = 448
chip_w = 448
prob_threshold = 25
csv_name = "detection_report.csv"

detect(detector,image,chip_h,chip_w,prob_threshold,csv_name)
```
I would suggest to use low probabilty threshold values during detection so as not to miss the palm trees. Later you can filter the results in csv with your desire threshold.

```
            detection started: 2021-08-10 17:13:55.259247 

            number of object detected:  14292

            detection completed:  2021-08-10 17:21:27.751945 


            detection results are saved in detection_report.csv
```
၈မိနစ်လောက်အတွင်းမှာ ရှာလို့ပြီးသွားပါတယ်။ အုံးပင် တစ်သောင်းလေးထောင်ကျော် တွေ့တယ်လို့ သတင်းပို့တာ မြင်ရပါတယ်။



### အုံးပင်စာရင်း ကို လျပ်တစ်ပျက် စစ်ဆေးခြင်း
ထွက်လာတဲ့ csv ကို pandas နဲ့ လျပ်တစ်ပျက်ကြည့်လိုရပါတယ်။ အလျား၊အနံ၊ ပုံအချိုး နဲ့ ဧရိယာ တန်ဖိုးတွေက pixel နဲ့ပါ ။ ဒါတွေကိုသုံးပြီး မသေခြာတဲ့ အုံးပင် တွေကို ဖယ်လို့ရပါတယ်။

| ![][csv_view]   |
:-----------------:

### မြေပုံနဲ့ကြည့်မယ်
စာရင်းကြည့်ယုံနဲ့အားမရသေးရင် notebook ထဲမှာပဲ Folium နဲ့ မြေပုံလုပ်ပြီး တစ်ခါတည်းကြည့်လို့ရပါတယ်။ လေးမှာစိုးလို့ မူလ mosaic ကို အပြည့်မတင်ပဲ ဖြတ်ပြီးတင်လိုက်တယ်။

| ![][folium_map] |
:-----------------:

### သင်စိုက်လို့ရတဲ့ အသီးအပွင့်တွေကို ရိတ်သိမ်းယူသွားပါ
Google colab ကို ပိတ်လိုက်ရင် csv ဖိုင် နဲ့ model ပါအကုန် ရှင်းပစ်လိုက်မှာပါ။ ဒါကြောင့် csv ဖိုင် နဲ့ model ဖိုင်နဲ့ model definition json ဖိုင်တွေကို ဒေါင်းထားဖို့လိုပါလိမ့်မယ်။ model ဖိုင်နဲ့ model definition json ဖိုင်တွေသုံးပြီး မိမိစက်ပေါ်မှာ ဆက်လုပ် လို့ရပါတယ်။ ရှာ ဖို့ အတွက် GPU မရှိလည်းရပါတယ် နည်းနည်းလေးပဲ ပိုကြာပါတယ်။

### ကဲ အခုစမ်းသပ်ချင်လာပြီလား!

code တွေ training data နဲ့ mosaic image တွေပါတဲ့ notebook ကို ကျနော့် ![github repo link](https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/raw/main/Palm_tree_detection_on_large_aerial_imagery_with_yolov3_and_ImageAI_github_ver.ipynb) ကနေရယူနိုင်ပါတယ်။ သူ့ကို colab ကနေဖွင့်ပြီး စမ်းကြည့်လို့ရပါတယ်။ open in colab ကိုသာနှိပ်လိုက်ပါ။

### ကျေးဇူးတင်ထိုက်သူများ 
ImageAI ဖန်တီးမျှဝေသူ Moses Olafenwa  နဲ့  ESRI ကိုကျေးဇူးတင်ရှိပါတယ်။ သူတို့တွေမျှဝေပေးလို့သာ ယခုလို စမ်းသပ်ကြည့်နိုင်ခဲ့တာပါ။သူတို့တွေဆီက နေသာ သင်ယူခွင့်မရခဲ့ ရင် deep learning ဆိုတာကြီးကို တော်တော်နဲ့ ထိတွေ့ဖြစ်အုံးမှာမဟုတ်ပါဘူး

### ရေးသားမျှဝေသူ
```
ကျော်နိုင်ဝင်း
၂၀၂၁ ခုနှစ် ဩဂုတ်လ ၁၁ ရက်
```


[imageai_10]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_10lines.PNG
[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG
[view_images]:https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/view_images.PNG          
[labelImg]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/labelImg.png
[xml]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/xml.PNG
[csv_view]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/csv_view.PNG

[imageai_6]: https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/imageai_6lines.PNG
[folium_map]:https://github.com/knwin/Detect-palmtrees-with-Yolo-and-ImageAI/blob/main/images/folium_map.PNG

