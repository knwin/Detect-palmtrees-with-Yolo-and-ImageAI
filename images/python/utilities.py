# to view the 6 training images with bounding boxes randomly
# written by Kyaw Naing Win
def view_images(image_folder,anno_folder): 
    import random
    import math
    import os
    import xml.etree.ElementTree as ET
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from PIL import Image
    
    sample_size = 6
    images = []
    formats = ['jpg','tif','png']
    for file in os.listdir(image_folder):
        if file[-3:].lower() in formats :
            images.append(file)
    
    sample_images = random.choices(images,k=sample_size)
    
    rows = math.ceil(sample_size/3)
    fig_h = rows * 5
    plt.figure(figsize=(14, fig_h))
    for i,img in enumerate(sample_images):
        image = image_folder+"/"+img
        xml = anno_folder+"/"+img[:-3]+"xml"
        image = Image.open(image)
        ax = plt.subplot(rows, 3, i + 1)    
        tree = ET.parse(xml)
        root = tree.getroot()
        for i in range (len(root)):
            if root[i].tag=="object":
                for j in range(len(root[i])):
                    if root[i][j].tag=="bndbox":
                        name = root[i][0].text
                        xmin = float(root[i][j][0].text) # xmin
                        ymin = float(root[i][j][1].text) # ymin
                        xmax = float(root[i][j][2].text) # xmax
                        ymax = float(root[i][j][3].text) # ymax
                        # Create a Rectangle patch (x0,y0),w,h
                        rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
        plt.imshow(image)
        plt.title(str(img))
        plt.axis("off")