##This Markdown script is essentially a Python script that performs various image processing tasks using popular libraries like
##numpy, matplotlib, requests, PIL, and cv2.
##Here is a breakdown of what each section does:


##[2]:** Importing necessary libraries for the image processing 

import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import cv2

##**[3]:** Fetching an image from a URL using the requests 
#library and converting it into an RGB image using the PIL library.

URL = 'https://3.bp.blogspot.com/-DG4rqFtAJ4c/T-8mUvVo6yI/AAAAAAAADKc/eSxKhutGC_Q/s1600/green+glades+017.JPG'
response = requests.get(URL, stream=True)
img = Image.open(response.raw).convert("RGB")

##**[4]:** Displaying the original image using matplotlib.
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()

##**[5]:** Converting the image into a numpy array and printing its shape.

img_array = np.array(img)
print("Shape:", img_array.shape)

##*[6]:** Resizing the image to a specific size using the cv2 library.
resized_image = cv2.resize(img_array, (224, 224))

##**[7]:** Displaying the resized image using matplotlib.
plt.imshow(resized_image)
plt.title("Resized Image")
plt.axis('off')
plt.show()

##**[8]:** Converting the resized image to grayscale using the cv2 library.

##* Printing the shape of the grayscale image.

grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
print("Grayscale Shape:", grayscale_image.shape)

##* Printing the shape of the grayscale image.

## Displaying the grayscale image using matplotlib.
plt.imshow(grayscale_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

##**[10]:** Creating 10 random filters and applying them to the grayscale image to generate 10 feature maps. 
##The filters are displayed alongside their corresponding feature maps using matplotlib.

filter_size = 3
filters = [np.random.randn(filter_size, filter_size) for _ in range(10)]

fig, axs = plt.subplots(10, 2, figsize=(10, 30))
for i, filt in enumerate(filters):
    feature_map = cv2.filter2D(grayscale_image, -1, filt)

    axs[i, 0].imshow(filt, cmap='gray')
    axs[i, 0].set_title(f"Filter {i + 1}")
    axs[i, 0].axis('off')

    axs[i, 1].imshow(feature_map, cmap='gray')
    axs[i, 1].set_title(f"Feature Map {i + 1}")
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

##In summary, the script fetches an image from a URL,
##processes it by resizing and converting it to grayscale, 
##and then applies a set of randomly generated filters to create feature maps. 
##The resulting images are displayed for visualization.

###code 
view code [here](https://colab.research.google.com/drive/1PvbuQr4TIVPRV_VeN25je_OwQJ7jiNOZ)
