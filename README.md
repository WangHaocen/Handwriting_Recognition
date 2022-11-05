# IP6 handwriting recognition
## Program:

Tensorflow==2.8.0

python==3.8.0

cv2==4.5.5

An additional Library deslant_img: https://github.com/githubharald/DeslantImg => need to be clone in a folder and pip install . to install this Library


## Project Inspiration:

For the online model I took the inspiration from this project: https://github.com/chunkyjasper/IAMhwr
I took the idea and implemtented almost everything by myself again. Only the Trie Beam Search (Although not work properly) and the Application that I used it directly. With the Application I added the offline model with it to predict the offline image, too.

For the offline model I took the inspiration of keras example https://keras.io/examples/vision/handwriting_recognition/  
I used the idea of the Reshape Size and added an extra deslant_img for the preprocessing. I also took the idea to change the image and label to tensor, but still I implemented by myself.


## Application with Video:

I have made a video for the application, so you can see how it works. There are 8 prediction results. The first 4 results for the Online Model and the last 4 for the Offline Model. 

## Attention!!! The weigts from the offline model did not work properly. 

When the kernel restarts and load the weights from the file, the model predicts like an untrained model, which will give an complete wrong result. Only if you run the offline.ipynb file from very beginning and trained the model from the start, which will take some time (for me the case, I ended 15min/epoch in the end). I made the video to show the application in case you don't want to train the offline model and then using the application.
(The place of the mouse did not pass with the application because I moved the application position while I was recording)






