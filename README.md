# pneumonia_detector


Below in the url you can find the dataset that i used in this study.
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

If you want to try this out first thing you want to do is to download the dataset above.
Then change the directory of the dataset in train_model.py ( it is static). 
(Warning dataset is pretty large so you might want to reduce it to 300 images for each category)

python train_model.py -> will give you figure1, figure 2, complete.png and bestmodel.pt

Then we are gonna use this simple script to use our model. So in order to try this out you should type :
python test.py <path_to_image>

I made couple of test and the output is in the output results.png
.
