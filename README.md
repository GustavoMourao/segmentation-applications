# segmentation-applications
Set of codes related to evaluation of Senai segmentation tasks

## Source structure
```
.
├── data/
│   ├── test/
│   │   └── ...
│   ├── train/
│   │   └── ...
├── models/
│   └── ...
├── results/
│   └── ...
├── results/
│   └── ...
├── uploads/
│   └── ...
├── training_workflows
│   └── workflow_1.ipynb
│   └── workflow_2.ipynb
├── app_anm.py
├── app_tags.py
```

- `data` contains the dataset divided into test and train. inside the each folder add the .json file associated with labels

- `models` contains the trained models. Download address:
https://drive.google.com/drive/folders/1xh2msCHGk4DraHzJLYtzBNLOw6U_hc8F?usp=sharing

- `results` storage the responses of inference tasks from image collected from frontend

- `uploads` contains the image uploads from frontend

- `training_workflows` storage the notebooks responsible by train the model (resnet101). 
   - It have used transfer learning based on mask_rcnn_coco.h5 (keras framework)
   - Batch size and epochs were 2 and 50 respectively
   - For both workflows it were used data augmentation due the lack of dataset:
      - pixellib.custom_train.train_model
   - For those models it was used mAP metric (area under the precision-recall curve). The results were around 0.8.
   - In both models the validation loss were higher than training loss: **overfit**
      - It should regulirize them
        - reduce the models size
        - L2 normalization (it has been implemented, however it would be interesting to change some parameters)
        - Droput
        - Increase dataset (even though I have used data augmentation)

- `app_anm.py` executes the animals detection application

- `app_tags.py` executes the tags detection application

### Run using python

```
1. virtualenv venv
2. source venv\bin\activate
3. pip install -r requirements.txt

4.1. python app_tags.py (for tag application)

4.2. python app_anm.py (for animals application)
```

### Run using docker
1. Build
docker build -t app_tags .

3. Run
docker run -it --rm -p 5002:5002 app_tags

To run the animals detection application edit Docker configuration by adding into `CMD [ "python" , "app_tags.py"]` the script `app_anm.py`.

### References:

[1] Bonlime, Keras implementation of Deeplab v3+ with pretrained weights https://github.com/bonlime/keras-deeplab-v3-plus

[2] Liang-Chieh Chen. et al, Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation https://arxiv.org/abs/1802.02611

[3] Matterport, Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow https://github.com/matterport/Mask_RCNN

[4] Mask R-CNN code made compatible with tensorflow 2.0, https://github.com/tomgross/Mask_RCNN/tree/tensorflow-2.0

[5] Kaiming He et al, Mask R-CNN https://arxiv.org/abs/1703.06870

[6] TensorFlow DeepLab Model Zoo https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

[7] Pascalvoc and Ade20k datasets' colormaps https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py

[8] Object-Detection-Python https://github.com/Yunus0or1/Object-Detection-Python

[9] https://github.com/mtobeiyf/keras-flask-deploy-webapp

[10] https://github.com/ayoolaolafenwa/PixelLib
