#!/usr/bin/env python
# coding: utf-8

# # Chest X-Ray Medical Diagnosis with Deep Learning

# In[1]:


pwd 


# __Welcome to project of AI for Medical Diagnosis!__
# you will explore medical image diagnosis by building a state-of-the-art chest X-ray classifier using Keras. 
# 
# The assignment will walk through some of the steps of building and evaluating this deep learning classifier model. In particular, you will:
# - Pre-process and prepare a real-world X-ray dataset.
# - Use transfer learning to retrain a DenseNet model for X-ray image classification.
# - Learn a technique to handle class imbalance
# - Measure diagnostic performance by computing the AUC (Area Under the Curve) for the ROC (Receiver Operating Characteristic) curve.
# - Visualize model activity using GradCAMs.

# ## 1. Import Packages and Functions
# 
# We'll make use of the following packages:
# - `numpy` and `pandas` is what we'll use to manipulate our data
# - `matplotlib.pyplot` and `seaborn` will be used to produce plots for visualization
# - `util` will provide the locally defined utility functions that have been provided for this assignment
# 
# We will also use several modules from the `keras` framework for building deep learning models.
# 

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras import backend as k
import python_utils
from public_test import *
from test_utils import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ## 2. Load the Datasets

# we will be using the [ChestX-ray8 dataset](https://arxiv.org/abs/1705.02315) which contains 108,948 frontal-view X-ray images of 32,717 unique patients. 
# - Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions. 
# - These in turn can be used by physicians to diagnose 8 different diseases. 
# - We will use this data to develop a single model that will provide binary classification predictions for each of the 14 labeled pathologies. 
# - In other words it will predict 'positive' or 'negative' for each of the pathologies.
#  
# You can download the entire dataset for free [here](https://nihcc.app.box.com/v/ChestXray-NIHCC). 
# - We have provided a ~1000 image subset of the images for you.
# - These can be accessed in the folder path stored in the `IMAGE_DIR` variable.
# 
# The dataset includes a CSV file that provides the labels for each X-ray.

# In[48]:


cd F:\Artifitial Intelligence\Machine Learning for Healthcare\projects\Files\Files\home\jovyan\work\data\nih


# In[54]:


train_df = pd.read_csv("train-small.csv")
valid_df = pd.read_csv("valid-small.csv")

test_df = pd.read_csv("test.csv")
print(f'there are {train_df.shape[0]} rows and {train_df.shape[1]} columns in this dataframe')

train_df.head()


# In[55]:


labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']


# In[51]:


train_df.loc[:, labels] = train_df.loc[:, labels].astype(float)


# ### 2.2 Preventing Data Leakage
#  t is worth noting that our dataset contains multiple images for each patient. This could be the case, for example, when a patient has taken multiple X-ray images at different times during their hospital visits. In our data splitting, we have ensured that the split is done on the patient level so that there is no data "leakage" between the train, validation, and test datasets.

# In[56]:


def check_for_leakage(df1,df2,patient_col):
    df1_patients_unique=set(df1[patient_col].values)
    df2_patients_unique=set(df2[patient_col].values)
    patients_in_both_groups=list(df1_patients_unique.intersection(df2_patients_unique))
    leakage=len(patients_in_both_groups)>0
    return leakage


# In[57]:


print("leakage between train and valid:{}".format(check_for_leakage(train_df,valid_df,'PatientId')))
print("leakage between train and test:{}".format(check_for_leakage(train_df,test_df,'PatientId')))
print("leakage between valid and test:{}".format(check_for_leakage(valid_df,test_df,'PatientId')))


# ### 2.3 Preparing Images

# With our dataset splits ready, we can now proceed with setting up our model to consume them. 
# - For this we will use the off-the-shelf [ImageDataGenerator](https://keras.io/preprocessing/image/) class from the Keras framework, which allows us to build a "generator" for images specified in a dataframe. 
# - This class also provides support for basic data augmentation such as random horizontal flipping of images.
# - We also use the generator to transform the values in each batch so that their mean is $0$ and their standard deviation is 1. 
#     - This will facilitate model training by standardizing the input distribution. 
# - The generator also converts our single channel X-ray images (gray-scale) to a three-channel format by repeating the values in the image across all channels.
#     - We will want this because the pre-trained model that we'll use requires three-channel inputs.
# 
# Since it is mainly a matter of reading and understanding Keras documentation, we have implemented the generator for you. There are a few things to note: 
# 1. We normalize the mean and standard deviation of the data
# 3. We shuffle the input after each epoch.
# 4. We set the image size to be 320px by 320px

# In[58]:


def get_train_generator(df,image_dir,x_col,y_cols,shuffle=True,batch_size=8,seed=1,target_w=320,target_h=320):
    print("getting train generator...")
    image_generator=ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
    
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",       #  Mode for yielding the targets, one of "binary", "categorical", "input", "multi_output", "raw", sparse" or None. Default: "categorical".
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    return generator


# #### Build a separate generator for valid and test sets
# 
# Now we need to build a new generator for validation and testing data. 
# 
# **Why can't we use the same generator as for the training data?**
# 
# Look back at the generator we wrote for the training data. 
# - It normalizes each image **per batch**, meaning that it uses batch statistics. 
# - We should not do this with the test and validation data, since in a real life scenario we don't process incoming images a batch at a time (we process one image at a time). 
# - Knowing the average per batch of test data would effectively give our model an advantage.  
#     - The model should not have any information about the test data.
# 
# What we need to do is normalize incoming test data using the statistics **computed from the training set**. 
# * We implement this in the function below. 
# * There is one technical note. Ideally, we would want to compute our sample mean and standard deviation using the entire training set. 
# * However, since this is extremely large, that would be very time consuming. 
# * In the interest of time, we'll take a random sample of the dataset and calcualte the sample mean and sample standard deviation.

# In[59]:


def get_test_and_valid_generator(valid_df,test_df,train_df,image_dir,x_col,y_cols,sample_size=100,batch_size=8,seed=1,target_w = 320, target_h = 320):
    
    print("getting train and valid generators...")
    raw_train_generator=ImageDataGenerator().flow_from_dataframe(dataframe=train_df,directory=IMAGE_DIR,x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]
    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)
    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# In[60]:


IMAGE_DIR=r"F:\Artifitial Intelligence\Machine Learning for Healthcare\projects\Files\Files\home\jovyan\work\data\nih\images-small"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# In[61]:


x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);


# ## 3. Model Development
# 
# Now we'll move on to model training and development. 
# We have a few practical challenges to deal with before actually training a neural network, though. The first is class imbalance.

# ### 3.1 Addressing Class Imbalance
# One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets. Let's plot the frequency of each of the labels in our dataset:

# In[62]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# We can see from this plot that the prevalance of positive cases varies significantly across the different pathologies. (These trends mirror the ones in the full dataset as well.) 
# * The `Hernia` pathology has the greatest imbalance with the proportion of positive training cases being about 0.2%. 
# * But even the `Infiltration` pathology, which has the least amount of imbalance, has only 17.5% of the training cases labelled positive.
# 
# Ideally, we would train our model using an evenly balanced dataset so that the positive and negative training cases would contribute equally to the loss. 
# 
# If we use a normal cross-entropy loss function with a highly unbalanced dataset, as we are seeing here, then the algorithm will be incentivized to prioritize the majority class (i.e negative in our case), since it contributes more to the loss. 

# In[64]:


def compute_class_freqs(labels):
    N=labels.shape[0]
    positive_frequencies=np.sum(labels,axis=0)/N
    negative_frequencies=(N-np.sum(labels,axis=0))/N
    return positive_frequencies, negative_frequencies


# In[65]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos


# In[15]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


# In[66]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# applying these weightings the positive and negative labels within each class would have the same aggregate contribution to the loss function. Now let's implement such a loss function.

# In[67]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);


# #### Weighted Loss

# In[68]:


def get_weighted_loss(pos_weights,neg_weights,epsilon=1e-7):
    def weighted_loss(y_true,y_pred):
        loss=0.0
        for i in range(len(pos_weights)):
            loss+= -pos_weights[i]*k.mean(y_true[:,i]*k.log(y_pred[:,i]+epsilon))                   -neg_weights[i]*k.mean(1-y_true[:,i]*k.log(1-y_pred[:,i]+epsilon))
        return loss
    return weighted_loss


# ### DenseNet121
# 
# Next, we will use a pre-trained [DenseNet121](https://www.kaggle.com/pytorch/densenet121) model which we can load directly from Keras and then add two layers on top of it:
# 1. A `GlobalAveragePooling2D` layer to get the average of the last convolution layers from DenseNet121.
# 2. A `Dense` layer with `sigmoid` activation to get the prediction logits for each of our classes.
# 
# We can set our custom loss function for the model by specifying the `loss` parameter in the `compile()` function.

# In[69]:


weights=r"F:\Artifitial Intelligence\Machine Learning for Healthcare\projects\Files\Files\home\jovyan\work\models\nih\densenet.hdf5"


# In[70]:


def load_C3M3_model():
    class_pos=train_df.loc[:,labels].sum(axis=0)
    class_neg=len(train_df)-class_pos
    class_total=class_pos+class_neg
    pos_weights=class_pos/class_total
    neg_weights=class_neg/class_total
    print("Got loss weights")


    # create the base pre-trained model
    base_model = DenseNet121(weights=weights, include_top=False)
    print("Loaded DenseNet")
    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    print("Added layers")
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
    print("Compiled Model")
    model.load_weights(r"F:\Artifitial Intelligence\Machine Learning for Healthcare\projects\Files\Files\home\jovyan\work\models\nih\pretrained_model.h5")
    print("Loaded Weights")
    return model


# In[71]:


model = load_C3M3_model()


# In[72]:


model.summary()


# In[73]:


# Print out the total number of layers
layers=model.layers
print('total number of layers =',len(layers))


# In[74]:


# The find() method returns an integer value:
# If substring doesn't exist inside the string, it returns -1, otherwise returns first occurence index
conv2D_layers = [layer for layer in model.layers 
                if str(type(layer)).find('Conv2D') > -1]


# In[75]:


print('Model input -------------->', model.input)
print('Feature extractor output ->', model.get_layer('conv5_block16_concat').output)
print('Model output ------------->', model.output)


# ## 4. Training

# With our model ready for training, we will use the `model.fit()` function in Keras to train our model. 
# - We are training on a small subset of the dataset (~1%).  
# - So what we care about at this point is to make sure that the loss on the training set is decreasing.
# 
# Since training can take a considerable time, for pedagogical purposes we have chosen not to train the model here but rather to load a set of pre-trained weights in the next section. However, you can use the code shown below to practice training the model locally on your machine or in Colab.
# 
# **NOTE:** Do not run the code below on the Coursera platform as it will exceed the platform's memory limitations.
# 
# Python Code for training the model:
# 
# ```python
# history = model.fit_generator(train_generator, 
#                               validation_data=valid_generator,
#                               steps_per_epoch=100, 
#                               validation_steps=25, 
#                               epochs = 3)
# 
# plt.plot(history.history['loss'])
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.title("Training Loss Curve")
# plt.show()
# ```

# In[26]:


history = model.fit_generator(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 1)
plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()


# ### 4.1 Training on the Larger Dataset
# 
# Given that the original dataset is 40GB+ in size and the training process on the full dataset takes a few hours, we have trained the model on a GPU-equipped machine for you and provided the weights file from our model.
# The model architecture for our pre-trained model is exactly the same, but we used a few useful Keras "callbacks" for this training. Do spend time to read about these callbacks at your leisure as they will be very useful for managing long-running training sessions:

# In[76]:


model.load_weights(r"F:\Artifitial Intelligence\Machine Learning for Healthcare\projects\Files\Files\home\jovyan\work\models\nih\pretrained_model.h5")


# ## 5. Prediction and Evaluation

# Now that we have a model, let's evaluate it using our test set. We can conveniently use the predict_generator function to generate the predictions for the images in our test set.

# In[77]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))


# ### 6. ROC Curve and AUROC

# Compute metric called the AUC (Area Under the Curve) from the ROC (Receiver Operating Characteristic) curve. ideally we want a curve that is more to the left so that the top has more "area" under it, which indicates that the model is performing better.

# In[78]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.savefig('ROC.png')
    plt.show()
    return auc_roc_vals


# In[30]:


auc_rocs = get_roc_curve(labels, predicted_vals, test_generator)


# For reference, here's the AUC figure from the ChexNeXt paper which includes AUC values for their model as well as radiologists on this dataset:
# 
# <img src="https://journals.plos.org/plosmedicine/article/figure/image?size=large&id=10.1371/journal.pmed.1002686.t001" width="80%">
# 
# This method does take advantage of a few other tricks such as self-training and ensembling as well, which can give a significant boost to the performance.

# In[79]:


print("areas under the curve : {} \n for all {} classes".format(auc_rocs,len(auc_rocs)))


# ### Interpreting Deep Learning Models
# Let's load in an X-ray image.
# 
# 

# In[83]:


from keras.preprocessing import image


# In[111]:


IMAGE_DIR="F:/Artifitial Intelligence/Machine Learning for Healthcare/projects/Files/Files/home/jovyan/work/data/nih/images-small/"


# In[119]:


sns.reset_defaults()
def get_mean_std_per_batch(df,w=320,h=320):
    sample_data=[]
    for idx,img in enumerate(df.sample(100)["Image"].values):
        path=IMAGE_DIR+img
        sample_data.append(np.array(image.load_img(path,target_size=(w,h))))
    mean=np.mean(sample_data[0])
    std=np.std(sample_data[0])
    return mean,std
def load_image_normalize(path,mean,std,w=320,h=320):
    x=image.load_img(path,target_size=(h,w))
    x-=mean
    x/=std
    x=np.expand_dims(x,axis=0)
    return x
def load_image(path,df,preprocess=True,h=320,w=320):
    x=image.load_img(path,target_size=(h,w))
    if preprocess:
        mean, std = get_mean_std_per_batch(df, h=h, w=w)
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x
im_path = IMAGE_DIR+ "00025288_001.png"
x = load_image(im_path, train_df, preprocess=False)
plt.imshow(x, cmap = 'gray')
plt.show()


# Next, let's get our predictions. Before we plug the image into our model, we have to normalize it. Run the next cell to compute the mean and standard deviation of the images in our training set.

# In[120]:


mean,std=get_mean_std_per_batch(train_df)


# Now we are ready to normalize and run the image through our model to get predictions.

# In[121]:


labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
preprocessed_img=load_image_normalize(im_path,mean,std)
preds=model.predict(preprocessed_img)
pred_df=pd.DataFrame(preds,columns=labels)
pred_df.loc[0, :].plot.bar()
plt.title("Predictions")
plt.savefig('predictions.png')
plt.show()


# In[122]:


pred_df


# We see, for example, that the model predicts Mass (abnormal spot or area in the lungs that are more than 3 centimeters) with high probability. Indeed, this patient was diagnosed with mass. However, we don't know where the model is looking when it's making its own diagnosis. To gain more insight into what the model is looking at, we can use GradCAMs.

# ### GradCAM

# In[123]:


spatial_maps=model.get_layer('conv5_block16_concat').output
spatial_maps


# In[125]:


get_spatial_maps = k.function([model.input], [spatial_maps])
print(get_spatial_maps)


# In[126]:


# get an image
x = load_image_normalize(im_path, mean, std)
print(f"x is of type {type(x)}")
print(f"x is of shape {x.shape}")


# In[127]:


# get the 0th item in the list
spatial_maps_x = get_spatial_maps([x])[0]
print(f"spatial_maps_x is of type {type(spatial_maps_x)}")
print(f"spatial_maps_x is of shape {spatial_maps_x.shape}")
print(f"spatial_maps_x without the batch dimension has shape {spatial_maps_x[0].shape}")


# #### Getting Gradients
# The other major step in computing GradCAMs is getting gradients with respect to the output for a particular class.

# In[128]:


# get the output of the model
output_with_batch_dim = model.output
print(f"Model output includes batch dimension, has shape {output_with_batch_dim.shape}")
print(f"excluding the batch dimension, the output for all 14 categories of disease has shape {output_with_batch_dim[0].shape}")


# In[129]:


# Get the first category's output (Cardiomegaly) at index 0
y_category_0 = output_with_batch_dim[0][0]
print(f"The Cardiomegaly output is at index 0, and has shape {y_category_0.shape}")


# In[ ]:





# In[ ]:


df = pd.read_csv("train-small.csv")
IMAGE_DIR = r"F:\Artifitial Intelligence\Machine Learning for Healthcare\projects\Files\Files\home\jovyan\work\models\nih\pretrained_model.h5"

# only show the labels with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]


# In[ ]:


util.compute_gradcam(model, '00008270_015.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:




