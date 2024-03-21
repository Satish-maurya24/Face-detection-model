#!/usr/bin/env python
# coding: utf-8

# # 1.1 collecting image

# In[ ]:





# In[1]:


import os
import time
import uuid
import cv2


# In[3]:


get_ipython().system('pip install opencv-python')


# # 1.2 Annotation image with labelme

# In[7]:


IMAGE_PATH = os.path.join('data', 'images')
number_images = 50


# In[27]:


cap = cv2.VideoCapture(0)
for i in range (number_images):
    print ('collecting image_num{}'.format(i))
    ret , frame = cap.read()
    imgname = os.path.join(IMAGE_PATH , f'{str(uuid.uuid1())}.jpg')

    cv2.imwrite(imgname , frame)
    cv2.imshow('frame' , frame)
    time.sleep(0.6)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
                            


# In[28]:


get_ipython().system('labelme')


# In[10]:


pip install labelme


# # 2. Review dataset ans build image loading function

# ## 2.1 Import tensof=rflow and Deps

# In[2]:


import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


get_ipython().system(' pip install matplotlib')


# ## 2.2 limit GPU memory Growth

# In[3]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    


# In[4]:


tf.config.list_physical_devices('GPU')


# ## 2.3 load images into TF Data pipeline
# 

# In[46]:


images = tf.data.Dataset.list_files('data\\images\\*.jpg')


# In[30]:


images.as_numpy_iterator().next()


# In[5]:


def load_img(x):
    byte_img = tf.io.read_file(x)
    img  = tf.io.decode_jpeg(byte_img)
    return img


# In[48]:


images = images.map(load_img)


# In[33]:


images.as_numpy_iterator().next()


# In[20]:


type(images)


# ## 2.4 view raw images with matplotlib

# In[49]:


image_generator = images.batch(4).as_numpy_iterator()


# In[35]:


plot_images = image_generator.next()


# In[ ]:


fig , ax = plt.subplots(ncols=4 ,figsize = (20,20))
for indx , image in enumerate(plot_images):
    ax[indx].imshow(image)
plt.show()


#  # 3.Partition and unaugmented Data

# ## 3.1 mannualy split data into train test and val

# ## 3.2 move the matching labels

# In[37]:


import shutil


# In[38]:


for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data',folder , 'images')):
        
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels',filename)
     

        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data', folder, 'labels')
            shutil.move(existing_filepath, new_filepath)



# # 4. aplly image augmentation on images and labels using albumentations

# ## 4.1 setup albumentation transform pipleine

# In[6]:


import albumentations as alb


# In[29]:


get_ipython().system('pip install --user albumentations')


# In[7]:


augmentor = alb.Compose([alb.RandomCrop(width=450 , height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                         bbox_params=alb.BboxParams(format = 'albumentations',
                                                  label_fields=['class_labels'])
                        )


# ## 4.2 Load a test image and annotation with opencv and json
# 

# In[52]:


img = cv2.imread(os.path.join('data','train','images','0a4c1b9e-5dc2-11ee-a8f3-089798f1ad4e.jpg'))


# In[53]:


img


# In[54]:


with open(os.path.join('data','train','labels','0a4c1b9e-5dc2-11ee-a8f3-089798f1ad4e.json'),'r') as f:
         label = json.load(f)


# In[55]:


label['shapes'][0]['points']


# ## 4.3 Extract Coordinates and Rescale to Match Image Resolution

# In[56]:


coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]


# In[57]:


coords


# In[58]:


coords = list(np.divide(coords,[640 , 480 , 640 , 480]))


# In[59]:


coords


# ## 4.4 Apply Augmentation and View Results

# In[60]:


augmented = augmentor(image=img , bboxes=[coords], class_labels=['face'])


# In[61]:


augmented


# In[62]:


cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
               tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)),
               (255,0,0),2)
plt.imshow(augmented['image'])


# # 5 Build and Run Augmentation pipeline

# ## 5.1 Run augmentaion pipeline

# In[8]:


for partition in ['train','test','val']: 
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))

        try: 
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)


# ## 5.2 Load Augmented Images to tensoeflow Dataset

# In[6]:


train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_img)
train_images = train_images.map(lambda x:tf.image.resize(x,(120,120)))
train_images = train_images.map(lambda x: x/225)


# In[7]:


test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_img)
test_images = test_images.map(lambda x:tf.image.resize(x,(120,120)))
test_images = test_images.map(lambda x: x/225)


# In[8]:


val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_img)
val_images = val_images.map(lambda x:tf.image.resize(x,(120,120)))
val_images = val_images.map(lambda x: x/225)


# In[9]:


train_images.as_numpy_iterator().next()


# # 6 prepare labels 

# ## 6.1 build label loading function

# In[10]:


def laod_labels(label_path):
    with open(label_path.numpy() , 'r' , encoding="utf-8") as f:
        label = json.load(f)
    return [label['class']] , label['bbox']


# In[11]:


train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json' ,shuffle = False)
train_labels = train_labels.map(lambda x: tf.py_function(laod_labels, [x], [tf.uint8, tf.float16]))


# In[12]:


test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json' ,shuffle = False)
test_labels = test_labels.map(lambda x: tf.py_function(laod_labels, [x], [tf.uint8, tf.float16]))


# In[13]:


val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json' ,shuffle = False)
val_labels = val_labels.map(lambda x: tf.py_function(laod_labels, [x], [tf.uint8, tf.float16]))


# In[14]:


val_labels.as_numpy_iterator().next()


# # 7 Combine Label and image samples

# ## 7.1 check partition length

# In[15]:


train = tf.data.Dataset.zip((train_images , train_labels))
train = train.shuffle(4000)
train = train.batch(8)
train = train.prefetch(4)


# In[16]:


test = tf.data.Dataset.zip((test_images , test_labels))
test = test.shuffle(4000)
test = test.batch(8)
test = test.prefetch(4)


# In[17]:


val = tf.data.Dataset.zip((val_images , val_labels))
val = val.shuffle(4000)
val = val.batch(8)
val = val.prefetch(4)


# In[18]:


val.as_numpy_iterator().next()


# In[19]:


data_samples = train.as_numpy_iterator()


# In[20]:


res = data_samples.next()


# In[21]:


res


# In[ ]:





# In[22]:


fig, ax= plt.subplots(ncols = 4 , figsize=(20,20))
for idx in range(4):
    sample_img = res[0][idx]
    sample_coords  =res[1][1][idx]
    
    cv2.rectangle(sample_img , 
                  tuple(np.multiply(sample_coords[:2] ,[120,120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:] , [120,120]).astype(int)),
                       (255,0,0) ,1)
    ax[idx].imshow(sample_img)


# In[23]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input ,Conv2D,  Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16


# ## 8.2 Download vgg16

# In[24]:


vgg = VGG16(include_top=False)


# In[25]:


vgg.summary()


# ## 8.3 Build instance of Network

# In[26]:


def build_model():
    input_layer = Input(shape=(120,120,3))

    vgg = VGG16(include_top =False)(input_layer)
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048 , activation='relu')(f1)
    class2 = Dense(1,activation='sigmoid')(class1)

    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048 , activation='relu')(f2)
    regress2 = Dense(4,activation='sigmoid')(regress1)

    facetracker = Model(inputs =input_layer , outputs=[class2,regress2])
    return facetracker


# In[27]:


train.as_numpy_iterator().next()


# In[28]:


facetracker = build_model()


# In[29]:


facetracker.summary()


# In[30]:


x,y=train.as_numpy_iterator().next()


# In[31]:


x


# In[32]:


classes,coords = facetracker.predict(x)


# In[33]:


classes,coords


# In[34]:


y


# # 9 define Losses and Optimizer

# ## 9.1 define lr and Optimizer

# In[35]:


batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch


# In[36]:


len(train)


# In[37]:


opt= tf.keras.optimizers.Adam(learning_rate=0.0001,decay = lr_decay)


# ## 9.2 Create localization Loss and Classification Loss

# In[38]:


def localization_loss(y_true , yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2]-yhat[:,:2]))

    h_true = y_true[: , 3] -y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]

    h_pred= yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]

    delta_size = tf.reduce_sum(tf.square(w_true-w_pred)  + tf.square(h_true-h_pred))

    return delta_coord + delta_size


# In[39]:


classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss


# In[40]:


regressloss(y[1] , coords)


# In[41]:


classloss(y[0] , classes)


# # 10. Train Neural Network
# 

# In[42]:


class FaceTracker(Model):
    def __init__(self,eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker
        
    def compile(self , opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
        
    def train_step(self, batch, **kwargs):
        
        X, y = batch
        
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
            
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs):
        X, y =batch
        
        classses, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss=0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


# In[43]:


model = FaceTracker(facetracker)


# In[44]:


model.compile(opt, classloss, regressloss)


# ## 10.2 train 

# In[45]:


logdir='logs'


# In[46]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[47]:


hist = model.fit(train , epochs=40 , validation_data=val, callbacks=[tensorboard_callback])


# In[48]:


hist.history


# In[49]:


fig, ax = plt.subplots(ncols=3 , figsize=(20,5))
ax[0].plot(hist.history['total_loss'], color='teal', label= 'loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label= 'class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label= ' regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regress Loss')
ax[2].legend()

plt.show()





# # 11. Make predictions

# ## 11.1 make prediction on test set

# In[50]:


test_data = test.as_numpy_iterator()


# In[51]:


test_sample=test_data.next()


# In[52]:


yhat = facetracker.predict(test_sample[0])


# In[53]:


fig ,ax= plt.subplots(ncols =4 , figsize=(20,20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]
    
    if yhat[0][idx] > 0.5:
        cv2.rectangle(sample_image,
                     tuple(np.multiply(sample_coords[:2],[120,120]).astype(int)),
                     tuple(np.multiply(sample_coords[2:],[120,120]).astype(int)),
                           (255,0,0),2)
        
    ax[idx].imshow(sample_image)


# In[3]:


from tensorflow.keras.models import load_model


# In[1]:


import tensorflow as tf
import numpy as np
import cv2


# In[57]:


# facetracker.save('facetracker.h5')


# In[4]:


facetracker = load_model('facetracker.h5')


# In[ ]:





# In[5]:


cap =cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame= cap.read()
    frame = frame[50:500, 50:500, ]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5:
        cv2.rectangle(frame ,
                     tuple(np.multiply(sample_coords[:2],[450,450]).astype(int)),
                     tuple(np.multiply(sample_coords[2:],[450,450]).astype(int)),
                      (255,0,0),1)
        cv2.rectangle(frame,
                     tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:],[450,450]).astype(int)),
                      (255,0,0),1)
    
        cv2.putText(frame, 'face',tuple(np.add(np.multiply(sample_coords[:2],[450,450]).astype(int),
                                              [0,-5])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA
                   )
    cv2.imshow('EyeTrack' , frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

        
            


# In[76]:


get_ipython().system('python --version')


# In[ ]:


# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open camera.")
# else:
#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         frame = frame[50:500, 50:500]

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         resized = tf.image.resize(rgb, (120, 120))

#         yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
#         sample_coords = yhat[1][0]

#         if yhat[0] > 0.5:
#             cv2.rectangle(frame,
#                           tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
#                           tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
#                           (255, 0, 0), 2)

#             cv2.rectangle(frame,
#                           tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
#                           tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
#                           (255, 0, 0), 2)
#             cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
#                                                     [0, -5])),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow('EyeTrace', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
        
            


# In[1]:


import cv2

