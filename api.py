from email.mime import image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware




import tensorflow as tf
import cv2
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Conv2D, MaxPooling2D
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from uvicorn import run
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
import aiofiles
import glob
import requests





def loadImage(imagefile):
    #img_path = get_file(origin = imagefile)
    #img = load_img(img_path, target_size = (110, 220))
    


    #tempDir = "/Test/imgs"
    #Image.save(tempDir, 'imagefile')
    img_PIL = Image.open(imagefile)
    #img_PIL.save(f"{tempDir}/image.JPG")

    #return img



def create_model():
    model = Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    Conv2D(32, (3,3), activation='relu', input_shape=(110, 220, 3)
    ),
    MaxPooling2D(2, 2),
    # The second convolution
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    # The third convolution
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    # The fourth convolution
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    Flatten(),
    Dropout(0.5),
    # 512 neuron hidden layer
    Dense(512, activation='relu'),
    Dense(17, activation='softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def hazelnut_type():
    #loadImage(imageFile)
    model = create_model()

    # Crete the test generator
    batch_size = 100
    test_datagen2 = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    test_generator2 = test_datagen2.flow_from_directory(
            'Test',  # this is the target directory
            target_size=(110, 220),  # all images will be resized to 110x220
            batch_size=batch_size,
            class_mode='categorical')
    
    #intilize classes:


    train_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'classes',  # this is the target directory
            target_size=(110, 220),  # all images will be resized to 90x200
            batch_size=batch_size,
            class_mode='categorical') 

    


    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    #load the training model

    # this is the last model got during trainng with val_loss: 0.8449 - val_accuracy: 0.8450 
    #model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/hazelnut project/myModel.h5')

    #this is the best moddel got during train with  val_loss: 0.5371 - val_accuracy: 0.8350
    model.load_weights('best_model.h5')

    #reset the test
    test_generator2.reset()

    #make prediciton

    pred=model.predict_generator(test_generator2)

    print(pred)

    #find the max prediciton

    

    predicted_class_indices = np.argmax(pred,axis=1)

    #assign the labes to higher prediction

    predictions = [labels[k]
    for k in predicted_class_indices
    ]

    

    print(predictions)
    print(labels)
    print(predicted_class_indices)
    return predictions


app = FastAPI()
#predictor = hazelnut_type

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Hazelnut classifiction API!"}




@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...) ):

    
    print("filename = ", file.filename) # getting filename
    files = glob.glob('Test/imgs/*')
    for f in files:
        os.remove(f)
    destination_file_path = "Test/imgs/"+file.filename # location to store file
    async with aiofiles.open(destination_file_path, 'wb') as out_file:
        content = await file.read()
        #while content := await file.read(1024):  # async read file chunk
        await out_file.write(content)  # async write file chunk

    prediction = hazelnut_type()

    multiple_files = [('files', (destination_file_path, open(destination_file_path, 'rb'), 'image/jpg'))]

    text_data = {'type':prediction}
    headers = {
        # "Authorization" : "xxx",
        'accept': 'application/json',
        'Content-Type': 'application/json'
        }

    r = requests.post("http://localhost:7655/api/image/", files=multiple_files, data=text_data)

    return prediction
   

    




if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)




    

