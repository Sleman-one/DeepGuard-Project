##
## this file is the source code of our model
##
## and you can see how we are going to unzip, load and process the dataset
## and how we are importing The TFViTFforImageClassification and The ViTImageProcessor
## The TFViTFforImageClassification is the structure for our model and we are going to train it on our dataset
## The ViTImageProcessor is our Image Processor and we are going to pass the dataset on it first before tringing
## and we need also to pass any Image we are going to predict in the future to our Processor first so the image can be similar to images in training
## then we pass it to our model to predict


# !pip install -q transformers - you need to run this line in you terminal if you are going to run it locally
# importing the the libraries that we are going to need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

# !wget "https://raw.githubusercontent.com/slemanone/ML/refs/heads/main/helper_functions.py" - this line will download a python script from my github that have functions that will help us
import helper_functions as hf

hf.unzip_data("/content/Dataset.zip")
train_dir = "/content/Train"
validation_dir = "/content/Validation"
test_dir = "/content/Test"

# here we are preparing the data using tf.keras.preprocessing.image_dataset_from_directory
# that will load the data for us, change the images size, batching the images and shuffling as needed
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                 image_size=(224, 224),
                                                                 label_mode="int",
                                                                 batch_size=32)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                                      image_size=(224, 224),
                                                                      label_mode="int",
                                                                      batch_size=32,
                                                                      shuffle=False)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                image_size=(224, 224),
                                                                label_mode="int",
                                                                batch_size=32,
                                                                shuffle=False)
classes_names = train_data.class_names
from transformers import TFViTForImageClassification, ViTImageProcessor

## here we are importing the model and the processor
model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",
                                                    num_labels=1)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


## our function that we will use to process our image
def image_preprocess(image, label):
    image = tf.cast(image, tf.uint8)

    def _preprocess_numpy(image_np):
        inputs = processor(image_np, return_tensors="np")
        return inputs["pixel_values"]

    pixel_values = tf.numpy_function(_preprocess_numpy, [image], tf.float32)
    pixel_values.set_shape((None, 3, 224, 224))
    return pixel_values, label


## passing our images to the processor
train_data = train_data.map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

## making the mixed_precision to mixed_float16
## it will make our training faster a little bit
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

## importing the create_optimizer from transformers
## and we are going to use it to create the optimizer and the schedule for our model
from transformers import create_optimizer

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_train_steps=len(train_data),
    num_warmup_steps=int(0.1 * len(train_data))
)

## compiling our model and setting the loss function to BinaryCrossentropy
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

## training the model
model_history = model.fit(train_data,  ## train images
                          steps_per_epoch=len(train_data),
                          ## telling our model how many image we are going to have each epoch in training
                          epochs=3,
                          validation_data=(validation_data),  ## validation images
                          validation_steps=len(validation_data),
                          ## telling our model how many image we are going to have each epoch in validation
                          verbose=1)

## saving our model and our processor so next time we need to use it we just import it
model.save_pretrained("drive/MyDrive/Tensorfow_Models/transformers_model_3.keras")
processor.save_pretrained("drive/MyDrive/Tensorfow_Models/transformers_model_3.keras")

## evaluating our model on the test_data
model.evaluate(test_data)

## evaluating the model manually
probs_preds = model.predict(test_data)
probs_preds = probs_preds['logits']
probs_preds = tf.sigmoid(probs_preds)
y_pred = tf.cast(probs_preds > 0.5, tf.int32)
y_true = []

for image, label in test_data.unbatch():
    y_true.append(label)
hf.calculate_results(y_true, y_pred)

## importing ConfusionMatrixDisplay from sklearn.metrics
## to make Confusion Matrix for our model evaluate
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_true,
                                        y_pred,
                                        display_labels=classes_names,
                                        cmap="Blues",
                                        xticks_rotation="vertical",
                                        # make it bigger
                                        ax=plt.figure(figsize=(5, 5), dpi=150).subplots())

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


## a function that will predict only one image and show the result
## it will take the model, processor and the image
## it will pass the image to the processor first then to the model to predict the result
## and finally it will plot the result with image
def predict_image(model, processor, image_path, threshold=0.5):
    """
    Predict class of a single image using a trained ViT model and display it.

    Args:
        model: Trained TFViTForImageClassification model.
        processor: ViTImageProcessor instance from HuggingFace.
        image_path: Path to the image file.
        threshold: Probability threshold (default = 0.5)

    Returns:
        Tuple of (probability, predicted_label)
    """

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="tf")

    if 'pixel_values' not in inputs:
        raise ValueError("The processor output does not contain 'pixel_values'.")

    output = model(inputs['pixel_values'])
    logits = output.logits
    prob = tf.sigmoid(logits)[0][0].numpy()
    label = int(prob > threshold)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Prediction: {label} (Prob: {prob:.2f})')
    plt.show()

    return prob, label


import transformers

## importing our trained model to use it
model = transformers.TFViTForImageClassification.from_pretrained(
    "/content/drive/MyDrive/Tensorfow_Models/transformers_model_3.keras")
processor = transformers.ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# !wget "https://th.bing.com/th/id/OIP.K164dIN9DXddlpIdPn1s4gAAAA?rs=1&pid=ImgDetMain" - this line will load the image from the web
## testing the model
predict_image(model=model, processor=processor,
              image_path="/content/OIP.K164dIN9DXddlpIdPn1s4gAAAA?rs=1&pid=ImgDetMain")
