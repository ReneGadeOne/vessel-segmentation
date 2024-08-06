import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from glob import glob
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

H = 256
W = 256

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.png")))
    y = sorted(glob(os.path.join(path, "mask", "*.png")))
    return x, y 

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = clahe(x,50)
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x

def normalize_pred(img):
    y_pred = model.predict(np.expand_dims(img, axis=0))[0] 
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = np.squeeze(y_pred)
    return y_pred

def strided_crop(img, img_h,img_w,height, width,stride=1):
    out_image = np.zeros((img_h, img_w),dtype=np.int32)
    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img_arr = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                pred_coarse = normalize_pred(crop_img_arr)
                out_image[h * stride:(h * stride) + height,w * stride:(w * stride) + width] = pred_coarse
                i = i + 1
    return  out_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=128,required=False)
    args = parser.parse_args() 

    """ Save the results in this folder """
    create_dir("results")

    """ Load the dataset """
    dataset_path = "Data/FullImages/test"
    test_x, test_y = load_data(dataset_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Load the model """
    model = tf.keras.models.load_model('model.h5',compile=False)
    model.compile()

    # Make the prediction and calculate the metrics values
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        # Extracting name 
        name = x.split("\\")[-1].split(".")[0]

        # Read the image and mask 
        ori_x, x = read_image(x)

        ori_y, y = read_mask(y)
        y_pred = strided_crop(x,x.shape[0], x.shape[1], 256, 256, stride= args.stride)

        # Saving the images 
        save_image_path = "results/" + name +".png"
        cv2.imwrite(save_image_path, y_pred* 255)

        y = y.flatten()
        y_pred = y_pred.flatten()

        # precision, recall, f1_score, iou_score, dice_score =  test_eval(y ,y_pred)  
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    SCORE = np.array(SCORE, dtype=object)
    print("acc_value: " + str(np.mean(SCORE[:, 1])))
    print("f1_value : " + str(np.mean(SCORE[:,2])))
    print("jac_value : " + str(np.mean(SCORE[:, 3])))
    print("recall_value : " + str(np.mean(SCORE[:, 4])))
    print("precision_value : " + str(np.mean(SCORE[:, 5])))

    df = pd.DataFrame(SCORE, columns=["Image", "acc_value", "f1_value", "jac_value", "recall_value","precision_value"])
    df.to_csv("files/test_score.csv")

    shutil.make_archive("results","zip" , "results")
