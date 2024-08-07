import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from metrics import dice_loss, dice_coef, iou
from data import create_dir, load_data, shuffling, clahe_3d
from strided_crop import patching
import argparse
from models import build_unet, build_densenet121_unet, attention_unet



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = clahe_3d(x,50) 
    x = cv2.bilateralFilter(x, d=8, sigmaColor=50, sigmaSpace=50)
    # x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
    # x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)              
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_shape', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--augment', type=str2bool, default=False)
    parser.add_argument('--model', type=str, default='unet', required=False, choices=['unet' , 'attunet' , 'denseunet'])
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory to save files 
    create_dir("files")

    # Hyperparameters
    batch_size = args.bs
    lr = args.lr
    num_epochs = args.epoch
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")
    H = args.crop_shape
    W = args.crop_shape

    """ Dataset """
    data_path = "Dataset/train"
    data_x, data_y = load_data(data_path)
    data_x, data_y = shuffling(data_x, data_y)

    train_x, train_y = data_x[0:int(len(data_x)*0.85)], data_y[0:int(len(data_y)*0.85)]
    valid_x, valid_y = data_x[int(len(data_x)*0.85):], data_y[int(len(data_y)*0.85):]

    print(f"Train Images: {len(train_x)} - {len(train_y)}")
    print(f"Valid Images: {len(valid_x)} - {len(valid_y)}")

    patching(images_path=train_x, input_dim=H, stride=args.stride, data="train", augment=args.augment)
    patching(images_path=valid_x, input_dim=H, stride=args.stride, data="valid", augment=False)

    dataset_path = "Crops"
    train_x, train_y = load_data(os.path.join(dataset_path, "train"))
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(os.path.join(dataset_path, "valid"))

    print(f"Train Crops: {len(train_x)} - {len(train_y)}")
    print(f"Valid Crops: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size)

    train_steps = len(train_x)//batch_size
    valid_setps = len(valid_x)//batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(valid_x) % batch_size != 0:
        valid_setps += 1

    if args.model == "unet":
        model = build_unet((H, W, 3))
    if args.model == "attunet":
        model = attention_unet((H, W, 3))
    if args.model == "denseunet":
        model = build_densenet121_unet((H, W, 3))

    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])
    # model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_setps,
        callbacks=callbacks
    )

    # # LearningRateScheduler maybe can help!

   
