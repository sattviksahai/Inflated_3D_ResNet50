import numpy as np
from keras.models import Model
from keras.layers import Input, Add, Activation, Dense, Conv3D, ZeroPadding3D, MaxPooling3D, BatchNormalization, AveragePooling3D, Flatten
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50

# define a function to add bottleneck
def add_bottle_neck(x, stride_val, filters_1, filters_2, regress_indentity):
    x_identity = x
    x_layer = Conv3D(filters_1, kernel_size=(1, 1, 1), padding='valid', strides=stride_val)(x)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    x_layer = Activation(activation='relu')(x_layer)
    x_layer = Conv3D(filters_1, kernel_size=(3, 3, 1), padding='same', strides=(1, 1, 1))(x_layer)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    x_layer = Activation(activation='relu')(x_layer)
    x_layer = Conv3D(filters_2, kernel_size=(1, 1, 1), padding='valid', strides=(1, 1, 1))(x_layer)
    x_layer = BatchNormalization(axis=-1)(x_layer)
    if (regress_indentity == True):
        shortcut_path = Conv3D(filters_2, kernel_size=(1, 1, 1), padding='valid', strides=stride_val)(x_identity)
        shortcut_path = BatchNormalization(axis=-1)(shortcut_path)
        final_layer = Add()([x_layer, shortcut_path])
    else:
        final_layer = Add()([x_layer, x_identity])
    
    final_layer = Activation(activation='relu')(final_layer)
    return final_layer


# define model input
input_x = Input(shape=(224, 224, 8, 3))

# define 3d ResNet50
X = ZeroPadding3D((3, 3, 0))(input_x)
X = Conv3D(64, kernel_size=(7, 7, 1), padding='valid', strides=(2, 2, 1))(X)
X = BatchNormalization(axis=-1)(X)
X = Activation(activation='relu')(X)
X = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1),padding='valid')(X)
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=64, filters_2=256, regress_indentity=True) # conv2_1
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=64, filters_2=256, regress_indentity=False) # conv2_2
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=64, filters_2=256, regress_indentity=False) # conv2_3
X = add_bottle_neck(X, stride_val=(2, 2, 1), filters_1=128, filters_2=512, regress_indentity=True) # conv3_1
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=128, filters_2=512, regress_indentity=False) # conv3_2
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=128, filters_2=512, regress_indentity=False) # conv3_3
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=128, filters_2=512, regress_indentity=False) # conv3_4
X = add_bottle_neck(X, stride_val=(2, 2, 1), filters_1=256, filters_2=1024, regress_indentity=True) # conv4_1
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=256, filters_2=1024, regress_indentity=False) # conv4_2
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=256, filters_2=1024, regress_indentity=False) # conv4_3
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=256, filters_2=1024, regress_indentity=False) # conv4_4
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=256, filters_2=1024, regress_indentity=False) # conv4_5
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=256, filters_2=1024, regress_indentity=False) # conv4_6
X = add_bottle_neck(X, stride_val=(2, 2, 1), filters_1=512, filters_2=2048, regress_indentity=True) # conv5_1
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=512, filters_2=2048, regress_indentity=False) # conv5_2
X = add_bottle_neck(X, stride_val=(1, 1, 1), filters_1=512, filters_2=2048, regress_indentity=False) # conv5_3
X = AveragePooling3D(pool_size=(7,7,8), strides=None, padding='valid')(X) # Modify for glimpse clouds
X = Flatten()(X)
X = Dense(1000, activation='softmax')(X)

# Create model
resnet50_3d_model = Model(inputs = input_x, outputs = X, name='ResNet50_3D')

# model summary and plot
print(resnet50_3d_model.summary())
plot_model(resnet50_3d_model, to_file='model.png')

# Load pretrained weights for a 2D resnet trained on imagenet

model = ResNet50(weights='imagenet')
print("Model Loaded")

# read weights
weights_2d = model.get_weights()

# Declare array containing weight inxedes which have to be expanded
index_array = [0, 6, 12, 18, 20, 30, 36, 42, 48, 54, 60, 66, 72, 78, 80, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 158, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 272, 282, 288, 294, 300, 306, 312]

weights_3d = []
for i in range(0, len(weights_2d)):
    if i in index_array:
        #N = weights_2d[i].shape[1]
        N = 1
        weights_3d.append(np.repeat(weights_2d[i][:,:, np.newaxis,:,:], N, axis=2))
        # Normalize weights for H, W, and D to keep activation values same as the 2D network
        weights_3d[i][:, :, :, 1, 1] /= (N+1)
    else:
        weights_3d.append(weights_2d[i])
    print(weights_3d[i].shape)

print("Weight dimensions inflated")

# Load weights into the model
resnet50_3d_model.set_weights(weights_3d)
print("weights loaded into the model")