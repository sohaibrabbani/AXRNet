import importlib
import keras
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model


class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                  weights_path=None, input_shape=None, for_test=False):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self.models_[model_name]['module_name']}"
            ),
            model_name)

        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        # predictions = Dense(len(class_names)+9, activation="sigmoid", name="predictions")(x)
        predictions = Dense(14, activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)

        if not for_test:
            if weights_path == "":
                weights_path = None

            if weights_path is not None:
                print(f"load model weights_path: {weights_path}")
                model.load_weights(weights_path)

        print(model.summary())
        # if for_test:
        for layer in model.layers[:-7]:
            layer.trainable = False

        model.layers.pop()
        model.layers.pop()

        inputs = Input(shape=input_shape, name="attention_map")

        x = keras.layers.Conv2D(32, kernel_size=(3, 3))(inputs)
        # x = keras.layers.Activation('relu')(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3))(x)

        x = keras.layers.Conv2D(64, kernel_size=(3, 3))(x)
        # x = keras.layers.Activation('relu')(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3))(x)

        x = keras.layers.Conv2D(1024, kernel_size=(3, 3))(x)
        # x = keras.layers.Activation('relu')(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3))(x)

        x = keras.layers.Add()([model.layers[-1].output, x])
        # x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Activation('relu')(x)
        # # x = keras.layers.LeakyReLU()(x)
        # # x = keras.layers.ZeroPadding2D((1, 1))(x)
        # x = keras.layers.Conv2D(1024, kernel_size=(3, 3), use_bias=False, kernel_initializer='he_normal')(x)
        # x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Activation('relu')(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024)(x)
        # x = keras.layers.Activation('relu')(x)
        x = keras.layers.LeakyReLU()(x)
        # x = keras.layers.Dense(1024, activation='sigmoid')(x)  # Num Classes for CIFAR-10
        outputs = keras.layers.Dense(len(class_names), activation='sigmoid')(x)  # Num Classes for CIFAR-10
        # outputs = keras.layers.Activation('sigmoid')(x)

        model = keras.models.Model(inputs=[model.input, inputs], outputs=[outputs])

        if for_test:
            if weights_path == "":
                weights_path = None

            if weights_path is not None:
                print(f"load model weights_path: {weights_path}")
                model.load_weights(weights_path)
        return model

    # def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
    #               weights_path=None, input_shape=None, for_test=False):
    #
    #     if use_base_weights is True:
    #         base_weights = "imagenet"
    #     else:
    #         base_weights = None
    #
    #     base_model_class = getattr(
    #         importlib.import_module(
    #             f"keras.applications.{self.models_[model_name]['module_name']}"
    #         ),
    #         model_name)
    #
    #     if input_shape is None:
    #         input_shape = self.models_[model_name]["input_shape"]
    #
    #     img_input = Input(shape=input_shape)
    #
    #     base_model = base_model_class(
    #         include_top=False,
    #         input_tensor=img_input,
    #         input_shape=input_shape,
    #         weights=base_weights,
    #         pooling="avg")
    #     x = base_model.output
    #     predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    #     model = Model(inputs=img_input, outputs=predictions)
    #
    #     if for_test:
    #         for layer in model.layers[:-5]:
    #             layer.trainable = False
    #
    #         model.layers.pop()
    #         model.layers.pop()
    #
    #         inputs = Input(shape=input_shape, name="attention_map")
    #
    #         x = keras.layers.Conv2D(32, kernel_size=(3, 3))(inputs)
    #         x = keras.layers.BatchNormalization()(x)
    #         # x = keras.layers.Activation('relu')(x)
    #         x = keras.layers.LeakyReLU()(x)
    #         x = keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    #
    #         x = keras.layers.Conv2D(64, kernel_size=(3, 3))(x)
    #         # x = keras.layers.Activation('relu')(x)
    #         x = keras.layers.LeakyReLU()(x)
    #         x = keras.layers.BatchNormalization()(x)
    #         x = keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    #
    #         x = keras.layers.Conv2D(1024, kernel_size=(3, 3))(x)
    #         # x = keras.layers.Activation('relu')(x)
    #         x = keras.layers.LeakyReLU()(x)
    #         x = keras.layers.BatchNormalization()(x)
    #         x = keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    #
    #         x = keras.layers.Add()([model.layers[-1].output, x])
    #         x = keras.layers.GlobalAveragePooling2D()(x)
    #         x = keras.layers.Dense(5)(x)  # Num Classes for CIFAR-10
    #         outputs = keras.layers.Activation('sigmoid')(x)
    #
    #         model = keras.models.Model(inputs=[model.input, inputs], outputs=[outputs])
    #
    #     if weights_path == "":
    #         weights_path = None
    #
    #     if weights_path is not None:
    #         print(f"load model weights_path: {weights_path}")
    #         model.load_weights(weights_path)
    #     return model
