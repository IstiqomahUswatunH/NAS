import os
import warnings
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

from CONSTANTS import *


class MLPSearchSpace(object):
    """
    MLPSearchSpace class for creating a search space for MLP architectures.

    Args:
        target_classes (int): The number of target classes for the classification task.

    Attributes:
        target_classes (int): The number of target classes.
        vocab (dict): A vocabulary dictionary mapping unique layer IDs to layer parameters.

    Methods:
        __init__(self, target_classes): Initialize the search space with the specified number of target classes.
        vocab_dict(self): Generate and return the vocabulary dictionary for layer configurations.
        encode_sequence(self, sequence): Encode a list of layer configurations into a sequence of layer IDs.
        decode_sequence(self, sequence): Decode a sequence of layer IDs into a list of layer configurations.
    """
    def __init__(self, target_classes):
        """
        Initialize the MLPSearchSpace with the specified number of target classes.

        Args:
            target_classes (int): The number of target classes for the classification task.
        """
        self.target_classes = target_classes
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        layer_params = []
        layer_id = []
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)
        vocab = dict(zip(layer_id, layer_params))
        vocab[len(vocab) + 1] = (('dropout'))
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        return vocab

    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence

    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence


class MLPGenerator(MLPSearchSpace):
    """
    MLPGenerator class for generating and training MLP models.

    This class inherits from MLPSearchSpace and extends its functionality to create, compile,
    update weights, set model weights, and train MLP models.

    Attributes:
        target_classes (int): The number of target classes for the classification task.
        mlp_optimizer (str): The optimizer used for training the MLP model.
        mlp_lr (float): The learning rate for the optimizer.
        mlp_decay (float): The weight decay for the optimizer.
        mlp_momentum (float): The momentum for SGD optimizer.
        mlp_dropout (float): The dropout rate applied to the model.
        mlp_loss_func (str): The loss function used for training.
        mlp_one_shot (bool): Whether to use shared weights for one-shot training.
        metrics (list): A list of metrics used for model evaluation.

    Methods:
        create_model(self, sequence, mlp_input_shape): Create an MLP model based on a sequence of layer configurations.
        compile_model(self, model): Compile the MLP model with specified optimizer, learning rate, and loss function.
        update_weights(self, model): Update shared weights based on the model's layer configurations.
        set_model_weights(self, model): Set the model's weights based on shared weights.
        train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None):
        Train the model with the given data and hyperparameters.
    """

    def __init__(self):
        """
        Initialize the MLPGenerator with hyperparameters and settings for model generation and training.
        """

        self.target_classes = TARGET_CLASSES
        self.mlp_optimizer = MLP_OPTIMIZER
        self.mlp_lr = MLP_LEARNING_RATE
        self.mlp_decay = MLP_DECAY
        self.mlp_momentum = MLP_MOMENTUM
        self.mlp_dropout = MLP_DROPOUT
        self.mlp_loss_func = MLP_LOSS_FUNCTION
        self.mlp_one_shot = MLP_ONE_SHOT
        self.metrics = ['accuracy']

        super().__init__(TARGET_CLASSES)


        if self.mlp_one_shot:
            self.weights_file = 'LOGS/shared_weights.pkl'
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)

    def create_model(self, sequence, mlp_input_shape):
        layer_configs = self.decode_sequence(sequence)
        model = Sequential()
        if len(mlp_input_shape) > 1:
            model.add(Flatten(name='flatten', input_shape=mlp_input_shape))
            for i, layer_conf in enumerate(layer_configs):
                if layer_conf is 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        else:
            for i, layer_conf in enumerate(layer_configs):
                if i == 0:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))
                elif layer_conf is 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        return model

    def compile_model(self, model):
        if self.mlp_optimizer == 'sgd':
            optim = optimizers.SGD(lr=self.mlp_lr, decay=self.mlp_decay, momentum=self.mlp_momentum)
        else:
            optim = getattr(optimizers, self.mlp_optimizer)(lr=self.mlp_lr, decay=self.mlp_decay)
        model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)
        return model

    def update_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) == 0:
                    self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
                                                                      'weights': layer.get_weights()},
                                                                     ignore_index=True)
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                j += 1
        self.shared_weights.to_pickle(self.weights_file)

    def set_model_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) > 0:
                    print("Transferring weights for layer:", config_ids[j])
                    layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                j += 1

    def train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None):
        if self.mlp_one_shot:
            self.set_model_weights(model)
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
            self.update_weights(model)
        else:
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
        return history