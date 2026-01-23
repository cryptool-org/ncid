import multiprocessing
from pathlib import Path

import argparse
import sys
import time
import shutil
from sklearn.model_selection import train_test_split
import os
import math
import pickle
import functools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from datetime import datetime

# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam  # , Adamax
import tensorflow_datasets as tfds
sys.path.append("../")
from cipherTypeDetection.nullDistributionStrategy import NullDistributionStrategy
import cipherTypeDetection.config as config
from cipherTypeDetection.trainingBatch import TrainingBatch
from cipherTypeDetection.cipherStatisticsDataset import RotorCiphertextsDatasetParameters, PlaintextPathsDatasetParameters, CipherStatisticsDataset
from cipherTypeDetection.predictionPerformanceMetrics import PredictionPerformanceMetrics
from cipherTypeDetection.miniBatchEarlyStoppingCallback import MiniBatchEarlyStopping
from cipherTypeDetection.transformer import TransformerBlock, TokenAndPositionEmbedding
from cipherTypeDetection.learningRateSchedulers import TimeBasedDecayLearningRateScheduler, CustomStepDecayLearningRateScheduler
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def create_model_with_distribution_strategy(architecture, extend_model, output_layer_size, max_train_len):
    """Creates models depending on the GPU count and on extend_model"""
    print('Creating model...')

    strategy = None
    gpu_count = (len(tf.config.list_physical_devices('GPU')) +
        len(tf.config.list_physical_devices('XLA_GPU')))
    if gpu_count > 1:
        print("Multiple GPUs found.")
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of mirrored devices: {strategy.num_replicas_in_sync}.")
        with strategy.scope():
            if extend_model is not None:
                extend_model = tf.keras.models.load_model(extend_model, compile=False)
            model = create_model(architecture, extend_model, output_layer_size, max_train_len)
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
            model.summary()
    else:
        print("Only one GPU found.")
        strategy = NullDistributionStrategy()
        if extend_model is not None:
            extend_model = tf.keras.models.load_model(extend_model, compile=False)
        model = create_model(architecture, extend_model, output_layer_size, max_train_len)
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
            model.summary()

    print('Model created.\n')
    return model, strategy

def create_model(architecture, extend_model, output_layer_size, max_train_len):
    """
    Creates an un-trained model to use in the training process. 

    The kind of model that is returned, depends on the provided architecture and
    the `extend_model` flag. 

    Parameters
    ----------
    architecture : str
        The architecture of the model to create. 
    extend_model
        When `extend_model` is not None and architecure in ('FFNN', 'CNN', 'LSTM'), 
        the `extend_model` will be further trained.
    output_layer_size : int
        Defines the size of the output layer of the neural networks

    """
    optimizer = Adam(
        learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, 
        epsilon=config.epsilon, amsgrad=config.amsgrad)

    # Depends on the number of features returned by `calculate_statistics()` in 
    # `featureCalculations.py`.
    input_layer_size = 724 
    hidden_layer_size = int(2 * (input_layer_size / 3) + output_layer_size)

    # Create a model based on an existing one for further trainings
    if extend_model is not None:
        # remove the last layer
        model = tf.keras.Sequential()
        for layer in extend_model.layers[:-1]:
            model.add(layer)
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax', name="output"))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    # Create new model based on architecture
    if architecture == "FFNN":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "CNN":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(
                filters=config.filters, kernel_size=config.kernel_size, 
                input_shape=(max_train_len, 1), activation='relu'))
        for _ in range(config.layers - 1):
            model.add(tf.keras.layers.Conv1D(filters=config.filters, kernel_size=config.kernel_size, activation='relu'))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "LSTM":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(56, 64, input_length=max_train_len))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(config.lstm_units))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "DT":
        return DecisionTreeClassifier(criterion=config.criterion, ccp_alpha=config.ccp_alpha)
    
    elif architecture == "NB":
        return MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
    
    elif architecture == "RF":
        return RandomForestClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                      bootstrap=config.bootstrap, n_jobs=30,
                                      max_features=config.max_features, max_depth=30, 
                                      min_samples_split=config.min_samples_split,
                                      min_samples_leaf=config.min_samples_leaf)
    
    elif architecture == "ET":
        return ExtraTreesClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                    bootstrap=config.bootstrap, n_jobs=30,
                                    max_features=config.max_features, max_depth=30, 
                                    min_samples_split=config.min_samples_split,
                                    min_samples_leaf=config.min_samples_leaf)
    
    elif architecture == "Transformer":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        vocab_size = config.vocab_size
        maxlen = max_train_len
        embed_dim = config.embed_dim  # Embedding size for each token
        num_heads = config.num_heads  # Number of attention heads
        ff_dim = config.ff_dim  # Hidden layer size in feed forward network inside transformer

        inputs = tf.keras.layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(output_layer_size, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                        metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "SVM":
        return SVC(probability=True, C=1, gamma=0.001, kernel="linear")
    
    elif architecture == "SVM-Rotor":
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(probability=True, C=10, gamma=0.001, kernel="rbf"))])
        
        return pipe
    
    elif architecture == "kNN":
        return KNeighborsClassifier(90, weights="distance", metric="euclidean")
    
    elif architecture == "[FFNN,NB]":
        model_ffnn = tf.keras.Sequential()
        model_ffnn.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model_ffnn.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model_ffnn.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_ffnn.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                        metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        model_nb = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
        return [model_ffnn, model_nb]
    
    elif architecture == "[DT,ET,RF,SVM,kNN]":
        dt = DecisionTreeClassifier(criterion=config.criterion, ccp_alpha=config.ccp_alpha)
        et = ExtraTreesClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                  bootstrap=config.bootstrap, n_jobs=30,
                                  max_features=config.max_features, max_depth=30, 
                                  min_samples_split=config.min_samples_split,
                                  min_samples_leaf=config.min_samples_leaf)
        rf = RandomForestClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                    bootstrap=config.bootstrap, n_jobs=30,
                                    max_features=config.max_features, max_depth=30, 
                                    min_samples_split=config.min_samples_split,
                                    min_samples_leaf=config.min_samples_leaf)
        svm = SVC(probability=True, C=1, gamma=0.001, kernel="linear")
        knn = KNeighborsClassifier(90, weights="distance", metric="euclidean")
        return [dt, et, rf, svm, knn]
    
    else:
        raise Exception(f"Could not create model. Unknown architecture '{architecture}'.")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Script', 
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--train_dataset_size', default=16000, type=int,
                        help='Dataset size per fit. This argument should be dividable \n'
                             'by the amount of --ciphers.')
    parser.add_argument('--dataset_workers', default=1, type=int,
                        help='The number of parallel workers for reading the \ninput files.')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Defines how many times the same data is used to fit the model.')
    parser.add_argument('--plaintext_input_directory', default='../data/gutenberg_en', type=str,
                        help='Input directory of the plaintexts for training the aca ciphers.')
    parser.add_argument('--rotor_input_directory', default='../data/rotor_ciphertexts', type=str,
                        help='Input directory of the rotor ciphertexts.')
    parser.add_argument('--download_dataset', default=True, type=str2bool,
                        help='Download the dataset automatically.')
    parser.add_argument('--save_directory', default='../data/models/',
                        help='Directory for saving generated models. \n'
                             'When interrupting, the current model is \n'
                             'saved as interrupted_...')
    parser.add_argument('--model_name', default='m.h5', type=str,
                        help='Name of the output model file. The file must \nhave the .h5 extension.')
    parser.add_argument('--ciphers', default='all', type=str,
                        help='A comma seperated list of the ciphers to be created.\n'
                             'Be careful to not use spaces or use \' to define the string.\n'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere,\n'
                             '        Columnar Transposition, Plaifair and Hill)\n'
                             '- aca (contains all currently implemented ciphers from \n'
                             '       https://www.cryptogram.org/resource-area/cipher-types/)\n'
                             '- rotor (contains Enigma, M209, Purple, Sigaba and Typex ciphers)'
                             '- all (contains aca and rotor ciphers)'
                             '- all aca ciphers in lower case'
                             '- simple_substitution\n'
                             '- vigenere\n'
                             '- columnar_transposition\n'
                             '- playfair\n'
                             '- hill\n')
    parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known \n'
                             'symbols are defined in the alphabet of the cipher.')
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping training.')
    parser.add_argument('--min_train_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--min_test_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--max_train_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--max_test_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--architecture', default='FFNN', type=str, 
                        choices=['FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer',
                                 'SVM', 'kNN', '[FFNN,NB]', '[DT,ET,RF,SVM,kNN]', 'SVM-Rotor'],
                        help='The architecture to be used for training. \n'
                             'Possible values are:\n'
                             '- FFNN\n'
                             '- CNN\n'
                             '- LSTM\n'
                             '- DT\n'
                             '- NB\n'
                             '- RF\n'
                             '- ET\n'
                             '- Transformer\n'
                             '- SVM\n'
                             '- kNN\n'
                             '- [FFNN,NB]\n'
                             '- [DT,ET,RF,SVM,kNN]'
                             '- SVM-Rotor'
                             )
    parser.add_argument('--extend_model', default=None, type=str,
                        help='Load a trained model from a file and use it as basis for the new training.')

    return parser.parse_args()
    
def should_download_plaintext_datasets(args):
    """Determines if the plaintext datasets should be loaded"""
    return (args.download_dataset and 
            not os.path.exists(args.plaintext_input_directory) and 
            args.plaintext_input_directory == os.path.abspath('../data/gutenberg_en'))

def download_plaintext_datasets(args):
    """Downloads plaintexts and saves them in the plaintext_input_directory"""
    print("Downloading Datsets...")
    checksums_dir = '../data/checksums/'
    if not Path(checksums_dir).exists():
        os.mkdir(checksums_dir)
    tfds.download.add_checksums_dir(checksums_dir)

    download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', 
                                                       extract_dir=args.plaintext_input_directory)
    data_url = ('https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download' +
        '&confirm=t&uuid=afbc362d-9d52-472a-832b-c2af331a8d5b')
    try:
        download_manager.download_and_extract(data_url)
    except Exception as e:
        print("Download of datasets failed. If this issues persists, try downloading the dataset yourself "
              "from: https://drive.google.com/file/d/1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V/view."
              "(For more information see the README.md of this project.)")
        print("Underlying error:")
        print(e)
        sys.exit(1)

    path = os.path.join(args.plaintext_input_directory, 
                        'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                        'HawxhTtGOlSdcCrro4fxfEI8A', 
                        os.path.basename(args.plaintext_input_directory))
    dir_name = os.listdir(path)
    for name in dir_name:
        p = Path(os.path.join(path, name))
        parent_dir = p.parents[2]
        p.rename(parent_dir / p.name)
    os.rmdir(path)
    os.rmdir(os.path.dirname(path))
    print("Datasets Downloaded.")

def load_rotor_ciphertext_datasets_from_disk(args, batch_size):
    """
    Load ciphertext input data for rotor ciphertexts from disk.

    This method (in contrast to `load_plaintext_datasets_from_disk`) loads the data 
    immediately from disk. Currently this does not take too long, since the input files
    and the number of ciphers that need ciphertexts as input for the feature extraction 
    are limited. (If this method should lazily load ciphertexts, `RotorCiphertextsDataset` 
    needs to be adapted.)

    Parameters
    ----------
    args : 
        The arguments parsed by the `ArgumentParser`. See `parse_arguments()`.
    batch_size : int
            The number of samples and labels per batch.

    Returns
    -------
    tuple[list, RotorCiphertextsDatasetParameters, list, RotorCiphertextsDatasetParameters]
        A tuple with the training and testing ciphertexts as well as their
        `RotorCiphertextsDatasetParameters` that are both provided to the 
        `CipherStatisticsDataset`s for training and testing.
    """

    def validate_ciphertext_path(ciphertext_path, cipher_types):
        """Check if the filename of the file at `ciphertext_path` matches the 
        names inside `cipher_types`."""
        file_name = Path(ciphertext_path).stem.lower()
        if not file_name in cipher_types:
            raise Exception(f"Filename must equal one of the expected cipher types. "
                            f"Expected cipher types are: {cipher_types}. Current "
                            f"filename is '{file_name}'.")
    
    
    # Filter cipher_types to exclude non-ciphertext ciphers

    # Filter cipher_types to exclude non-ciphertext ciphers
    rotor_cipher_types = [config.CIPHER_TYPES[i] for i in range(56, 61)]

    # Load all ciphertexts in the train and dev folders in `args.rotor_input_directory`
    rotor_cipher_dir = args.rotor_input_directory

    def find_ciphertext_paths_in_dir(folder_path):
        """Loads all .txt files in the given folder and checks that their names match the
        known cipher types."""
        file_names = os.listdir(folder_path)
        result = []
        for name in file_names:
            path = os.path.join(folder_path, name)
            file_name, file_type = os.path.splitext(name)
            if os.path.isfile(path) and file_name.lower() in rotor_cipher_types and file_type == ".txt":
                validate_ciphertext_path(path, config.ROTOR_CIPHER_TYPES)

                result.append(path)
        return result
    
    train_dir = os.path.join(rotor_cipher_dir, "train")
    test_dir = os.path.join(rotor_cipher_dir, "dev")
    train_rotor_ciphertext_paths = find_ciphertext_paths_in_dir(train_dir)
    test_rotor_cipherext_paths = find_ciphertext_paths_in_dir(test_dir)
    

    # Return empty lists and parameters if no requested ciphers were found on disk
    if len(train_rotor_ciphertext_paths) == 0 or len(test_rotor_cipherext_paths) == 0:
        empty_params = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES, 
                                                         0,
                                                         args.dataset_workers, 
                                                         args.min_train_len, 
                                                         args.max_train_len,
                                                         generate_evalutation_data=False)
        return ([], empty_params, [], empty_params)

    # Create dataset parameters, which will be used for creating a `CipherStatisticsDataset`.
    # This class will provide an iterator in `train_model` to convert the plaintext files
    # (applying the provided options) into statistics (features) used for training.
    train_rotor_ciphertexts_parameters = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES, 
                                                            batch_size,
                                                            args.dataset_workers, 
                                                            args.min_train_len, 
                                                            args.max_train_len,
                                                            generate_evalutation_data=False)
    test_rotor_ciphertexts_parameters = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES,
                                                            batch_size,
                                                            args.dataset_workers, 
                                                            args.min_test_len, 
                                                            args.max_test_len,
                                                            generate_evalutation_data=False)
    
    # Return the tuples of training and testing rotor_ciphertexts as well as the parameter
    # for initializing the `CipherStatisticsDataset`s.
    return (train_rotor_ciphertext_paths, train_rotor_ciphertexts_parameters, 
            test_rotor_cipherext_paths, test_rotor_ciphertexts_parameters)

def load_plaintext_datasets_from_disk(args, requested_cipher_types, batch_size):
    """
    Gets all plaintext paths found in `args.plaintext_input_directory`, and converts
    them into training and testing list and parameters, used to create 
    `CipherStatisticsDataset`s.

    This method does not load the contents of the plaintext files. This is done 
    lazily by the `CipherStatisticsDataset`.

    Parameters
    ----------
        args :
            The arguments parsed by the `ArgumentParser`. See `parse_arguments()`.
        requested_cipher_types : list
            A list of cipher types to provide as parameters to the `CipherStatisticsDataset`.
            The list is filtered and only ACA ciphers are used as parameters, since the features
            of rotor ciphers currently have to be extracted from ciphertext files.
        batch_size : int
            The number of samples and labels per batch.
        
    Returns
    -------
    tuple[list, PlaintextPathsDatasetParameters, list, PlaintextPathsDatasetParameters]
        A tuple with the training and testing plaintext paths as well as their
        `PlaintextPathsDatasetParameters` that are both provided to the 
        `CipherStatisticsDataset`s for training and testing.
    """
    # Filter cipher_types to exclude non-plaintext ciphers
    cipher_types = [type for type in requested_cipher_types if type not in config.ROTOR_CIPHER_TYPES]

    # Get all paths to plaintext files in the `plaintext_input_directory`
    plaintext_files = []
    dir_name = os.listdir(args.plaintext_input_directory)
    for name in dir_name:
        path = os.path.join(args.plaintext_input_directory, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    
    # Use some plaintext for training and others for testing
    train_plaintexts, test_plaintexts = train_test_split(plaintext_files, test_size=0.05, 
                                                         random_state=42, shuffle=True)
    
    # Create dataset parameters, which will be used for creating a `CipherStatisticsDataset`.
    # This class will provide an iterator in `train_model` to convert the plaintext files
    # (applying the provided options) into statistics (features) used for training.
    train_plaintext_parameters = PlaintextPathsDatasetParameters(cipher_types, batch_size, 
                                                args.min_train_len, args.max_train_len,
                                                args.keep_unknown_symbols, args.dataset_workers)
    test_plaintext_parameters = PlaintextPathsDatasetParameters(cipher_types, batch_size,     
                                               args.min_test_len, args.max_test_len,
                                               args.keep_unknown_symbols, args.dataset_workers)
    
    # Return the training and testing plaintexts as well as their parameters
    return (train_plaintexts, train_plaintext_parameters, test_plaintexts, test_plaintext_parameters)

def load_datasets_from_disk(args, requested_cipher_types):
    """
    Loads training and testing data from the file system. 

    In case of the ACA ciphers the datasets are plaintext files that need to be
    encrypted before the features can be extracted. In case of the rotor ciphers 
    there are already encrypted ciphertext files that can directly be used to 
    extract the features.
    To simplify the training code, both kinds of input data are returned in 
    `CipherStatisticsDataset`s that provide an iterator interface, returning
    `TrainingBatch`es of the requested size on each `next()` call.
    Plaintext input is loaded lazily, while ciphertexts currently are loaded 
    immediately.

    Parameters
    ----------
    args :
        The parsed commandline arguments. See also `parse_arguments()`.
    requested_cipher_types : list
        A list of the requested cipher types. These are provided as parameters to
        the returned `CipherStatisticsDataset`s as well as for selection of input
        files.
    max_rotor_lines : int
        Limits the amount of input lines loaded from ciphertext files. 

    Returns
    -------
    tuple[CipherStatisticsDataset]
        Training and testing `CipherStatisticsDataset` that lazily calculate the
        features for the input data on `next()` calls.
    """

    print("Loading Datasets...")

    # Filter cipher_types to exclude non-ciphertext ciphers
    rotor_cipher_types = [config.CIPHER_TYPES[i] for i in range(56, 61)]
    non_rotor_ciphers = [type for type in requested_cipher_types if type not in rotor_cipher_types]
    rotor_cipher_types = [type for type in requested_cipher_types if type in rotor_cipher_types]

    # Calculate batch size for rotor ciphers. If both aca and rotor ciphers are requested, 
    # the amount of samples of each rotor cipher per batch should be equal to the 
    # amount of samples of each aca cipher per loaded batch.
    number_of_rotor_ciphers = len(rotor_cipher_types)
    number_of_aca_ciphers = len(non_rotor_ciphers)
    if number_of_aca_ciphers <= 0:
        rotor_dataset_batch_size = args.train_dataset_size
        aca_dataset_batch_size = 0
    else:
        amount_of_samples_per_cipher = args.train_dataset_size // (number_of_aca_ciphers + number_of_rotor_ciphers)
        rotor_dataset_batch_size = amount_of_samples_per_cipher * number_of_rotor_ciphers
        aca_dataset_batch_size = amount_of_samples_per_cipher * number_of_aca_ciphers
    
    # Load the plaintext file paths and the rotor ciphertexts from disk.
    (train_plaintexts, 
     train_plaintext_parameters, 
     test_plaintexts, 
     test_plaintext_parameters) = load_plaintext_datasets_from_disk(args, 
                                                                    requested_cipher_types,
                                                                    aca_dataset_batch_size)
    (train_rotor_ciphertext_paths, 
     train_rotor_ciphertexts_parameters, 
     test_rotor_ciphertext_paths, 
     test_rotor_ciphertexts_parameters) = load_rotor_ciphertext_datasets_from_disk(args, 
                                                                                   rotor_dataset_batch_size)

    # Convert the training and testing ciphertexts and plaintexts, as well as 
    # their parameters into `CipherStatisticsDataset`s.
    train_ds = CipherStatisticsDataset(train_plaintexts, train_plaintext_parameters, train_rotor_ciphertext_paths, train_rotor_ciphertexts_parameters)
    test_ds = CipherStatisticsDataset(test_plaintexts, test_plaintext_parameters, test_rotor_ciphertext_paths, test_rotor_ciphertexts_parameters)

    if train_ds.key_lengths_count > 0 and args.train_dataset_size % train_ds.key_lengths_count != 0:
        print("WARNING: the --train_dataset_size parameter must be dividable by the amount of --ciphers  and the length configured "
              "KEY_LENGTHS in config.py. The current key_lengths_count is %d" % 
                  train_ds.key_lengths_count, file=sys.stderr)
        
    print("Datasets loaded.\n")

    # Return `CipherStatisticsDataset`s for training and testing.
    return train_ds, test_ds
    
def train_model(model, strategy, args, train_ds):
    """
    Trains the model with the given training dataset.

    Depending on the value of `args.architecture` a different approach is 
    taken to train the model. Some architectures need to be trained in one
    iteration, while others can be trained with multiple input batches. 
    While the training is in progress, status messages are logged to 
    stdout to indicate the amount of seen input data as well as the current
    accuracy of the trained model.

    Parameters
    ----------
    model :
        The model that will be trained. Needs to match `args.architecture`.
    strategy :
        A distribution strategy (of the `Tensorflow` library) to distribute
        the `fit` calls to multiple devices. Could also be a `NullStrategy`
        if no GPU devices are found on the system.
    args :
        Commandline arguments entered by the user. See also: `parse_arguments()`.
    train_ds :
        A `CipherStatisticsDataset` providing the features to use for training.

    Returns
    -------
    tuple
    """

    checkpoints_dir = Path('../data/checkpoints')
    def delete_previous_checkpoints():
        shutil.rmtree(checkpoints_dir)

    def create_checkpoint_callback():
        """Provides a `keras` `ModelCheckpoint` used to periodically save a model in training"""
        if not checkpoints_dir.exists():
            os.mkdir(checkpoints_dir)
        checkpoint_file_path = os.path.join(checkpoints_dir, 
                                            "epoch_{epoch:02d}-"
                                            "acc_{accuracy:.2f}.h5")

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=False,
            save_freq=100)

    print('Training model...')

    delete_previous_checkpoints()

    # Create callbacks for tensorflow models
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../data/logs', 
                                                          update_freq='epoch')
    early_stopping_callback = MiniBatchEarlyStopping(min_delta=1e-5, 
                                                     patience=250, 
                                                     monitor='accuracy', 
                                                     mode='max', 
                                                     restore_best_weights=True)
    custom_step_decay_lrate_callback = CustomStepDecayLearningRateScheduler(early_stopping_callback)
    checkpoint_callback = create_checkpoint_callback()

    # Initialize variables
    architecture = args.architecture
    start_time = time.time()
    train_iter = 0
    train_epoch = 0
    val_data = None
    val_labels = None
    training_batches = None
    combined_batch = TrainingBatch("mixed", [], [])
    classes = list(range(len(config.CIPHER_TYPES)))
    should_create_validation_data = True

    # Perform main training loop while the iterations don't exceed the user provided max_iter
    while train_ds.iteration < args.max_iter:
        training_batches = next(train_ds)

        # For architectures that only support one fit call: Sample all batches into one large batch.
        if architecture in ("DT", "RF", "ET", "SVM", "kNN", "SVM-Rotor", "[DT,ET,RF,SVM,kNN]"):
            for training_batch in training_batches:
                combined_batch.extend(training_batch)
            if train_ds.iteration < args.max_iter:
                print("Loaded %d ciphertexts." % train_ds.iteration)
                continue
            train_ds.stop_outstanding_tasks()
            print("Loaded %d ciphertexts." % train_ds.iteration)
            training_batches = [combined_batch]

        for index, training_batch in enumerate(training_batches):
            statistics, labels = training_batch.items()
            train_iter = train_ds.iteration - len(training_batch) * (len(training_batches) - index - 1)

            # Create small validation dataset on first iteration
            if should_create_validation_data:
                statistics, val_data, labels, val_labels = train_test_split(statistics.numpy(), 
                                                                            labels.numpy(), 
                                                                            test_size=0.3)
                statistics = tf.convert_to_tensor(statistics)
                val_data = tf.convert_to_tensor(val_data)
                labels = tf.convert_to_tensor(labels)
                val_labels = tf.convert_to_tensor(val_labels)
                should_create_validation_data = False
                train_iter -= len(training_batch) * 0.3

            # scikit-learn architectures:
            if architecture in ("DT", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
                train_iter = len(labels) * 0.7
                print(f"Start training the {architecture}.")
                if architecture == "kNN":
                    history = model.fit(statistics, labels)
                elif architecture == "SVM-Rotor":
                    history = model.fit(list(statistics), list(labels))
                    # print(f"RFE-support: \n{list(model.support_)}\n")
                    # print(f"RFE-rank: \n{list(model.ranking_)}\n")
                    # print()
                else:
                    history = model.fit(statistics, labels)
                if architecture == "DT":
                    plt.gcf().set_size_inches(25, 25 / math.sqrt(2))
                    print("Plotting tree.")
                    plot_tree(model, max_depth=3, fontsize=6, filled=True)
                    plt.savefig(args.model_name.split('.')[0] + '_decision_tree.svg', 
                                dpi=200, bbox_inches='tight', pad_inches=0.1)

            # Naive Bayes training
            elif architecture == "NB":
                history = model.partial_fit(statistics, 
                                            labels, 
                                            classes=classes)

            # Ensemble: [FFNN,NB]
            elif architecture == "[FFNN,NB]":
                with strategy.scope():
                    history = model[0].fit(statistics, 
                                        labels, 
                                        batch_size=args.batch_size, 
                                        validation_data=(val_data, val_labels), 
                                        epochs=args.epochs,
                                        callbacks=[early_stopping_callback, 
                                                    tensorboard_callback,    
                                                    custom_step_decay_lrate_callback, 
                                                    checkpoint_callback])
                history = model[1].partial_fit(statistics, 
                                               labels, 
                                               classes=classes)
            
            # Ensemble: [DT,ET,RF,SVM,kNN]
            elif architecture == "[DT,ET,RF,SVM,kNN]":
                print(f"Start training the {architecture}.")
                dt, et, rf, svm, knn = model
                for index, m in enumerate([dt, et, rf, svm]):
                    m.fit(statistics, labels)
                    print(f"Trained model {index + 1} of {len(model)}")
                knn.fit(statistics, labels)
                print(f"Trained model {len(model)} of {len(model)}")

            else:
                with strategy.scope():
                    history = model.fit(statistics, labels, 
                                    batch_size=args.batch_size, 
                                    validation_data=(val_data, val_labels), 
                                    epochs=args.epochs,
                                    callbacks=[early_stopping_callback, 
                                            tensorboard_callback, 
                                            custom_step_decay_lrate_callback, 
                                            checkpoint_callback])
                    
            # print for Decision Tree, Naive Bayes and Random Forests
            if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
                val_score = model.score(val_data, val_labels)
                train_score = model.score(statistics, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if architecture == "[FFNN,NB]":
                val_score = model[1].score(val_data, val_labels)
                train_score = model[1].score(statistics, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if architecture == "[DT,ET,RF,SVM,kNN]":
                for m in model:
                    val_score = m.score(val_data, val_labels)
                    train_score = m.score(statistics, labels)
                    print(f"{type(m).__name__}: train accuracy: {train_score}, "
                          f"validation accuracy: {val_score}")

            if train_ds.epoch > 0:
                train_epoch = (train_ds.iteration 
                               // ((train_iter + train_ds.batch_size * train_ds.dataset_workers) 
                               // train_ds.epoch))
                
            print("Epoch: %d, Iteration: %d" % (train_epoch, train_iter))
            if train_iter >= args.max_iter or early_stopping_callback.stop_training:
                break
            
        if train_ds.iteration >= args.max_iter or early_stopping_callback.stop_training:
            train_ds.stop_outstanding_tasks()
            break

    elapsed_training_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    training_stats = ("Finished training in %d days %d hours %d minutes %d seconds "
                      "with %d iterations and %d epochs.\n" 
                      % (elapsed_training_time.days, 
                         elapsed_training_time.seconds // 3600, 
                         (elapsed_training_time.seconds // 60) % 60,
                         elapsed_training_time.seconds % 60, 
                         train_iter, 
                         train_epoch))
    print(training_stats)
    return early_stopping_callback, train_iter, training_stats
        
def save_model(model, args):
    """Writes the model and the commandline arguments to disk."""

    print('Saving model...')
    architecture = args.architecture
    if not os.path.exists(args.save_directory):
        os.mkdir(args.save_directory)
    if args.model_name == 'm.h5':
        i = 1
        while os.path.exists(os.path.join(args.save_directory, args.model_name.split('.')[0] + str(i) + '.h5')):
            i += 1
        model_name = args.model_name.split('.')[0] + str(i) + '.h5'
    else:
        model_name = args.model_name
    model_path = os.path.join(args.save_directory, model_name)

    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        model.save(model_path)

    elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
        with open(model_path, "wb") as f:
            # this gets very large
            pickle.dump(model, f)

    elif architecture == "[FFNN,NB]":
        model[0].save('../data/models/' + model_path.split('.')[0] + "_ffnn.h5")
        with open('../data/models/' + model_path.split('.')[0] + "_nb.h5", "wb") as f:
            # this gets very large
            pickle.dump(model[1], f)

    elif architecture == "[DT,ET,RF,SVM,kNN]":
        for index, name in enumerate(["dt","et","rf","svm","knn"]):
            # TODO: Are these files actually in the h5 format? Probably not!
            with open('../data/models/' + model_path.split('.')[0] + f"_{name}.h5", "wb") as f:
                # this gets very large
                pickle.dump(model[index], f)

    # Write user provided commandline arguments into mode path
    with open('../data/' + model_path.split('.')[0] + '_parameters.txt', 'w') as f:
        for arg in vars(args):
            f.write("{:23s}= {:s}\n".format(arg, str(getattr(args, arg))))

    # Remove logs of previous run
    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        logs_destination = '../data/' + model_name.split('.')[0] + '_tensorboard_logs'
        try:
            if os.path.exists(logs_destination):
                shutil.rmtree(logs_destination)
            shutil.move('../data/logs', logs_destination)
        except Exception:
            print(f"Could not remove logs of previous run. Move of current logs "
                  f"from '../data/logs' to '{logs_destination}' failed.")
            
    print('Model saved.\n')

def predict_test_data(test_ds, model, args, early_stopping_callback, train_iter):
    """
    Testing the predictions of the model.

    The trained model is used to predict the data in `test_ds` and the results
    are evaluated in regard to accuracy, precision, recall, etc. The calculated
    metrics are printed to stdout.

    Parameters
    ----------
    test_ds : CipherStatisticsDataset
        The dataset used for prediction.
    model :
        The trained model to evaluate.
    args :
        The commandline arguments provided by the user.
    early_stopping_callback :
        Indicates whether the training was stopped before `args.max_iter` was 
        reached. Used together with `train_iter` and `args.max_iter` to control 
        the number of prediction iterations.
    train_iter : int
        The number of iterations used until the model converged. Used together with
        `early_stopping_callback` and `args.max_iter` to control the number of 
        prediction iterations.
    
    Returns
    -------
    str
        The statistics of this prediction run.
    """

    print('Predicting test data...\n')

    architecture = args.architecture
    start_time = time.time()
    total_len_prediction = 0
    cntr = 0
    test_iter = 0
    test_epoch = 0

    # Determine the number of iterations to use for evaluating the model
    prediction_dataset_factor = 10
    if early_stopping_callback.stop_training:
        while test_ds.dataset_workers * test_ds.batch_size > train_iter / prediction_dataset_factor and prediction_dataset_factor > 1:
            prediction_dataset_factor -= 1
        args.max_iter = int(train_iter / prediction_dataset_factor)
    else:
        while test_ds.dataset_workers * test_ds.batch_size > args.max_iter / prediction_dataset_factor and prediction_dataset_factor > 1:
            prediction_dataset_factor -= 1
        args.max_iter /= prediction_dataset_factor

    # Initialize `PredictionPerformanceMetrics` instances for all classifiers. These
    # are used to save and evaluate the batched prediction results of the models.
    prediction_metrics = {}
    if architecture == "[FFNN,NB]":
        prediction_metrics = {"FFNN": PredictionPerformanceMetrics(model_name="FFNN"),
                              "NB": PredictionPerformanceMetrics(model_name="NB")}
    elif architecture == "[DT,ET,RF,SVM,kNN]":
        prediction_metrics = {"DT": PredictionPerformanceMetrics(model_name="DT"),
                              "ET": PredictionPerformanceMetrics(model_name="ET"),
                              "RF": PredictionPerformanceMetrics(model_name="RF"),
                              "SVM": PredictionPerformanceMetrics(model_name="SVM"),
                              "kNN": PredictionPerformanceMetrics(model_name="kNN"),}
    else:
        prediction_metrics = {architecture: PredictionPerformanceMetrics(model_name=architecture)}

    combined_batch = TrainingBatch("mixed", [], [])
    while test_ds.iteration < args.max_iter:
        testing_batches = next(test_ds)

        # For architectures that only support one fit call: Sample all batches into one large batch.
        if architecture in ("DT", "RF", "ET", "SVM", "kNN", "SVM-Rotor", "[DT,ET,RF,SVM,kNN]"):
            for testing_batch in testing_batches:
                combined_batch.extend(testing_batch)
            if test_ds.iteration < args.max_iter:
                print("Loaded %d ciphertexts." % test_ds.iteration)
                continue
            test_ds.stop_outstanding_tasks()
            print("Loaded %d ciphertexts." % test_ds.iteration)
            testing_batches = [combined_batch]

        for testing_batch in testing_batches:
            statistics, labels = testing_batch.items()
            
            # Decision Tree, Naive Bayes prediction
            if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                prediction = model.predict_proba(statistics)
                prediction_metrics[architecture].add_predictions(labels, prediction)
            elif architecture == "SVM-Rotor":
                prediction = model.predict_proba(statistics)
                # add probability 0 to all aca labels that are missing in the prediction
                padded_prediction = []
                for p in list(prediction):
                    padded = [0] * 56 + list(p)
                    padded_prediction.append(padded)
                prediction_metrics[architecture].add_predictions(labels, padded_prediction)
            elif architecture == "[FFNN,NB]":
                prediction = model[0].predict(statistics, batch_size=args.batch_size, verbose=1)
                nb_prediction = model[1].predict_proba(statistics)
                prediction_metrics["FFNN"].add_predictions(labels, prediction)
                prediction_metrics["NB"].add_predictions(labels, nb_prediction)
            elif architecture == "[DT,ET,RF,SVM,kNN]":
                prediction = model[0].predict_proba(statistics)
                prediction_metrics["DT"].add_predictions(labels, prediction)
                prediction_metrics["ET"].add_predictions(labels, model[1].predict_proba(statistics))
                prediction_metrics["RF"].add_predictions(labels, model[2].predict_proba(statistics))
                prediction_metrics["SVM"].add_predictions(labels, model[3].predict_proba(statistics))
                prediction_metrics["kNN"].add_predictions(labels, model[4].predict_proba(statistics))
            else:
                prediction = model.predict(statistics, batch_size=args.batch_size, verbose=1)
                prediction_metrics[architecture].add_predictions(labels, prediction)

            total_len_prediction += len(prediction)
            cntr += 1
            test_iter = args.train_dataset_size * cntr
            test_epoch = test_ds.epoch
            if test_epoch > 0:
                test_epoch = test_iter // ((test_ds.iteration + test_ds.batch_size * test_ds.dataset_workers) // test_ds.epoch)
            print("Prediction Epoch: %d, Iteration: %d / %d" % (test_epoch, test_iter, args.max_iter))
            if test_iter >= args.max_iter:
                break
        if test_ds.iteration >= args.max_iter:
            break
    
    test_ds.stop_outstanding_tasks()
    elapsed_prediction_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)

    if total_len_prediction > args.train_dataset_size:
        total_len_prediction -= total_len_prediction % args.train_dataset_size
    print('\ntest data predicted: %d ciphertexts' % total_len_prediction)

    # print prediction metrics
    for metrics in prediction_metrics.values():
        metrics.print_evaluation()
    
    # print selected feature again
    if architecture == "SVM-Rotor":
        # print(f"RFE-support: \n{list(model.support_)}\n")
        # print(f"Support names: {convert_rfe_support_to_names(model.support_)}")
        # print(f"RFE-rank: \n{list(model.ranking_)}\n")
        print()

    # print("GridSearchCV:")
    # print(f"Best score: {model.best_score_}")
    # print(f"Best params: {model.best_params_}")

    prediction_stats = 'Prediction time: %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.' % (
        elapsed_prediction_time.days, elapsed_prediction_time.seconds // 3600, 
        (elapsed_prediction_time.seconds // 60) % 60,
        elapsed_prediction_time.seconds % 60, test_iter, test_epoch)
    
    return prediction_stats

def expand_cipher_groups(cipher_types):
    """Turn cipher group identifiers (ACA, MTC3, ROTOR, ALL) into a list of their ciphers."""
    expanded = cipher_types
    if config.MTC3 in expanded:
        del expanded[expanded.index(config.MTC3)]
        for i in range(5):
            expanded.append(config.CIPHER_TYPES[i])
    elif config.ACA in expanded:
        del expanded[expanded.index(config.ACA)]
        for i in range(56):
            expanded.append(config.CIPHER_TYPES[i])
    elif config.ROTOR in expanded:
        del expanded[expanded.index(config.ROTOR)]
        for i in range(56, 61):
            expanded.append(config.CIPHER_TYPES[i])
    elif config.ALL in expanded:
        del expanded[expanded.index(config.ALL)]
        expanded.extend(config.CIPHER_TYPES)
    return expanded

def main():
    # Don't fork processes to keep memory footprint low. 
    multiprocessing.set_start_method("spawn")

    args = parse_arguments()

    cpu_count = os.cpu_count()
    if cpu_count and cpu_count < args.dataset_workers:
        print("WARNING: More dataset_workers set than CPUs available.")

    # Print arguments
    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))

    args.plaintext_input_directory = os.path.abspath(args.plaintext_input_directory)
    args.rotor_input_directory = os.path.abspath(args.rotor_input_directory)
    args.ciphers = args.ciphers.lower()
    cipher_types = args.ciphers.split(',')
    architecture = args.architecture
    extend_model = args.extend_model

    # Validate inputs
    if len(os.path.splitext(args.model_name)) != 2 or os.path.splitext(args.model_name)[1] != '.h5':
        print('ERROR: The model name must have the ".h5" extension!', file=sys.stderr)
        sys.exit(1)

    if extend_model is not None:
        if architecture not in ('FFNN', 'CNN', 'LSTM'):
            print('ERROR: Models with the architecture %s can not be extended!' % architecture,
                  file=sys.stderr)
            sys.exit(1)
        if len(os.path.splitext(extend_model)) != 2 or os.path.splitext(extend_model)[1] != '.h5':
            print('ERROR: The extended model name must have the ".h5" extension!', file=sys.stderr)
            sys.exit(1)

    if architecture == "SVM-Rotor" and cipher_types[0] != "rotor":
        print(f"When training rotor-only model, the argument `ciphers` "
              f"should equal 'rotor'. Selected ciphers are: '{cipher_types}'.")
        sys.exit(1)

    if args.train_dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --train_dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % 
                  (args.train_dataset_size * args.dataset_workers, args.max_iter), 
              file=sys.stderr)
        sys.exit(1)

    # Convert commandline cipher argument (all, aca, mtc3, rotor, etc.) to list of 
    # all ciphers contained in the provided group. E.g. 'rotor' gets expanded
    # into 'enigma', 'm209', etc. 
    cipher_types = expand_cipher_groups(cipher_types)

    # Ensure plaintext dataset is available at `args.plaintext_input_directory`.
    if should_download_plaintext_datasets(args):
        download_plaintext_datasets(args)

    # Load the datasets for the requested cipher types. If aca and rotor cipher types
    # are contained in `cipher_types`, both plaintext and ciphertext datasets are loaded.
    train_ds, test_ds = load_datasets_from_disk(args, cipher_types)

    # Get the number of cipher classes to predict. Since the label numbers are fixed, 
    # it must be ensured that the output_layer_size of the neural networks contain 
    # enough nodes upto the higest wanted class label.
    output_layer_size = max([config.CIPHER_TYPES.index(type) for type in cipher_types]) + 1

    # Create a model and allow for distributed training on multi-GPU machines
    model, strategy = create_model_with_distribution_strategy(architecture, 
                                                    extend_model, 
                                                    output_layer_size=output_layer_size, 
                                                    max_train_len=args.max_train_len)
    
    early_stopping_callback, train_iter, training_stats = train_model(model, strategy, 
                                                                      args, train_ds)
    save_model(model, args)
    prediction_stats = predict_test_data(test_ds, model, args, early_stopping_callback, train_iter)
    
    print(training_stats)
    print(prediction_stats)

if __name__ == "__main__":
    main()    
