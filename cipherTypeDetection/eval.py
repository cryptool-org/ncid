import multiprocessing
from pathlib import Path
import argparse
import sys
import os
import pickle
import functools
import time
from datetime import datetime

import torch
import torch.nn.functional as F

# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.append("../")
from util.utils import map_text_into_numberspace, print_progress
import cipherTypeDetection.config as config
from cipherTypeDetection.cipherStatisticsDataset import CipherStatisticsDataset, PlaintextPathsDatasetParameters, RotorCiphertextsDatasetParameters, calculate_statistics, pad_sequences
from cipherTypeDetection.predictionPerformanceMetrics import PredictionPerformanceMetrics
from cipherTypeDetection.rotorDifferentiationEnsemble import RotorDifferentiationEnsemble
from cipherTypeDetection.ensembleModel import EnsembleModel
from cipherTypeDetection.models.modelFile import ModelFile
from util.utils import get_model_input_length
from cipherImplementations.cipher import OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER
from cipherTypeDetection.config import Backend
from cipherTypeDetection.models.predictionHelper import predict_ffnn, predict_lstm, predict_cnn, predict_transformer
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def evaluate_torch(model, inputs, labels, batch_size):
    with torch.no_grad():
        outputs = model.predict(inputs, batch_size)
        y = torch.tensor(labels.numpy(), dtype=torch.long)

        loss = F.cross_entropy(outputs, y)
        top1 = torch.argmax(outputs, dim=1)
        acc = (top1 == y).float().mean()

        # Calc top-3
        top3 = torch.topk(outputs, k=3, dim=1).indices
        y_expanded = y.unsqueeze(1).expand_as(top3)
        k3_acc = (top3 == y_expanded).any(dim=1).float().mean()

        return loss.item(), acc.item(), k3_acc.item()

def benchmark(args, model, architecture, backend):
    cipher_types = args.ciphers
    args.plaintext_folder = os.path.abspath(args.plaintext_folder)
    if args.dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --dataset_size * --dataset_workers must not be bigger than --max_iter. In this case it was %d > %d" % (
            args.dataset_size * args.dataset_workers, args.max_iter), file=sys.stderr)
        sys.exit(1)
    if args.download_dataset and not os.path.exists(args.plaintext_folder) and args.plaintext_folder == os.path.abspath(
            '../data/gutenberg_en'):
        print("Downloading Datsets...")
        tfds.download.add_checksums_dir('../data/checksums/')
        download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', extract_dir=args.plaintext_folder)
        download_manager.download_and_extract(
            'https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download')
        path = os.path.join(args.plaintext_folder, 'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                                                    'HawxhTtGOlSdcCrro4fxfEI8A', os.path.basename(args.plaintext_folder))
        dir_nam = os.listdir(path)
        for name in dir_nam:
            p = Path(os.path.join(path, name))
            parent_dir = p.parents[2]
            p.rename(parent_dir / p.name)
        os.rmdir(path)
        os.rmdir(os.path.dirname(path))
        print("Datasets Downloaded.")

    print("Loading Datasets...")
    def validate_ciphertext_path(ciphertext_path, cipher_types):
        file_name = Path(ciphertext_path).stem.lower()
        if not file_name in cipher_types:
            raise Exception(f"Filename must equal one of the expected cipher types. Expected cipher types are: {cipher_types}. Current filename is '{file_name}'.")
        
    plaintext_files = []
    dir_nam = os.listdir(args.plaintext_folder)
    for name in dir_nam:
        path = os.path.join(args.plaintext_folder, name)
        if os.path.isfile(path):
            plaintext_files.append(path)

    def find_ciphertext_paths_in_dir(folder_path):
        """Loads all .txt files in the given folder and checks that their names match the
        known cipher types."""
        file_names = os.listdir(folder_path)
        result = []
        for name in file_names:
            path = os.path.join(folder_path, name)
            file_name, file_type = os.path.splitext(name)
            if os.path.isfile(path) and file_name.lower() in cipher_types and file_type == ".txt":
                validate_ciphertext_path(path, config.ROTOR_CIPHER_TYPES)

                result.append(path)
        return result
    
    eval_rotor_ciphertext_paths = find_ciphertext_paths_in_dir(args.rotor_ciphertext_folder)
    
    # Calculate batch size for rotor ciphers. The amount of samples per rotor cipher should be
    # equal to the amount of samples per aca cipher.
    number_of_rotor_ciphers = len(config.ROTOR_CIPHER_TYPES)
    number_of_aca_ciphers = len(config.CIPHER_TYPES) - number_of_rotor_ciphers
    amount_of_samples_per_cipher = args.dataset_size // number_of_aca_ciphers
    rotor_train_dataset_size = amount_of_samples_per_cipher * number_of_rotor_ciphers

    plaintext_dataset_params = PlaintextPathsDatasetParameters(cipher_types[:56], 
                                                               args.dataset_size, 
                                                               args.min_text_len, 
                                                               args.max_text_len,
                                                               args.keep_unknown_symbols, 
                                                               args.dataset_workers, 
                                                               generate_evaluation_data=True)
    rotor_dataset_params = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES, 
                                                            rotor_train_dataset_size,
                                                            args.dataset_workers, 
                                                            args.min_text_len, 
                                                            args.max_text_len,
                                                            generate_evalutation_data=True)
    dataset = CipherStatisticsDataset(plaintext_files, plaintext_dataset_params, eval_rotor_ciphertext_paths, 
                                      rotor_dataset_params, generate_evaluation_data=True)

    if args.dataset_size % dataset.key_lengths_count != 0:
        print("WARNING: the --dataset_size parameter must be dividable by the amount of --ciphers  and the length configured KEY_LENGTHS in"
              " config.py. The current key_lengths_count is %d" % dataset.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")

    print('Evaluating model...')

    start_time = time.time()
    iteration = 0
    epoch = 0
    results = []
    prediction_metrics = PredictionPerformanceMetrics(model_name=architecture)

    while dataset.iteration < args.max_iter:
        batches = next(dataset)
        
        for index, batch in enumerate(batches):
            statistics, labels, ciphertexts = batch.items()

            if architecture == "FFNN" and backend == Backend.KERAS:
                results.append(model.evaluate(statistics, labels, batch_size=args.batch_size, verbose=1))
            elif architecture == "FFNN" and backend == Backend.PYTORCH:
                results.append(evaluate_torch(model, statistics, labels, batch_size=args.batch_size))
            elif architecture == "LSTM" and backend == Backend.PYTORCH:
                results.append(evaluate_torch(model, ciphertexts, labels, batch_size=args.batch_size))
            elif architecture in ("CNN", "Transformer"):
                results.append(model.evaluate(ciphertexts, labels, batch_size=args.batch_size, verbose=1))
            elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                results.append(model.score(statistics, labels))
                print("accuracy: %f" % (results[-1]))
            elif architecture == "Ensemble":
                results.append(model.evaluate(statistics, ciphertexts, labels, args.batch_size, prediction_metrics, verbose=1))

            iteration = dataset.iteration - len(batch) * (len(batches) - index - 1)
            epoch = dataset.epoch
            if epoch > 0:
                epoch = iteration // (dataset.iteration // dataset.epoch)
            print("Epoch: %d, Iteration: %d" % (epoch, iteration))
            if iteration >= args.max_iter:
                break

        if dataset.iteration >= args.max_iter:
            break
    
    elapsed_evaluation_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished evaluation in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_evaluation_time.days, elapsed_evaluation_time.seconds // 3600, (elapsed_evaluation_time.seconds // 60) % 60,
        elapsed_evaluation_time.seconds % 60, iteration, epoch))

    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        avg_loss = 0
        avg_acc = 0
        avg_k3_acc = 0
        for loss, acc_pred, k3_acc in results:
            avg_loss += loss
            avg_acc += acc_pred
            avg_k3_acc += k3_acc
        avg_loss = avg_loss / len(results)
        avg_acc = avg_acc / len(results)
        avg_k3_acc = avg_k3_acc / len(results)
        print("Average evaluation results: loss: %f, accuracy: %f, k3_accuracy: %f\n" % (avg_loss, avg_acc, avg_k3_acc))
    elif architecture in ("DT", "NB", "RF", "ET", "Ensemble", "SVM", "kNN"):
        avg_test_acc = 0
        avg_k3_acc = 0
        for acc, k3_acc in results:
            avg_test_acc += acc
            avg_k3_acc += k3_acc
        avg_test_acc = avg_test_acc / len(results)
        avg_k3_acc = avg_k3_acc / len(results)
        print("Average evaluation results from %d iterations: avg_test_acc=%f, k3_accuracy: %f\n" % (iteration, avg_test_acc, avg_k3_acc))

        print("Detailed results:")
        prediction_metrics.print_evaluation()


def evaluate(args, model, architecture):
    results_list = []
    dir_name = os.listdir(args.data_folder)
    dir_name.sort()
    cntr = 0
    iterations = 0
    for name in dir_name:
        if iterations > args.max_iter:
            break
        path = os.path.join(args.data_folder, name)
        if os.path.isfile(path) and path.endswith(".txt"):
            if iterations > args.max_iter:
                break
            batch = []
            batch_ciphertexts = []
            labels = []
            results = []
            dataset_cnt = 0
            input_length = get_model_input_length(model, args.architecture)
            with open(path, "rb") as fd:
                lines = fd.readlines()
            for line in lines:
                # remove newline
                line = line.strip(b'\n').decode()
                if line == '':
                    continue
                split_line = line.split(' ')
                labels.append(int(split_line[0]))
                statistics = [float(f) for f in split_line[1].split(',')]
                batch.append(statistics)
                ciphertext = [int(j) for j in split_line[2].split(',')]
                if input_length is not None:
                    if len(ciphertext) < input_length:
                        ciphertext = pad_sequences([ciphertext], maxlen=input_length)[0]
                    # if the length its too high, we need to strip it..
                    elif len(ciphertext) > input_length:
                        ciphertext = ciphertext[:input_length]
                batch_ciphertexts.append(ciphertext)
                iterations += 1
                if iterations == args.max_iter:
                    break
                if len(labels) == args.dataset_size:
                    if architecture == "FFNN":
                        results.append(model.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(labels), args.batch_size, verbose=0))
                    elif architecture in ("CNN", "LSTM", "Transformer"):
                        results.append(model.evaluate(tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels),
                                       args.batch_size, verbose=0))
                    elif architecture == "Ensemble":
                        results.append(model.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels),
                                       args.batch_size, verbose=0))
                    elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                        results.append(model.score(batch, tf.convert_to_tensor(labels)))
                    batch = []
                    batch_ciphertexts = []
                    labels = []
                    dataset_cnt += 1
            if len(labels) > 0:
                if architecture == "FFNN":
                    results.append(model.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(labels), args.batch_size, verbose=0))
                elif architecture in ("CNN", "LSTM", "Transformer"):
                    results.append(
                        model.evaluate(tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels), args.batch_size, verbose=0))
                elif architecture == "Ensemble":
                    results.append(
                        model.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels),
                                        args.batch_size, verbose=0))
                elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                    results.append(model.score(batch, tf.convert_to_tensor(labels)))
            if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
                avg_loss = 0
                avg_acc = 0
                avg_k3_acc = 0
                for loss, acc_pred, k3_acc in results:
                    avg_loss += loss
                    avg_acc += acc_pred
                    avg_k3_acc += k3_acc
                result = [avg_loss / len(results), avg_acc / len(results), avg_k3_acc / len(results)]
            elif architecture in ("DT", "NB", "RF", "ET", "Ensemble", "SVM", "kNN"):
                avg_test_acc = 0
                for acc in results:
                    avg_test_acc += acc
                result = avg_test_acc / len(results)
            results_list.append(result)
            cntr += 1
            if args.evaluation_mode == 'per_file':
                if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
                    print("%s (%d lines) test_loss: %f, test_accuracy: %f, test_k3_accuracy: %f (progress: %d%%)" % (
                        os.path.basename(path), len(batch) + dataset_cnt * args.dataset_size, result[0], result[1], result[2], max(
                            int(cntr / len(dir_name) * 100), int(iterations / args.max_iter) * 100)))
                elif architecture in ("DT", "NB", "RF", "ET", "Ensemble", "SVM", "kNN"):
                    print("%s (%d lines) test_accuracy: %f (progress: %d%%)" % (
                        os.path.basename(path), len(batch) + dataset_cnt * args.dataset_size, result,
                        max(int(cntr / len(dir_name) * 100), int(iterations / args.max_iter) * 100)))
            else:
                print_progress("Evaluating files: ", cntr, len(dir_name), factor=5)
            if iterations == args.max_iter:
                break

    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        avg_test_loss = 0
        avg_test_acc = 0
        avg_test_acc_k3 = 0
        for loss, acc, acc_k3 in results_list:
            avg_test_loss += loss
            avg_test_acc += acc
            avg_test_acc_k3 += acc_k3
        avg_test_loss = avg_test_loss / len(results_list)
        avg_test_acc = avg_test_acc / len(results_list)
        avg_test_acc_k3 = avg_test_acc_k3 / len(results_list)
        print("\n\nAverage evaluation results from %d iterations: avg_test_loss=%f, avg_test_acc=%f, avg_test_acc_k3=%f" % (
            iterations, avg_test_loss, avg_test_acc, avg_test_acc_k3))
    elif architecture in ("DT", "NB", "RF", "ET", "Ensemble", "SVM", "kNN"):
        avg_test_acc = 0
        for acc in results_list:
            avg_test_acc += acc
        avg_test_acc = avg_test_acc / len(results_list)
        print("\n\nAverage evaluation results from %d iterations: avg_test_acc=%f" % (iterations, avg_test_acc))

def predict_single_line(args, model, architecture, backend):
    cipher_id_result = ''
    ciphertexts = []
    result = []
    if args.ciphertext is not None:
        ciphertexts.append(args.ciphertext.encode())
    else:
        ciphertexts = open(args.file, 'rb')

    print()
    for line in ciphertexts:
        # remove newline
        line = line.strip(b'\n')
        if line == b'':
            continue
        # evaluate aca features file
        # label = line.split(b' ')[0]
        # statistics = ast.literal_eval(line.split(b' ')[1].decode())
        # ciphertext = ast.literal_eval(line.split(b' ')[2].decode())
        # print(config.CIPHER_TYPES[int(label.decode())], "length: %d" % len(ciphertext))

        # Append ciphertext to itself. This improves the reliablity of the results.
        while len(line) < 1000:
            line = line + line
        # Limit line to at most 1000 characters to limit the execution time
        line = line[:1000]

        print(line)
        ciphertext = map_text_into_numberspace(line, OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER)
        try:
            statistics = calculate_statistics(ciphertext)
        except ZeroDivisionError:
            print("\n")
            continue
        results = None
        if architecture == "FFNN":
            result = predict_ffnn(model, [statistics], args.batch_size, backend)
        elif architecture == "LSTM":
            result = predict_lstm(model, ciphertext, args.batch_size, backend)
        elif architecture == "CNN":
            result = predict_cnn(model, ciphertext, args.batch_size)
        elif architecture == "Transformer":
            result = predict_transformer(model, ciphertext, args.batch_size)
        elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
            result = model.predict_proba(tf.convert_to_tensor([statistics]))
        elif architecture == "Ensemble":
            result = model.predict([statistics], [ciphertext], args.batch_size, verbose=0)

        if isinstance(result, list):
            result_list = list(result[0])
        else:
            result_list = result[0].tolist()
        if results is not None and architecture not in ('Ensemble', 'LSTM', 'Transformer', 'CNN'):
            for j in range(len(result_list)):
                result_list[j] /= len(results)
        if args.verbose:
            for cipher in args.ciphers:
                print("{:23s} {:f}%".format(cipher, result_list[config.CIPHER_TYPES.index(cipher)]*100))
            max_val = max(result_list)
            cipher = config.CIPHER_TYPES[result_list.index(max_val)]
        else:
            max_val = max(result_list)
            cipher = config.CIPHER_TYPES[result_list.index(max_val)]
            print("{:s} {:f}%".format(cipher, max_val * 100))
        print()
        cipher_id_result += cipher[0].upper()

    if args.file is not None:
        ciphertexts.close()

    # return a list of probabilities (does only return the last one in case a file is used)
    res_dict = {}
    if len(result) != 0:
        for j, val in enumerate(result[0]):
            if j >= len(args.ciphers):
                # Stop in cases where model is trained to predict more classes then are requested
                break
            res_dict[args.ciphers[j]] = val * 100
    return res_dict


def load_model(architecture, args, model_path, cipher_types):
    strategy = args.strategy
    model_list = args.models
    architecture_list = args.architectures
    
    model = None
    backend = None
    if model_path.endswith(".pth"):
        backend = Backend.PYTORCH
    elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
        backend = Backend.SCIKIT
    else:
        backend = Backend.KERAS

    is_feature_engineering_arch = architecture in ("FFNN", "DT", "NB", "RF", "ET", "SVM", "kNN")
    is_feature_learning_arch = architecture in ("CNN", "LSTM", "Transformer")

    if is_feature_engineering_arch:
        model_file = ModelFile(model_path, architecture, backend=backend)
        model = model_file.load_model()
        config.FEATURE_ENGINEERING = True
        config.PAD_INPUT = False
    elif is_feature_learning_arch:
        model_file = ModelFile(model_path, architecture, backend=backend)
        model = model_file.load_model()
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
    elif architecture == 'Ensemble':
        cipher_indices = []
        for cipher_type in cipher_types:
            cipher_indices.append(config.CIPHER_TYPES.index(cipher_type))
        model_files = []
        for i, model in enumerate(model_list):
            arch = architecture_list[i]
            if model.endswith(".pth"):
                model_backend = Backend.PYTORCH
            elif arch in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                model_backend = Backend.SCIKIT
            else:
                model_backend = Backend.KERAS
            file = ModelFile(model, arch, model_backend)
            model_files.append(file)
        model = EnsembleModel(model_files, strategy, cipher_indices)
    else:
        raise ValueError("Unknown architecture: %s" % architecture)
    
    # Check if there are rotor ciphers among those requested
    has_rotor_ciphers = any(c in config.ROTOR_CIPHER_TYPES for c in cipher_types)

    # If there are rotor ciphers, also load the rotor_only model.
    if has_rotor_ciphers:
        rotor_only_model_path = args.rotor_only_model
        if not os.path.exists(rotor_only_model_path):
            raise FileNotFoundError(f"Rotor-only model is required but not found at {rotor_only_model_path}")
        with open(rotor_only_model_path, "rb") as f:
            rotor_only_model = pickle.load(f)
        return RotorDifferentiationEnsemble(architecture, model, rotor_only_model), "Ensemble", Backend.KERAS

    # If there are no rotor ciphers:
    # - if it’s an ensemble, return the ensemble directly
    # - otherwise return the normal model
    return model, architecture, backend


def expand_cipher_groups(cipher_types):
    """Turn cipher group identifiers (ACA, MTC3) into a list of their ciphers"""
    expanded = cipher_types
    if config.MTC3 in expanded:
        del expanded[expanded.index(config.MTC3)]
        for i in range(5):
            expanded.append(config.CIPHER_TYPES[i])
    elif config.ACA in expanded:
        del expanded[expanded.index(config.ACA)]
        for i in range(56):
            expanded.append(config.CIPHER_TYPES[i])
    elif "aca+rotor" in expanded:
        del expanded[expanded.index("aca+rotor")]
        for i in range(61):
            expanded.append(config.CIPHER_TYPES[i])
    return expanded

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Evaluation Script', formatter_class=argparse.RawTextHelpFormatter)
    sp = parser.add_subparsers()
    bench_parser = sp.add_parser('benchmark',
                                 help='Use this argument to create ciphertexts on the fly, \nlike in training mode, and evaluate them with '
                                      'the model. \nThis option is optimized for large throughput to test the model.')
    eval_parser = sp.add_parser('evaluate', help='Use this argument to evaluate cipher types for single files or directories.')
    single_line_parser = sp.add_parser('single_line', help='Use this argument to predict a single line of ciphertext.')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--max_iter', default=1000000000, type=int,
                        help='the maximal number of iterations before stopping evaluation.')
    parser.add_argument('--model', default='../data/models/m1.h5', type=str,
                        help='Path to the model file. The file must have the .h5 extension.')
    parser.add_argument('--rotor_only_model', default='../data/models/svm_rotor_only.h5', type=str,
                        help='Path to rotor only model. This model is used in conjunction '
                        'with the normal model in an ensemble to improve the recogintion '
                        'of rotor ciphers. The file must have the .h5 extension.')
    parser.add_argument('--architecture', default='FFNN', type=str, choices=[
        'FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer', 'SVM', 'kNN', 'Ensemble'],
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
             '- Ensemble')
    parser.add_argument('--ciphers', '--ciphers', default='aca', type=str,
                        help='A comma seperated list of the ciphers to be created.\n'
                             'Be careful to not use spaces or use \' to define the string.\n'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere,\n'
                             '        Columnar Transposition, Plaifair and Hill)\n'
                             '- aca (contains all currently implemented ciphers from \n'
                             '       https://www.cryptogram.org/resource-area/cipher-types/)\n'
                             '- aca+rotor\n'
                             '- simple_substitution\n'
                             '- vigenere\n'
                             '- columnar_transposition\n'
                             '- playfair\n'
                             '- hill\n')

    parser.add_argument('--models', action='append', default=None,
                        help='A list of models to be used in the ensemble model. The length of the list must be the same like the one in '
                             'the --architectures argument.')
    parser.add_argument('--architectures', action='append', default=None,
                        help='A list of the architectures to be used in the ensemble model. The length of the list must be the same like '
                             'the one in the --models argument.')
    parser.add_argument('--strategy', default='weighted', type=str, choices=['mean', 'weighted'],
                        help='The algorithm used for decisions.\n- Mean voting adds the probabilities from every class and returns the mean'
                             ' value of it. The highest value wins.\n- Weighted voting uses pre-calculated statistics, like for example '
                             'precision, to weight the output of a specific model for a specific class.')
    parser.add_argument('--dataset_size', default=16000, type=int,
                        help='Dataset size per evaluation. This argument should be dividable \nby the amount of --ciphers.')

    bench_parser.add_argument('--download_dataset', default=True, type=str2bool)
    bench_parser.add_argument('--dataset_workers', default=1, type=int)
    bench_parser.add_argument('--plaintext_folder', default='../data/gutenberg_en', type=str)
    bench_parser.add_argument('--rotor_ciphertext_folder', default='../data/rotor_ciphertexts', type=str)
    bench_parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool)
    bench_parser.add_argument('--min_text_len', default=50, type=int)
    bench_parser.add_argument('--max_text_len', default=-1, type=int)

    bench_group = parser.add_argument_group('benchmark')
    bench_group.add_argument('--download_dataset', help='Download the dataset automatically.')
    bench_group.add_argument('--dataset_workers', help='The number of parallel workers for reading the input files.')
    bench_group.add_argument('--plaintext_folder', help='Input folder of the plaintexts.')
    bench_group.add_argument('--keep_unknown_symbols', help='Keep unknown symbols in the plaintexts. Known \n'
                                                            'symbols are defined in the alphabet of the cipher.')
    bench_group.add_argument('--min_text_len', help='The minimum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no lower limit is used.')
    bench_group.add_argument('--max_text_len', help='The maximum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no upper limit is used.')

    eval_parser.add_argument('--evaluation_mode', nargs='?', choices=('summarized', 'per_file'), default='summarized', type=str)
    eval_parser.add_argument('--data_folder', default='../data/gutenberg_en', type=str)

    eval_group = parser.add_argument_group('evaluate')
    eval_group.add_argument('--evaluation_mode',
                            help='- To create an single evaluation result over all iterated data files use the \'summarized\' option.'
                                 '\n  This option is to be preferred over the benchmark option, if the tests should be reproducable.\n'
                                 '- To create an evaluation for every file use \'per_file\' option. This mode allows the \n'
                                 '  calculation of the \n  - average value of the prediction \n'
                                 '  - lower quartile - value at the position of 25 percent of the sorted predictions\n'
                                 '  - median - value at the position of 50 percent of the sorted predictions\n'
                                 '  - upper quartile - value at the position of 75 percent of the sorted predictions\n'
                                 '  With these statistics an expert can classify a ciphertext document to a specific cipher.')
    eval_group.add_argument('--data_folder', help='Input folder of the data files with labels and calculated features.')

    single_line_parser.add_argument('--verbose', default=True, type=str2bool)
    data = single_line_parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--ciphertext', default=None, type=str)
    data.add_argument('--file', default=None, type=str)

    single_line_group = parser.add_argument_group('single_line')
    single_line_group.add_argument('--ciphertext', help='A single line of ciphertext to be predicted by the model.')
    single_line_group.add_argument('--file', help='A file with lines of ciphertext to be predicted line by line by the model.')
    single_line_group.add_argument('--verbose', help='If true all predicted ciphers are printed. \n'
                                                     'If false only the most accurate prediction is printed.')

    return parser.parse_args()

def main():
    multiprocessing.set_start_method("spawn")

    args = parse_arguments()

    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))
    m = os.path.splitext(args.model)
    if os.path.splitext(args.model)[1] not in ('.h5', '.pth'):
        print('ERROR: The model must have extension ".h5" (for Keras) or ".pth" (for PyTorch FFNN).', file=sys.stderr)
        sys.exit(1)

    architecture = args.architecture
    model_path = args.model
    args.ciphers = args.ciphers.lower()
    cipher_types = args.ciphers.split(',')
    args.ciphers = expand_cipher_groups(cipher_types)
    if architecture == 'Ensemble':
        if not hasattr(args, 'models') or not hasattr(args, 'architectures'):
            raise ValueError("Please use the 'ensemble' subroutine if specifying the ensemble architecture.")
        if len(args.models) != len(args.architectures):
            raise ValueError("The length of --models must be the same like the length of --architectures.")
        models = []
        for i in range(len(args.models)):
            model = args.models[i]
            arch = args.architectures[i]
            abs_path = os.path.abspath(model)
            if not os.path.exists(abs_path):
                raise ValueError("Model in %s does not exist." % abs_path)
            if arch not in ('FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer', 'SVM', 'kNN'):
                raise ValueError("Unallowed architecture %s" % arch)
            if arch in ('CNN', 'Transformer') and not abs_path.endswith('.h5'):
                raise ValueError("Model names of the types %s must have the .h5 extension." % ['CNN', 'Transformer'])
            if arch in ('FFNN', 'LSTM') and not (abs_path.endswith('.h5') or abs_path.endswith('.pth')):
                raise ValueError("Model names of the types %s must have the .h5 or .pth extension." % ['FFNN', 'LSTM'])
    elif args.models is not None or args.architectures is not None:
        raise ValueError("It is only allowed to use the --models and --architectures with the Ensemble architecture.")

    print("Loading Model...")
    # There are some problems regarding the loading of models on multiple GPU's.
    # gpu_count = len(tf.config.list_physical_devices('GPU'))
    # if gpu_count > 1:
    #     strat = tf.distribute.MirroredStrategy()
    #     with strat.scope():
    #         model = load_model()
    # else:
    #     model = load_model()
    model, architecture, backend = load_model(architecture, args, model_path, cipher_types)
    print("Model Loaded.")

    # the program was started as in benchmark mode.
    if args.download_dataset is not None:
        benchmark(args, model, architecture, backend)
    # the program was started in single_line mode.
    elif args.ciphertext is not None or args.file is not None:
        predict_single_line(args, model, architecture, backend)
    # the program was started in prediction mode.
    else:
        evaluate(args, model, architecture)

if __name__ == "__main__":
    main()
