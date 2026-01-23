from itertools import groupby
from pathlib import Path
import tensorflow as tf
import cipherTypeDetection.config as config
from cipherImplementations.simpleSubstitution import SimpleSubstitution
import sys
from cipherTypeDetection.trainingBatch import EvaluationBatch, TrainingBatch
from util.utils import map_text_into_numberspace
import copy
import random
import multiprocessing
from multiprocessing import pool as multiprocessing_pool
import logging
import numpy as np
from collections import deque
from cipherTypeDetection.featureCalculations import calculate_statistics
sys.path.append("../")


def encrypt(plaintext, label, key_length, keep_unknown_symbols, return_key=False):
    cipher = config.CIPHER_IMPLEMENTATIONS[label]
    plaintext = cipher.filter(plaintext, keep_unknown_symbols)
    if cipher.needs_plaintext_of_specific_length:
        plaintext = cipher.truncate_plaintext(plaintext, key_length)
    key = cipher.generate_random_key(key_length)
    if return_key:
        orig_key = copy.deepcopy(key)
    plaintext_numberspace = map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
    if isinstance(key, bytes):
        key = map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], bytes) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], int):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], int) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], (list, np.ndarray)) and (len(key[0]) == 5 or len(
            key[0]) == 10) and isinstance(key[1], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], list) and isinstance(key[1], np.ndarray) and isinstance(
            key[2], bytes):
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, dict):
        new_key_dict = {}
        for k in key:
            new_key_dict[cipher.alphabet.index(k)] = key[k]
        key = new_key_dict

    ciphertext = cipher.encrypt(plaintext_numberspace, key)
    if b'j' not in cipher.alphabet and config.CIPHER_TYPES[label] != 'homophonic':
        ciphertext = normalize_text(ciphertext, 9)
    if b'x' not in cipher.alphabet:
        ciphertext = normalize_text(ciphertext, 23)
    if return_key:
        return ciphertext, orig_key
    return ciphertext


def normalize_text(text, pos):
    for i in range(len(text)):
        if 26 >= text[i] >= pos:
            text[i] += 1
    return text

def pad_sequences(sequences, maxlen):
    """Pad sequences with data from itself."""
    ret_sequences = []
    for sequence in sequences:
        length = len(sequence)
        sequence = sequence * (maxlen // length) + sequence[:maxlen % length]
        ret_sequences.append(sequence)
    return np.array(ret_sequences)


multiprocessing_logger = multiprocessing.log_to_stderr(logging.INFO)

class CipherStatisticsDataset:
    """This class takes inputs for the composed `PlaintextPathsDataset` and 
    `RotorCiphertextsDataset`, does some processing (e.g. filtering of the characters and
    encryption) and finally calculate the statistics for the inputs. 
    The class provides a iterator interface which returns the calculated statistics and their 
    labels as `TrainingBatch`es. The number of training batches that are returned by each iteration 
    depend on `dataset_workers`. Each `TrainingBatch` contains labels and statistics for both kinds
    of underlying datasets (RotorCiphertextsDataset and PlaintextPathsDataset).
    To process the inputs, `dataset_workers` processes are used to calculate the statistics
    in parallel. The processes are organized in process pools to keep the overhead for small
    `batch_size`es small. To limit the time each iteration takes, before returning the current
    iterations results, new processes for the next iteration are already started and moved into
    the `_processing_queue`.
    """

    def __init__(self, plaintext_paths, plaintext_dataset_params, rotor_ciphertext_paths, 
                 rotor_ciphertext_dataset_params, *, generate_evaluation_data=False):
        assert plaintext_dataset_params.dataset_workers == rotor_ciphertext_dataset_params.dataset_workers

        dataset_workers = plaintext_dataset_params.dataset_workers

        self._iteration = 0
        self._epoch = 0
        self._dataset_workers = dataset_workers
        
        self._pool = multiprocessing_pool.Pool(self._dataset_workers)
        # double ended queue for storing asynchronously processing functions
        self._processing_queue = deque() 
        self._logger = multiprocessing_logger

        self._plaintext_paths = plaintext_paths
        self._plaintext_dataset_params = plaintext_dataset_params
        self._rotor_ciphertext_paths = rotor_ciphertext_paths
        self._rotor_ciphertext_dataset_params = rotor_ciphertext_dataset_params

        self._initialize_datasets()

        self._generate_evaluation_data = generate_evaluation_data
    
    def _initialize_datasets(self):
        self._plaintext_dataset = PlaintextPathsDataset(self._plaintext_paths, self._plaintext_dataset_params, 
                                                        self._logger)
        self._ciphertext_dataset = RotorCiphertextsDataset(self._rotor_ciphertext_paths, 
                                                           self._rotor_ciphertext_dataset_params, 
                                                           self._logger)
    @property
    def both_datasets_initialized(self):
        """Inidcates whether both ciphertext and plaintext datasets are initialized.
        Useful for mixing the different results of the workers in `next`.
        """
        return self._ciphertext_dataset.is_initialized and self._plaintext_dataset.is_initialized
    
    @property
    def iteration(self):
        """The iteration corresponds to the number of lines processed by the dataset."""
        return self._iteration
    
    @property
    def epoch(self):
        """Each epoch represents the processing of all available inputs. If the epoch is 
        increased, the dataset will restart iterating it's inputs from the beginning."""
        return self._epoch
    
    @property
    def dataset_workers(self):
        """The number of parallel workers to use when calculating the statistics of the input. 
        This number also equals the number of `TrainingBatch`es returned by `__next__`."""
        return self._dataset_workers
    
    @property
    def batch_size(self):
        """The amount of statistics and labels in the returned `TrainingBatch`es of 
        iterator method `__next__`."""
        return (self._plaintext_dataset_params.batch_size + 
                self._rotor_ciphertext_dataset_params.batch_size)
    
    @property
    def key_lengths_count(self):
        """Returns the combined count of all key length values for all supported ciphers
        of the plaintext path dataset. See also `config.KEY_LENGTHS`.
        """
        return self._plaintext_dataset.key_lengths_count
    
    def stop_outstanding_tasks(self):
        self._pool.terminate()
        self._processing_queue = deque()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        config_params = ConfigParams(config.CIPHER_TYPES, config.KEY_LENGTHS, 
                                     config.FEATURE_ENGINEERING, config.PAD_INPUT)
        
        # init epoch on first iteration
        if self._epoch == 0:
            self._epoch = 1

        ciphertext_inputs_exhausted = False
        plaintext_inputs_exhausted = False

        # process rotor cipher datasets
        ciphertext_worker = CiphertextLine2CipherStatisticsWorker(
            self._rotor_ciphertext_dataset_params, config_params)
        plaintext_worker = PlaintextLine2CipherStatisticsWorker(
            self._plaintext_dataset_params, config_params)

        # Number of workers to start to yield a single combined batch of rotor and aca 
        # ciphers.
        combined_process_count = self._dataset_workers * 2

        # Process both kinds of ciphers at once. Queue `combined_process_count * 2` processes, 
        # to ensure that the next `TrainingBatch`es are prepared while the models in `train.py` 
        # are trained. This ensures that the wait times for each iteration of the dataset are 
        # shorter then if we only start preprocessing at the begining of __next__ calls.
        # (Since the pool is initialized with `_dataset_workers`, the number of processes is
        # still kept at `_dataset_workers`.)
        ciphertext_inputs_exhausted, plaintext_inputs_exhausted = self._dispatch_concurrent(
            ciphertext_worker=ciphertext_worker, ciphertext_dataset=self._ciphertext_dataset, 
            plaintext_worker=plaintext_worker, plaintext_dataset=self._plaintext_dataset,
            number_of_processes=combined_process_count * 2)
        
        # Inputs exhausted: Increase epoch, re-initialize datasets and therefore begin 
        # iteration from the start.
        if ciphertext_inputs_exhausted or plaintext_inputs_exhausted:
            exhausted_dataset_type = "Ciphertexts" if ciphertext_inputs_exhausted else "Plaintext"
            print(f"CipherStatisticsDataset: {exhausted_dataset_type} of epoch {self._epoch} exhausted! Resetting iterators!")
            self._epoch += 1
            self._initialize_datasets()

        # Get all results of this iteration
        all_results = self._wait_for_results(number_of_processes=combined_process_count)

        assert len(all_results) > 0, "Results are empty. Were the datasets initialized?"

        # If a dataset is not initalized: No need to mix ciphertext and plaintext results. 
        # Simply return all results of this iteration.
        if not self.both_datasets_initialized:
            return all_results

        # Wait until the workers of both cipher types have finished. Otherwise the returned
        # batch could only contain aca or rotor ciphers.
        while TrainingBatch.represent_equal_cipher_type(all_results):
            assert len(self._processing_queue) > 0, "Expected different cipher type in queue!"
            next_results = self._wait_for_results(number_of_processes=combined_process_count)
            all_results.extend(next_results)

        # Combine ACA and rotor cipher training batches. Each batch should contain some 
        # statistics and labels of both.
        if self._generate_evaluation_data:
            paired_cipher_types = EvaluationBatch.paired_cipher_types(all_results)
            return [EvaluationBatch.combined(pair) for pair in paired_cipher_types] 
        else:
            paired_cipher_types = TrainingBatch.paired_cipher_types(all_results)
            return [TrainingBatch.combined(pair) for pair in paired_cipher_types] 

    def _dispatch_concurrent(self, *, ciphertext_worker, ciphertext_dataset, plaintext_worker, 
                             plaintext_dataset, number_of_processes):
        error_callback = lambda error: print(f"ERROR in ParallelIterator: {error}")

        ciphertext_inputs_exhausted = False
        plaintext_inputs_exhausted = False

        worker = None
        input_batch = None

        for index in range(number_of_processes):
            # start rotor and plaintext worker alternatly after one another
            if index % 2 == 0:
                try:
                    if ciphertext_dataset.is_initialized:
                        worker = ciphertext_worker
                        input_batch = next(ciphertext_dataset)
                except StopIteration:
                    ciphertext_inputs_exhausted = True
                    continue
            else:
                try:
                    if plaintext_dataset.is_initialized:
                        worker = plaintext_worker
                        input_batch = next(plaintext_dataset)
                except StopIteration:
                    plaintext_inputs_exhausted = True
                    continue

            if worker and input_batch:
                batch = self._pool.apply_async(worker.perform, 
                                                (input_batch, ),
                                                error_callback=error_callback)
                self._processing_queue.append(batch)
        
        return (ciphertext_inputs_exhausted, plaintext_inputs_exhausted)

    def _wait_for_results(self, number_of_processes):
        training_batches = []
        for _ in range(number_of_processes):
            try:
                result = self._processing_queue.popleft()
            except IndexError:
                continue
            training_batch = result.get()
            self._iteration += len(training_batch)
            training_batches.append(training_batch)

        return training_batches

class RotorCiphertextsDatasetParameters:
    """Encapsulates the parameters of `RotorCiphertextsDataset`. These parameters are used
    for the initialization of the dataset itself, as well as for the worker processes and
    the main statistics dataset."""

    def __init__(self, cipher_types, batch_size, dataset_workers, 
                 min_text_len, max_text_len, generate_evalutation_data):
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.dataset_workers = dataset_workers
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.generate_evaluation_data = generate_evalutation_data
    
class RotorCiphertextsDataset:
    """Takes paths to the ciphertext files as input and returns batched
    lists of the ciphertext and label pairs as iteration result.
    The inputs are converted to match the `max_text_len` parameter of 
    `RotorCiphertextsDatasetParameters` and rearranged to alternate
    the samples of the different cipher types.
    The batching allows for splitting the input in workable chunks 
    (that fit in RAM) and can be distributed to multiple worker processes."""
    def __init__(self, ciphertext_paths, dataset_params, logger):
        try:
            ciphertexts = {}
            for path in ciphertext_paths:
                file_name = Path(path).stem.lower()
                ciphertexts[file_name] = open(path, "r")

            self._ciphertext_files = ciphertexts
            
            self._number_of_cipher_classes = len(self._ciphertext_files)
            self._min_text_len = dataset_params.min_text_len
            self._max_text_len = dataset_params.max_text_len

            self._ciphertext_buffer = []
            self._buffer_size = 4000
            self._fill_buffer()

            self._index = 0

            self._batch_size = dataset_params.batch_size
            self._logger = logger
        except Exception as e:
            self._cleanup()
            raise e
    
    def __del__(self):
        self._cleanup()
    
    def _cleanup(self):
        for file in self._ciphertext_files.values():
            file.close()

    def _rearrange_ciphertexts(self, ciphertext_label_pairs):
        """Rearranges the pairs of ciphertexts and labels so that each
        label is alternated in the result."""
        get_label = lambda element: element[1]
        # Sort pairs by label
        ciphertext_label_pairs = sorted(ciphertext_label_pairs, key=get_label)

        # Group by label
        grouped = []
        for _, group in groupby(ciphertext_label_pairs, key=get_label):
            grouped.append(list(group))

        # Alternatingly use each entry of each group
        rearranged = []
        for group in zip(*grouped):
            rearranged.extend(group)

        return rearranged
    
    @property
    def is_initialized(self):
        """Indicates whether the dataset was initialized with input data."""
        return self._number_of_cipher_classes != 0 and self._batch_size > 0
    
    def _are_files_closed(self):
        """Returns True if at least one ciphertext file is closed."""
        for file in self._ciphertext_files.values():
            if file.closed:
                return True
        return False
    
    def _fill_buffer(self):
        """Loads `self._buffer_size` lines from the `self._ciphertext_files`
        into the buffer. Returns True if buffer was completetly filled, False otherwise."""
        if self._number_of_cipher_classes == 0 or self._are_files_closed():
            return False
        
        reached_end = False
        for __ in range(self._buffer_size // self._number_of_cipher_classes):
            for cipher, file in self._ciphertext_files.items():
                ciphertext = file.readline().strip()
                if len(ciphertext) == 0:
                    reached_end = True
                    break

                # Use random length substrings of `ciphertext` as samples for 
                # the buffer until `ciphertext` is too short to be splitted again.
                while len(ciphertext) >= self._min_text_len:
                    random_length = random.randint(self._min_text_len, self._max_text_len)
                    truncated_ciphertext = ciphertext[:random_length]
                    self._ciphertext_buffer.append((truncated_ciphertext, cipher))
                    ciphertext = ciphertext[random_length:]

        random.shuffle(self._ciphertext_buffer)
        self._ciphertext_buffer = self._rearrange_ciphertexts(self._ciphertext_buffer)

        return not reached_end

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            input_exhausted = False
            if len(self._ciphertext_buffer) < self._batch_size:
                input_exhausted = not self._fill_buffer()
            
            if len(self._ciphertext_buffer) == 0 and input_exhausted:
                raise StopIteration()

            result = self._ciphertext_buffer[:self._batch_size]
            self._ciphertext_buffer = self._ciphertext_buffer[self._batch_size:]

            self._index += 1
            self._logger.info(f"RotorCiphertextDataset: Returning batch {self._index}")

            return result
        except Exception as e:
            self._cleanup()
            raise e
    
class PlaintextPathsDatasetParameters:
    """Encapsulates the parameters of `PlaintextPathsDataset`. These parameters are used
    for the initialization of the dataset itself, as well as for the worker processes and
    the main statistics dataset."""

    def __init__(self, cipher_types, batch_size, min_text_len, max_text_len, 
                 keep_unknown_symbols=False, dataset_workers=None,
                 generate_evaluation_data=False):
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.keep_unknown_symbols = keep_unknown_symbols
        self.dataset_workers = dataset_workers
        self.generate_evaluation_data = generate_evaluation_data
    
class PlaintextPathsDataset:
    """Takes paths to plaintexts and returns list of size `batch_size` with lines
    from the plaintext files."""

    def __init__(self, plaintext_paths, dataset_params, logger):
        self._batch_size = dataset_params.batch_size
        self._min_text_len = dataset_params.min_text_len
        self._max_text_len = dataset_params.max_text_len
        self._keep_unknown_symbols = dataset_params.keep_unknown_symbols
        self._logger = logger

        self._cipher_types = dataset_params.cipher_types

        key_lengths_count = 0
        for cipher_t in self._cipher_types:
            index = config.CIPHER_TYPES.index(cipher_t)
            if isinstance(config.KEY_LENGTHS[index], list):
                key_lengths_count += len(config.KEY_LENGTHS[index])
            else:
                key_lengths_count += 1
        self._key_lengths_count = key_lengths_count

        self._plaintext_paths = plaintext_paths
        self._plaintext_dataset = tf.data.TextLineDataset(plaintext_paths)
        self._dataset_iter = self._plaintext_dataset.__iter__()

        self._index = 0
    
    @property
    def key_lengths_count(self):
        return self._key_lengths_count
    
    @property
    def is_initialized(self):
        """Indicates whether the dataset was initialized with input data."""
        return len(self._plaintext_paths) != 0 and len(self._cipher_types) != 0 and self._batch_size > 0

    def __iter__(self):
        return self

    def __next__(self):
        c = SimpleSubstitution(config.INPUT_ALPHABET, config.UNKNOWN_SYMBOL, config.UNKNOWN_SYMBOL_NUMBER)

        result = []
        number_of_lines = self._batch_size // self._key_lengths_count
        if number_of_lines == 0:
            print(f"ERROR: Batch size is too small to calculate the features for all cipher and key"
                  f"length combinations! Current batch size: {self._batch_size}. Minimum batch size: "
                  f"{self._key_lengths_count}.")
            raise StopIteration

        for _ in range(number_of_lines):
            # use the basic prefilter to get the most accurate text length
            filtered_data = c.filter(next(self._dataset_iter).numpy(), self._keep_unknown_symbols)
            # Select a random min length of the filtered_data to provide more variance in the 
            # resulting text length
            random_min_length = random.randint(self._min_text_len, self._max_text_len)
            while len(filtered_data) < random_min_length:
                # add the new data to the existing to speed up the searching process.
                filtered_data += c.filter(next(self._dataset_iter).numpy(), self._keep_unknown_symbols)
            if len(filtered_data) > self._max_text_len:
                result.append(filtered_data[:self._max_text_len-(self._max_text_len % 2)])
            else:
                result.append(filtered_data[:len(filtered_data)-(len(filtered_data) % 2)])
        
        self._logger.info(f"PlaintextPathsDataset: Returning batch {self._index}")
        self._index += 1

        return result

class ConfigParams:
    """Encapsulates some entries of the `config.py`. This removes the calls to global state
    from the workers. Should help to reason about the code, especially since the workers
    are typically executed in a concurrent process"""
    def __init__(self, cipher_types, key_lengths, feature_engineering, pad_input):
        # corresponds to config.CIPHER_TYPES
        self.cipher_types = copy.deepcopy(cipher_types)
        # corresponds to config.KEY_LENGTHS
        self.key_lengths = copy.deepcopy(key_lengths)
        # corresponds to config.FEATURE_ENGINEERING
        self.feature_engineering = copy.deepcopy(feature_engineering)
        #corresponds to config.PAD_INPUT
        self.pad_input = copy.deepcopy(pad_input)

class CiphertextLine2CipherStatisticsWorker:
    """This class provides an iterator that returns `TrainingBatch`es.
    It takes ciphertext lines and their corresponding labels (cipher names) and 
    calculates the statistics (features) for those lines.
    The size of the batches depends on the `batch_size` and `dataset_workers`. 
    Each `dataset_worker` will calculate the statistics for `batch_size` lines of the
    input. Therefore the output of the __next__ method will return lists of length 
    `dataset_workers`."""

    def __init__(self, dataset_params, config_params):
        self._max_text_len = dataset_params.max_text_len
        self._generate_evaluation_data = dataset_params.generate_evaluation_data
        self._config_params = config_params

    def perform(self, ciphertexts_with_labels):
        features = []
        labels = []

        config = self._config_params
        test_data = []

        for ciphertext_line, label in ciphertexts_with_labels:
            processed_line = self._preprocess_ciphertext_line(ciphertext_line)
            if config.feature_engineering:
                try:
                    feature = calculate_statistics(processed_line)
                    features.append(feature)
                except Exception as e:
                    multiprocessing_logger.error("Error occured while calculating statistics "
                                                 "for ciphertext. Skipping line...")
                    continue
            else:
                features.append(processed_line)
            if self._generate_evaluation_data:
                test_data.append(processed_line)
            label_index = config.cipher_types.index(label)
            labels.append(label_index)
            
        if config.pad_input and len(features) != 0:
            features = pad_sequences(features, maxlen=self._max_text_len)
            features = features.reshape(features.shape[0], features.shape[1], 1)
        
        if self._generate_evaluation_data:
            return EvaluationBatch("rotor", features, labels, test_data)
        else:
            return TrainingBatch("rotor", features, labels)
    
    def _preprocess_ciphertext_line(self, ciphertext_line):
        cleaned = ciphertext_line.strip().replace(' ', '').replace('\n', '')
        mapped = self._map_text_into_numberspace(cleaned.lower())
        return mapped
    
    def _map_text_into_numberspace(self, text):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        result = []
        for index in range(len(text)):
            try:
                result.append(alphabet.index(text[index]))
            except ValueError:
                raise Exception(f"Ciphertext contains unknown character '{text[index]}'. "
                                f"Known characters are: '{alphabet}'.")
        return result

class PlaintextLine2CipherStatisticsWorker:
    """This class takes paths to plaintext files and provides an iterator 
    interface, which will return batches of statistics and labels for training
    of a classifier. Internally it will encrypt the lines of the plaintext files
    with the given `cipher_types` and calculate the features for the encrypted 
    lines.
    Each `dataset_worker` will calculate the statistics for `batch_size` lines of the
    encrypted lines. Therefore the output of the __next__ method will return lists of length 
    `dataset_workers`. Each item in the list is of type `TrainingBatch`.
    """
    def __init__(self, dataset_params, config_params):
        self._keep_unknown_symbols = dataset_params.keep_unknown_symbols
        self._cipher_types = dataset_params.cipher_types
        self._max_text_len = dataset_params.max_text_len
        self._generate_evaluation_data = dataset_params.generate_evaluation_data
        self._config_params = config_params

    def perform(self, plaintexts):
        batch = []
        labels = []

        config = self._config_params
        ciphertexts = []

        for line in plaintexts:
            for cipher_type in self._cipher_types:
                label = config.cipher_types.index(cipher_type)
                if isinstance(config.key_lengths[label], list):
                    key_lengths = config.key_lengths[label]
                else:
                    key_lengths = [config.key_lengths[label]]
                for key_length in key_lengths:
                    try:
                        ciphertext = encrypt(line, label, key_length, self._keep_unknown_symbols)
                    except:
                        multiprocessing_logger.error(f"Could not encrypt line with cipher "
                                                     f"'{cipher_type}'. and key length {key_length}. "
                                                     f"Skipping line...")
                        continue
                    if config.feature_engineering:
                        try:
                            statistics = calculate_statistics(ciphertext)
                            batch.append(statistics)
                        except Exception as e:
                            multiprocessing_logger.error("Error occured while calculating statistics "
                                                         "for ciphertext. Skipping line...")
                            continue
                    else:
                        batch.append(list(ciphertext))
                    if self._generate_evaluation_data:
                        ciphertexts.append(ciphertext)
                    labels.append(label)

        if config.pad_input:
            batch = pad_sequences(batch, maxlen=self._max_text_len)
            batch = batch.reshape(batch.shape[0], batch.shape[1], 1)

        # multiprocessing_logger.info(f"Batch: '{batch}'; labels: '{labels}'.")
        if self._generate_evaluation_data:
            return EvaluationBatch("aca", batch, labels, ciphertexts)
        else:
            return TrainingBatch("aca", batch, labels)
