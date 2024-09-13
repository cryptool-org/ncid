# ncid

A neural network to detect and analyze ciphers from historical texts.

# NCID - Neural Cipher IDentifier

This project contains code for training machine learning models for the detection and classification of classical ciphers. CrypTool-Online provides a webapp that uses the trained models to classify ciphertexts: https://www.cryptool.org/cto/ncid.

While the project was focused on ACA ciphers at first, a later extension added the possibility to detect the rotor ciphers Enigma, M209, Purple, Sigaba, and Typex.

Users that are experienced in machine learning can use the tools provided in this project to train and evaluate ML models using the `train.py` and `eval.py` scripts. For further information see the following sections *Training* and *Evaluation*. 

# License

This software and the online version on https://www.cryptool.org/cto/ncid are licensed with the GPLv3 license. Private use of this software is allowed. Software using parts of the code from this repository must not be commercially used and also must be GPLv3 licensed.

Publications on websites and the like MUST be explicitly allowed by the author. For further information contact me at leierzopf@ins.jku.at.

# Installation

- Clone this repository and enter it:
  ```Shell
  git clone git@github.com:cryptool-org/ncid.git
  cd ncid
  ```

- Make sure to use at least Python 3.11.

- Install the recommended and tested libraries by using requirements.txt:

  ```
  pip3 install -r requirements.txt
  ```

# Training

By default we train the models to identify ACA ciphers listed [here](https://www.cryptogram.org/resource-area/cipher-types/), as well as 5 rotor ciphers: Enigma, M209, Purple, Sigaba, and Typex. The plaintexts used are already filtered and automatically downloaded in the train.py or eval.py scripts.  You can turn off this behavior by setting `--download_dataset=False`. For more information about handling of custom datasets see: *Data Preparation*. The rotor ciphers need pre-generated ciphertexts to work. These can be generated with CrypTool 2. To limit the cipher types to train, the option `--ciphers` can e.g. be set to `aca`, `rotor` or `all`.

To see all options of `train.py`, run the `--help` or `-h` command.

```
python3 train.py --help
```

## Example Commands

- ```
  python3 train.py --dataset_workers=50 --train_dataset_size=64960 --batch_size=512 --max_iter=1000000000 --min_train_len=100 --max_train_len=100 --min_test_len=100 --max_test_len=100 --model_name=t30.h5 > weights/t30.txt 2> weights/err_t30.txt &
  ```

- ```
  python3 train.py --model_name=mtc3_model.h5 --ciphers=mtc3  # first config.py must be adapted
  ```

- ```
  python3 train.py --architecture=FFNN --dataset_workers=50 --train_dataset_size=64960 --batch_size=512 --max_iter=1000000000 --min_train_len=100 --max_train_len=100 --min_test_len=100 --max_test_len=100 --model_name=t30.h5 > weights/t30.txt 2> weights/err_t30.txt &
  ```


## Multi-GPU Support

NCID now supports multiple GPUs seamlessly during training:

Before running any of the scripts, run: `export CUDA_VISIBLE_DEVICES=[gpus]`

- Where you should replace [gpus] with a comma separated list of the index of each GPU you want to use (e.g., 0,1,2,3).
- You should still do this if only using 1 GPU.
- You can check the indices of your GPUs with `nvidia-smi`.

# Evaluation

There are multiple ways to evaluate the models. First of all it is needed to put the corresponding model file in the `../data/models` directory and run one of the following commands:

- **benchmark** - Use this argument to create ciphertexts on the fly, like in training mode, and evaluate them with the model. This option is optimized for large throughput to test the model. Example usages:

  ```
  python3 eval.py --model=../data/models/t128_ffnn_final_100.h5 --max_iter=1000000 --dataset_size=1120 benchmark --dataset_workers=10 --min_text_len=100 --max_text_len=100
  ```

  ```
  python3 eval.py --architecture=Ensemble --models=../data/models/t128_ffnn_final_100.h5 --models=../data/models/t129_lstm_final_100.h5 --models=../data/models/t128_nb_final_100.h5 --models=../data/models/t99_rf_final_100.h5 --models=../data/models/t96_transformer_final_100.h5 --architectures=FFNN --architectures=LSTM --architectures=NB --architectures=RF --architectures=Transformer --strategy=weighted --batch_size=512 --max_iter=1000000 --dataset_size=64960 benchmark --dataset_workers=10 --min_text_len=100 --max_text_len=100 > ../data/benchmark.txt 2> ../data/err_benchmark.txt
  ```

- **evaluate** - Use this argument to evaluate cipher types for directories with ciphertext files in it. There are two *evaluation_modes*: 

  - *per_file* - every file is evaluated on it's own. The average of all evaluations of that file is the output. 

  - *summarized* - all files are evaluated and the average is printed at the end of the script. This mode is like *benchmark*, but is more reproducible and comparable, because the inputs are consistent.

  Example usages:

  ```
  python3 eval.py --model=../data/models/t128_ffnn_final_100.h5 --max_iter=100000 evaluate --evaluation_mode=per_file --ciphertext_folder=../data/generated_data
  ```

  ```
  python3 eval.py --architecture=Ensemble --models=../data/models/t128_ffnn_final_100.h5 --models=../data/models/t129_lstm_final_100.h5 --models=../data/models/t128_nb_final_100.h5 --models=../data/models/t99_rf_final_100.h5 --models=../data/models/t96_transformer_final_100.h5 --architectures=FFNN --architectures=LSTM --architectures=NB --architectures=RF --architectures=Transformer --strategy=weighted --batch_size=512 --dataset_size=64960 --max_iter=10000000 evaluate --data_folder=../data/generated_data --evaluation_mode=per_file > ../data/eval.txt 2> ../data/err_eval.txt
  ```

- **single_line** - Use this argument to predict a single line of ciphertext. The difference of this command is, that in contrast to the other modes, the results are predicted without knowledge of the real cipher type. There are two types of data this command can process:

  - *ciphertext* - A single line of ciphertext to be predicted by the model. Example usages:

  ```
  python3 eval.py --model=../data/models/t128_ffnn_final_100.h5 single_line --ciphertext=yingraobhoarhthosortmkicwhaslcbpirpocuedcfthcezvoryyrsrdyaffcleaetiaaeuhtyegeadsneanmatedbtrdndres
  ```

  ```
  python3 eval.py --architecture=Ensemble --models=../data/models/t128_ffnn_final_100.h5 --models=../data/models/t129_lstm_final_100.h5 --models=../data/models/t128_nb_final_100.h5 --models=../data/models/t99_rf_final_100.h5 --models=../data/models/t96_transformer_final_100.h5 --architectures=FFNN --architectures=LSTM --architectures=NB --architectures=RF --architectures=Transformer --strategy=weighted --batch_size=512 single_line --file=../data/generated_data/aca_features.txt --verbose=False > weights/../data/predict.txt 2> weights/err_predict.txt
  ```

  - *file* - A file with mixed lines of ciphertext to be predicted line by line by the model. Example usages:

  ```
  python3 eval.py --model=../data/models/t128_ffnn_final_100.h5 single_line --verbose=false --file=../data/cipherstexts.txt
  ```

To see all options of `eval.py`, run the `--help` or `-h` command.

```
python3 eval.py --help
```

# Data Preparation

Note: The following steps are only needed when the automatic downloading and preprocessing steps of the `train.py` or `eval.py` scripts fail or other data sources are wanted.

## Generate Plaintexts (Optional)

First of all this usage is not recommended as first option. Try running the `train.py` or `eval.py` script with the argument `--download_dataset=True`, if you only want to train or test on the filtered dataset. Optionally you can download the dataset on your own from [here](https://drive.google.com/open?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V)

If you'd like to create your own plaintexts, you can use the `generatePlainTextFiles.py` script. Therefore you first need to download some texts, for example the Gutenberg Library. You can do that by using following command, which downloads all English e-books compressed with zip.  Note that this script can take a while and dumps about 14gb of files into `./data/gutenberg_en` and 5.3gb additionaly if you do not delete the `gutenberg_en.zip`.

```shell
wget -m -H -nd "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en" > /tmp/wget-log 2>&1
```

The `generatePlainTextFiles.py` script automatically unpacks the zips, with the parameter `--restructure_directory`.  Every line in a plaintext is seperated by a '\n', so be sure to save it in the right format or use the `generatePlainTextFiles.py` script to reformat all files from '\r\n' to '\n'. For further description read the help by using the `--help` parameter. Example usage:

```
python3 generatePlainTextFiles.py --directory=../gutenberg_en --restructure_directory=true
```

## Generate ACA Ciphertexts (Optional)

You might want to predict with one or more models by using the same ciphertext files. The `generateCipherTextFiles.py` script encrypts plaintext files to multiple ciphertext files. The naming convention is *fileName-cipherType-minLenXXX-maxLenXXX-keyLenXX.txt*. This script generates ciphertexts out of plaintexts. **Beware:** Currently only ACA ciphertexts can be generated. To generate ciphertexts for rotor ciphers, an external tool (like CrypTool 2) has to be used. 

If a line is not long enough it is concatenated with the next line. If a line is too long it is sliced into max_text_len length. For further description read the help by using the `--help` parameter. Example usage:

```
python3 generateCipherTextFiles.py --min_text_len=100 --max_text_len=100 --max_files_count=100
```

## Generate Calculated Features (Optional)

To evaluate multiple models in the most comparable way, the features and ciphertexts are precalculated and saved into files using the `generateCalculatedFeatures.py` script.

```
python3 generateCalculatedFeatures.py --dataset_workers=50 --min_len=100 --max_len=100 --save_directory=../data/generated_data --batch_size=512 --dataset_size=64960 --max_iter=10000000 > ../data/generate_data.txt 2> ../data/err_generate_data.txt
```

# Unit-Tests

Multiple Unit-Tests ensure the functionality of the implemented ciphers and the TextLine2CipherStatisticsDataset. 

Every test case can be executed by using following command in the main directory:

```
python3 -m unittest discover -s unit -p '*Test.py'
```

Single test classes can be executed with this command:

  ```
python3 -m unittest <path/to/test/class>
  ```

for example:

  ```
python3 -m unittest unit/cipherTypeDetection/textLine2CipherStatisticsDataset.py
  ```

# Qualitative Results

## Original models trained for recognition of ACA ciphers

Following are our training results from a DGX-1 with 2 GPUs on the models with length 100 and 6 GPUs on models with length 51-428. Models are differentiated into feature-engineering (FFNN, RF and NB) and feature-extracting (LSTM and Transformer) models. Models are evaluated with a dataset of 10 million self generated records.

| Model Name                    | Accuracy in % | Iterations in Mio. | Training Time |
| :---------------------------- | :-----------: | :----------------: | :-----------: |
| t128_ffnn_final_100           |     78.31     |        181         |  7d 11h 14m   |
| t96_transformer_final_100     |     72.33     |        303         |   5d 8h 58m   |
| t99_rf_final_100              |     73.50     |        2.5         |    3h 24m     |
| t128_nb_final_100             |     52.79     |        181         |  7d 11h 14m   |
| t129_lstm_final_100           |     72.16     |        162         |  2d 21h 31m   |
| ensemble_mean_100             |     82.67     |         -          |       -       |
| ensemble_weighted_100         |     82.78     |         -          |       -       |
| t142_final_aca428_ffnn        |     67.43     |        100         |   4d 5h 17m   |
| t145_transformer_final_aca428 |     59.54     |        114         |     8h 8m     |
| t144_rf_final_aca428          |     59.15     |        2.5         |    3h 18m     |
| t142_final_aca428_nb          |     50.71     |        100         |   4d 5h 17m   |
| t143_lstm_final_aca428        |     63.41     |         89         |     9h 6m     |
| ensemble_mean428              |     70.79     |         -          |       -       |
| ensemble_weighted428          |     70.78     |         -          |       -       |

## Extended models trained for recognition of ACA and rotor ciphers

The models are trained on variable length ciphertexts in between 100 and 1000 characters. This was done to improve the recognition of the models towards rotor ciphers. For resonable recognitions of rotor ciphers longer ciphertexts are needed.

| Model Name                    | Accuracy in % | Iterations in Mio. |
| :---------------------------- | :-----------: | :----------------: |
| transformer_var_10000000      |     67.54     |         10         |
| rf_var_1000000                |     74.12     |          1         |
| lstm_var_10000000             |     66.40     |         10         |
| nb_var_10000000               |     53.50     |         10         |
| ffnn_var_10000000             |     72.98     |         10         |

These models are always part of an ensemble model with a SVM trained only on rotor ciphers. When the main models recognize rotor ciphers, the SVM is used to differentiate
between the rotor ciphers. This helps with the results since the original models can differentiate between ACA and rotor ciphers but are bad at differentiating rotor ciphers from each other.

| Model Name                    | Accuracy in % | Iterations in Mio. | Training Time |
| :---------------------------- | :-----------: | :----------------: | :-----------: |
| svm_rotor_only_1000_16000     |     61.50     |       0.016        |  0d 01h 01m   |


# Publications

[Masterarbeit: Systematische Evaluierung von Architekturen und Features für die Bestimmung von klassischen Chiffren-Typen mit neuronalen Netzen](https://www.cryptool.org/media/publications/theses/MA_Leierzopf.pdf)

[Histocrypt 2021: A Massive Machine-Learning Approach For Classical Cipher Type Detection Using Feature Engineering](https://doi.org/10.3384/ecp183)

AusDM 2021: Detection of Classical Cipher Types with Feature-Learning  Approaches

- [Proceedings](https://link.springer.com/book/10.1007/978-981-16-8531-6)
- [Pre-Print](https://www.cryptool.org/download/ncid/Detect-Classical-Cipher-Types-with-Feature-Learning_AusDM2021_PrePrint.pdf)
