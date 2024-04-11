# DistilBERT-FiNER

## Setup
* Python 3.12
* libraries in the `requirements.txt` file.
  * `pip install -r requirements.txt`

## How to train the model

Execute the whole jupyter notebook `FiNER_fine_tune.ipynb`. It will generate a ONNX model that later can be used for the demo application.

## How to test the model

An ONNX model can be downloaded from [Google Drive](https://drive.google.com/file/d/1pUG8v4JyDPQ7DAfay4CdpThOA-ewL6Wo/view?usp=sharing) if the full fine-tunning is not executed. The file has to be placed in the `onnx` directory.

To test the model execute the command line tool `cli_demo.py` as following:

`python cli_demo.py "<input text>"`

Example:

`python cli_demo.py "As of November 2015, $ 151.8 million of the originated loans were sold"`