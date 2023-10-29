<div align="center">
</br>
<img src="resources/LOGO.png" width="200" />

</div>

<h1 align="center">libcom: everything about image composition</h1>

</br>

[![PyPI](https://img.shields.io/pypi/v/libcom)](https://pypi.org/project/libcom)
[![GitHub](https://img.shields.io/github/license/bcmi/libcom)](https://github.com/bcmi/libcom/blob/main/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/bcmi/libcom)
![GitHub forks](https://img.shields.io/github/forks/bcmi/libcom)
[![Static Badge](https://img.shields.io/badge/Image%20Composition%20Demo-Green)](https://bcmi.sjtu.edu.cn/home/niuli/demo_image_composition/)
[![Static Badge](https://img.shields.io/badge/Survey-Arxiv%3A2106.14490-red)](https://arxiv.org/pdf/2106.14490.pdf)


## Introduction
**_libcom_** is an image composition toolbox covering various related tasks, including naive image composition, image blending, color transfer, image harmonization, painterly harmonization, object placement, *etc*. Detailed technical details and api descriptions can be found in the official [[documents]](xxxx). 

## Usage

### Create runtime environment

```shell
git clone https://github.com/bcmi/libcom.git
cd libcom/requirements
conda env create -f libcom.yaml
conda activate Libcom
pip install -r runtime.txt # -i https://pypi.tuna.tsinghua.edu.cn/simple
# install a specific version of taming-transformers from source code
cd ../libcom/controllable_composition/source/ControlCom/src/taming-transformers
python setup.py install
```

### Installation
```shell
pip install libcom
```
or
```shell
python setup.py install
```
After that, you can verify the installation by running:
```shell
cd tests
sh run_all_tests.sh
```
The visualization results can be found in `results` folder.

### Download pretrained models
During using the toolbox, the pretrained models and related files will be automatically downloaded to the installation directory. Note downloading the pretrained models may take some time when you first call some models, especially `ControlComModel` and `PainterlyHarmonizationModel`.

Alternatively, you can download these files from [[Modelscope]](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/files) or [[Huggingface]](https://huggingface.co/BCMIZB/Libcom_pretrained_models/tree/main) in advance, and move them to the installation directory.
