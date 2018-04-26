# Jpeg Compresson

Compress JPEG (or whatever) images.

## How

1.  Discrete cossine transform
2.  Clip values exceeding threshold
3.  Inverse DCT

## Requirement

* macOS / Ubuntu (prefered)
* Python 3.6 or newer

## Install

```
pip3 install -r requirements.txt
```

## Usage

change the threshold value to adjust the compression amount of an image.

```
python3 dct.py --input ./source.jpg --threshold 0.02
python3 dct.py --input ./source.jpg --threshold 0.05
python3 dct.py --input ./source.jpg --threshold 0.3
```

If you want to change filter size, see the `filter_shape` tuple.

## Ask

```
Yasuaki Uechi
uechi@sfc.keio.ac.jp
```
