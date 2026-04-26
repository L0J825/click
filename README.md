# CLICK: Efficient Medical Image Prediction for Cold Data Compression

The official PyTorch implementation of CLICK (Context-aware Lossless Image Compression Kit).

## Dependency

- See [requirements.txt](requirements.txt) for Python dependencies.
- Additional CUDA extensions may be required for deformable convolution operations.
- The lossy codec is based on [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-DC). Please refer to the official repository for environment setup and pretrained weights.

## Training

### 1. Enhancement Model (RFDMNet)

Train the recurrent feature deblur module to enhance lossy reconstructions:

```bash
python src/main_en.py --dataset <dataset_name>
```

Supported datasets: `axial`, `coronal`, `sagittal` (8-bit), `mosmed`, `chaosct` (16-bit).

### 2. Lossless Model (LCEN)

Train the lossless entropy coding network:

```bash
python src/main.py --dataset <dataset_name>
```

## Pretrained Models

Download the pretrained model weights from [Baidu Netdisk](https://pan.baidu.com/s/1356WduhocaGjAoJJdSzjYA?pwd=y6vv) (password: `y6vv`).

## Testing

Run compression evaluation on test datasets:

```bash
python src/test.py --dataset <dataset_name> --decode
```

The `--decode` flag enables full encode-decode verification.

## Datasets

We conduct experiments on multiple public volumetric datasets:

| Name       | Position | Type | Bit Depth | Resolution          |
| ---------- | -------- | ---- | --------- | ------------------- |
| MRNet      | Knee     | MRI  | 8         | 17\~61 × 256 × 256  |
| MosMedData | Lung     | CT   | 16        | 33\~72 × 512 × 512  |
| CHAOS-CT   | Abdomen  | CT   | 16        | 77\~105 × 512 × 512 |

