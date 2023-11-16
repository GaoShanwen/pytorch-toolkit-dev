## structure of the project

```bash
<pytorch-toolkit-dev> <timm-dev>
    ├── dataset
    │   ├── 1.jpg
    │   ├── exp-data # train/validate data
    │   └── README.md
    ├── docs
    │   ├── environment.md
    │   └── structure.md
    ├── local_lib
    │   ├── data
    │   ├── models
    │   └── utils
    ├── output 
    │   └── train # .pth model path
    ├── README.md
    ├── requirements.txt
    └── tools
        ├── before
        │   ├── check_data.py
        │   ├── mongo_client.py
        │   └── ready_det_data.py
        ├── convert
        │   ├── onnx2rknn.py
        │   └── onnx_export.py
        ├── eval_feats.py
        ├── post
        │   ├── feat_extract.py
        │   ├── feat_tools.py
        │   └── write_mysql.py
        ├── train.py
        ├── validate.py
        └── visualize
```
