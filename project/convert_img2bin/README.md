## Command Guides

<details open>

<summary>Prepare so file</summary>

```bash
<pytorch-toolkit-dev> ~$ # cpp -> so
        g++ ./project/convert_img2bin/write_nx.cpp -o ./project/convert_img2bin/write_nx.so -lssl -lcrypto
        g++ ./project/convert_img2bin/read_nx.cpp -o ./project/convert_img2bin/read_nx.so -lssl -lcrypto
```

</details>

<details open>

<summary>Start convert tasks</summary>

```bash
<pytorch-toolkit-dev> ~$ sh project/convert_imgs2bin/run_server.sh
```

the output will show the convert log, details of each brand will be shown in 'dataset/feature_pack/tmp/log'.

```bash
Starting convert imgs to bin for 107 tasks...
[   1 / 107] Task dataset info | brand_id:   71, images: 13214, Running task on GPU 0 |
[   2 / 107] Task dataset info | brand_id:   75, images:  8598, Running task on GPU 1 |
[   3 / 107] Task dataset info | brand_id:   76, images: 27852, Running task on GPU 2 |
...
[ 105 / 107] Task dataset info | brand_id:   97, images: 12304, Running task on GPU 7 |
[ 106 / 107] Task dataset info | brand_id:   98, images: 16698, Running task on GPU 5 |
start-time: 2024-07-07 22:42:00, end-time: 2024-07-08 01:32:19, total img count: 2635447
```

</details>
