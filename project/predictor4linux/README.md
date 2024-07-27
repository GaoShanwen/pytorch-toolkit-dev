## Command Guides

<details open>

<summary> Build the predictor library on Linux </summary>

```bash
<pytorch-toolkit-dev> # build the predictor with cmake;
<pytorch-toolkit-dev> cd project/predictor4linux;rm -r build; mkdir build && cd build; cmake ..; make; cd ../../..
<pytorch-toolkit-dev> # test the predictor.so
<pytorch-toolkit-dev> python project/predictor4linux/test.py
```

<details open>

<summary> Run the predictor on a new device </summary>

```bash
<pytorch-toolkit-dev> # get the lib files for predictor4linux:
<pytorch-toolkit-dev> ldd project/predictor4linux/build/libsmart_predictor.so
<pytorch-toolkit-dev> # copy the <lib files> and <test.py> to your directory

</the/directory/of/your/project> # run the predictor on your device:
</the/directory/of/your/project> export LD_LIBRARY_PATH=/the/path/to/your/lib/directory:$LD_LIBRARY_PATH
</the/directory/of/your/project> python test.py
```

TODO: Remove OpenCV library dependency.