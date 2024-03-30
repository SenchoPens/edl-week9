Делал на VPS сервере (H100), не было времени настраивать jupyter notebook, поэтому запускал из терминала.


# Task 0
```
(edl) ubuntu@edl1:~/edl-week9$ python3 task0.py
Files already downloaded and verified
Files already downloaded and verified
/home/ubuntu/miniforge3/envs/edl/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
  warnings.warn(
/home/ubuntu/miniforge3/envs/edl/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet101_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet101_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
loss=0.04599834233522415, accuracy=0.8768: : 10it [07:03, 38.32s/it]Accuracy stabilized to 0.8768 at epoch 11
loss=0.04599834233522415, accuracy=0.8768: : 10it [07:03, 42.39s/it]
```


# Task 1
(edl) ubuntu@edl1-2:~/edl-week9$ python3 task1.py
Files already downloaded and verified
Files already downloaded and verified
loss=0.2514187693595886, accuracy=0.7511: : 16it [03:11, 11.20s/it]Accuracy stabilized to 0.7511 at epoch 17
loss=0.2514187693595886, accuracy=0.7511: : 16it [03:11, 11.94s/it]
0.7585 16 0.2514187693595886
