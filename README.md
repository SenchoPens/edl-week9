Делал на VPS сервере (H100), не было времени настраивать jupyter notebook, поэтому запускал из терминала.


# Task 1
## 0
```
(edl) ubuntu@edl1:~/edl-week9$ python3 task0.py
loss=0.04599834233522415, accuracy=0.8768: : 10it [07:03, 38.32s/it]
Accuracy stabilized to 0.8768 at epoch 11
0.8809 10 0.04599834233522415
```

## 1
```
(edl) ubuntu@edl1-2:~/edl-week9$ python3 task1.py
loss=0.2514187693595886, accuracy=0.7511: : 16it [03:11, 11.20s/it]
Accuracy stabilized to 0.7511 at epoch 17
0.7585 16 0.2514187693595886
```


## 2
```
(edl) ubuntu@edl1-2:~/edl-week9$ python3 task2.py
loss=0.9624457359313965, accuracy=0.7909: : 15it [04:53, 18.28s/it]
Accuracy stabilized to 0.7909 at epoch 16
0.7827 15 0.9624457359313965
```


## 3
```
(edl) ubuntu@edl1-2:~/edl-week9$ python3 task3.py
loss=0.45746535062789917, accuracy=0.8041: : 10it [03:27, 18.78s/it]
Accuracy stabilized to 0.8041 at epoch 11
0.8035 10 0.45746535062789917
```

Итого:


# Task 2
```
Training loss: 0.216  Accuracy: 0.877: 100%|██████████████████████| 50/50 [14:03<00:00, 16.87s/it]
Test Accuracy: 0.6912
```

checkpoints/baseline_resnet.onnx, batch_size=1: results=BenchmarkResults({'items_per_second': 181.95070081723932, 'ms_per_batch': 5.495994219909335, 'batch_times_mean': 0.005495994219909334, 'batch_times_median': 0.004809830500107637, 'batch_times_std': 0.0010582844641023777})
checkpoints/baseline_resnet.onnx, batch_size=32: results=BenchmarkResults({'items_per_second': 318.6010798793157, 'ms_per_batch': 100.43908203990213, 'batch_times_mean': 0.10043908203990214, 'batch_times_median': 0.10019688399961524, 'batch_times_std': 0.0007332040748538435})
checkpoints/pruned_quantized_resnet.onnx, batch_size=1: results=BenchmarkResults({'items_per_second': 504.0603929468853, 'ms_per_batch': 1.9838892600819236, 'batch_times_mean': 0.0019838892600819234, 'batch_times_median': 0.00198296099961226, 'batch_times_std': 2.5423880172748976e-05})
checkpoints/pruned_quantized_resnet.onnx, batch_size=32: results=BenchmarkResults({'items_per_second': 741.3899072354204, 'ms_per_batch': 43.162173759992584, 'batch_times_mean': 0.04316217375999258, 'batch_times_median': 0.043145712000296044, 'batch_times_std': 0.0004697399549358728})
```
