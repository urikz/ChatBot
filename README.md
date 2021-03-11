# ChatBots

# Best parameters for the models

### Seq2Seq Model on OpenSubtitles2018
```
$ python ShaLab/engine.py -d ~/data/OpenSubtitles2018/preprocessed.personachat/ --sort-batches -lr 0.5 -m 0.1 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 10 -dp 0.35 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 10000 -o ~/models/OpenSubtitles2018/preprocessed.personachat/Seq2Seq/lr-0.5 --gpu 0
2018-05-12 18:49:08 INFO The best checkpoint /home/urikz/models/OpenSubtitles2018/preprocessed.personachat/Seq2Seq/lr-0.5/model.checkpoint.epoch-10.pth.tar. Picking up the model from there
2018-05-12 18:49:08 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.personachat.original] after epoch 10 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 65.396
----- Validation [valid.personachat.train.original] after epoch 10 (122496 samples, 11.73 avg source length, 12.89 avg target length) perplexity 77.164
----- Validation [valid.OpenSubtitles2009] after epoch 10 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 18.490
----- Validation [valid] after epoch 10 (13579648 samples, 7.17 avg source length, 8.13 avg target length) perplexity 19.830
----- Validation [valid.personachat.revised] after epoch 10 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 65.395
----- Validation [test.OpenSubtitles2009] after epoch 10 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 18.301
----- Validation [test.OpenSubtitles2018] after epoch 10 (13537536 samples, 7.18 avg source length, 8.14 avg target length) perplexity 19.867
----- Validation [test.personachat.revised] after epoch 10 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 69.681
----- Validation [test.personachat.original] after epoch 10 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 69.671
```

### Seq2Seq Model on PersonaChat
```
$ python ShaLab/engine.py -d ~/data/personachat/preprocessed/vocab-full.train-original/ --sort-batches -lr 1.0 -m 0.1 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 10 -dp 0.35 --glove ~/word_vectors/glove.6B.300d.txt -o /tmp/model-0 --gpu 0

2018-04-19 12:17:06 INFO The best checkpoint /tmp/model-0/model.checkpoint.epoch-9.pth.tar. Picking up the model from there
2018-04-19 12:17:06 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.OpenSubtitles2009] after epoch 10 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 226.403
----- Validation [valid.revised] after epoch 10 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 33.646
----- Validation [valid] after epoch 10 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 33.646
----- Validation [test.OpenSubtitles2009] after epoch 10 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 227.319
----- Validation [test.original] after epoch 10 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 31.538
----- Validation [test.revised] after epoch 10 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 31.538
```

### Seq2Seq Model on PersonaChat and OpenSubtitles2009
```
$ python ShaLab/engine.py -d ~/data/OpenSubtitles/preprocessed.personachat/ --sort-batches -lr 1.0 -m 0.1 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 10 -dp 0.35 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o ~/models/OpenSubtitles/preprocessed.personachat/Seq2Seq/baseline --gpu 0
2018-05-09 16:06:22 INFO The best checkpoint /home/urikz/models/OpenSubtitles/preprocessed.personachat/Seq2Seq/baseline/model.checkpoint.epoch-10.pth.tar. Picking up the model from there
2018-05-09 16:06:22 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.personachat.original] after epoch 10 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 106.489
----- Validation [valid.personachat.train.original] after epoch 10 (122496 samples, 11.73 avg source length, 12.89 avg target length) perplexity 126.017
----- Validation [valid] after epoch 10 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 23.232
----- Validation [valid.personachat.revised] after epoch 10 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 106.494
----- Validation [test.OpenSubtitles2009] after epoch 10 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 23.606
----- Validation [test.personachat.revised] after epoch 10 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 114.291
----- Validation [test.personachat.original] after epoch 10 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 114.270
```

### Seq2Seq Model on PersonaChat and OpenSubtitles2009
```
$ python ShaLab/engine.py -d ~/data/personachat_with_OS2009 --sort-batches -lr 1.0 -m 0.1 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 10 -dp 0.35 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o /tmp/model-5.seq2seq --gpu 0

2018-04-29 18:42:49 INFO The best checkpoint /tmp/model-5.seq2seq/model.checkpoint.epoch-10.pth.tar. Picking up the model from there
2018-04-29 18:42:50 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.OpenSubtitles2009] after epoch 20 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 23.105
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 31.039
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 31.041
----- Validation [test.OpenSubtitles2009] after epoch 20 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 23.376
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.022
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.023
```

### ProfileMemoryModel (without default memory) on PersonaChat
```
$ python ShaLab/engine.py -d ~/data/personachat/preprocessed/vocab-full.train-original/ --sort-batches -lr 1.0 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.5 --glove ~/word_vectors/glove.6B.300d.txt -o ~/models/personachat/ProfileMemoryModel/no_default_memory --gpu 3 --profile-memory-attention general

2018-05-01 21:41:44 INFO The best checkpoint /home/urikz/models/personachat/ProfileMemoryModel/no_default_memory/model.checkpoint.epoch-20.pth.tar. Picking up the model from there
2018-05-01 21:41:44 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 32.711
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.686
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.406
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.911
```

### ProfileMemoryModel (without default memory) on PersonaChat and OpenSubtitles2009
```
$ python ShaLab/engine.py -d ~/data/personachat_with_OS2009 --sort-batches -lr 0.5 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.4 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o ~/models/personachat+OS2009/ProfileMemoryModel/no_default_memory --gpu 2 --profile-memory-attention general

2018-05-01 23:35:37 INFO The best checkpoint /home/urikz/models/personachat+OS2009/ProfileMemoryModel/no_default_memory/model.checkpoint.epoch-14.pth.tar. Picking up the model from there
2018-05-01 23:35:37 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.443
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 28.252
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 27.373
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.450
```

### ProfileMemoryModel (without default memory) on PersonaChat and DailyDialog
```
$ python ShaLab/engine.py -d ~/data/PersonaChat/withDailyDialog --sort-batches -lr 0.5 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.4 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o ~/models/personachat+DailyDialog/ProfileMemoryModel/no_default_memory --gpu 0 --profile-memory-attention general --model-type profile-memory

2018-07-01 17:03:20 INFO The best checkpoint /home/urikz/models/personachat+DailyDialog/ProfileMemoryModel/no_default_memory/model.checkpoint.epoch-12.pth.tar. Picking up the model from there
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 32.008
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.463
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.270
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.394
```

### ProfileMemoryModel (without default memory) on PersonaChat and CornellMovieDialogCorpus
```
$ python ShaLab/engine.py -d ~/data/PersonaChat/withCornellMovieDialogCorpus/ --sort-batches -lr 0.5 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.4 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o ~/models/personachat+CornellMovieDialogCorpus/ProfileMemoryModel/no_default_memory --gpu 2 --profile-memory-attention general --model-type profile-memory

2018-07-01 17:10:40 INFO The best checkpoint /home/urikz/models/personachat+CornellMovieDialogCorpus/ProfileMemoryModel/no_default_memory/model.checkpoint.epoch-14.pth.tar. Picking up the model from there
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 31.682
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.436
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.423
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.330
```

### ProfileMemoryModel on PersonaChat
```
$ python ShaLab/engine.py -d ~/data/personachat/preprocessed/vocab-full.train-original/ --sort-batches -lr 1.0 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.5 --glove ~/word_vectors/glove.6B.300d.txt -o /tmp/model-3 --gpu 3 --profile-memory-attention general --use-default-memory

2018-04-19 12:27:55 INFO The best checkpoint /tmp/model-3/model.checkpoint.epoch-19.pth.tar. Picking up the model from there
2018-04-19 12:27:55 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.OpenSubtitles2009] after epoch 20 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 236.495
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 32.722
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.483
----- Validation [test.OpenSubtitles2009] after epoch 20 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 236.940
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.098
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.779
```

### ProfileMemoryModel on PersonaChat and OpenSubtitles2009
```
$ python ShaLab/engine.py -d ~/data/personachat_with_OS2009 --sort-batches -lr 0.5 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.4 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o /tmp/model-2 --gpu 2 --profile-memory-attention general --use-default-memory

2018-05-13 00:58:57 INFO The best checkpoint /tmp/model-0/model.checkpoint.epoch-15.pth.tar. Picking up the model from there
2018-05-13 00:58:57 INFO Loaded dialog model from checkpoint in 0 seconds
----- Validation [valid.OpenSubtitles2009] after epoch 20 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 23.053
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.667
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 27.505
----- Validation [test.OpenSubtitles2009] after epoch 20 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 23.224
----- Validation [test.CornellMovieDialogCorpus] after epoch 20 (85760 samples, 8.29 avg source length, 9.21 avg target length) perplexity 29.818
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 26.764
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.622
```

### ProfileMemoryModel on PersonaChat and DailyDialog
```
$ python ShaLab/engine.py -d ~/data/PersonaChat/withDailyDialog --sort-batches -lr 0.5 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.4 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o ~/models/personachat+DailyDialog/ProfileMemoryModel/baseline --gpu 0 --profile-memory-attention general --use-default-memory --model-type profile-memory

2018-07-01 16:21:36 INFO The best checkpoint /home/urikz/models/personachat+DailyDialog/ProfileMemoryModel/baseline/model.checkpoint.epoch-11.pth.tar. Picking up the model from there
----- Validation [valid.OpenSubtitles2009] after epoch 20 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 62.022
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 31.694
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 29.119
----- Validation [test.OpenSubtitles2009] after epoch 20 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 60.853
----- Validation [test.CornellMovieDialogCorpus] after epoch 20 (85760 samples, 8.29 avg source length, 9.21 avg target length) perplexity 57.354
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 28.033
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.220
```

### ProfileMemoryModel on PersonaChat and CornellMovieDialogCorpus
```
$ python ShaLab/engine.py -d ~/data/PersonaChat/withCornellMovieDialogCorpus/ --sort-batches -lr 0.5 -m 0.2 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 20 -dp 0.4 --glove ~/word_vectors/glove.6B.300d.txt --log-interval 1000 -o ~/models/personachat+CornellMovieDialogCorpus/ProfileMemoryModel/baseline --gpu 2 --profile-memory-attention general --use-default-memory --model-type profile-memory

2018-07-01 16:32:57 INFO The best checkpoint /home/urikz/models/personachat+CornellMovieDialogCorpus/ProfileMemoryModel/baseline/model.checkpoint.epoch-14.pth.tar. Picking up the model from there
----- Validation [valid.OpenSubtitles2009] after epoch 20 (167296 samples, 7.10 avg source length, 8.05 avg target length) perplexity 38.776
----- Validation [valid.revised] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 31.611
----- Validation [valid] after epoch 20 (14592 samples, 11.99 avg source length, 13.11 avg target length) perplexity 28.971
----- Validation [test.OpenSubtitles2009] after epoch 20 (152704 samples, 7.04 avg source length, 8.00 avg target length) perplexity 38.226
----- Validation [test.original] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 27.980
----- Validation [test.revised] after epoch 20 (13952 samples, 11.78 avg source length, 12.94 avg target length) perplexity 30.235
----- Validation [test.DailyDialog] after epoch 20 (47744 samples, 10.29 avg source length, 11.19 avg target length) perplexity 42.194
```
