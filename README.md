# ChatBot Projects

The project is focusing on developing data-driven dialog models.

##### Personalizing Dialogue Agents: I have a dog, do you have pets too?
Steps to reproduce results from the paper https://arxiv.org/abs/1801.07243
1. Download and extract data
2. Prepare data with the script
```
$ python scripts/prepare_personachat.py --data ~/data/personachat/ --out <DATA_DIR>
Building vocabulary from /home/urikz/data/personachat/train_both_original.txt
... extracting words from file /home/urikz/data/personachat/train_both_original.txt
..... finished in 94 seconds
Total number of sentences: 1460464
2018-03-04 17:51:49 INFO Built dictionary (18828 words) from corpus in 4 seconds
Preparing file /home/urikz/data/personachat/valid_none_original.txt
Parsed file /home/urikz/data/personachat/valid_none_original.txt in 1 seconds
Found 841 unique unknown words (0.78% = 1362/175188):
2018-03-04 17:51:51 INFO Built a PackedIndexedCorpus from 14602 sentences in 0 seconds
2018-03-04 17:51:51 INFO Serialized PackedIndexedCorpus to /tmp/out-data/tune.source.npz in 0 seconds
Found 891 unique unknown words (0.80% = 1422/176828):
2018-03-04 17:51:51 INFO Built a PackedIndexedCorpus from 14602 sentences in 0 seconds
2018-03-04 17:51:51 INFO Serialized PackedIndexedCorpus to /tmp/out-data/tune.target.npz in 0 seconds
Preparing file /home/urikz/data/personachat/test_none_original.txt
Parsed file /home/urikz/data/personachat/test_none_original.txt in 1 seconds
Found 687 unique unknown words (0.67% = 1116/166561):
2018-03-04 17:51:53 INFO Built a PackedIndexedCorpus from 14056 sentences in 0 seconds
2018-03-04 17:51:53 INFO Serialized PackedIndexedCorpus to /tmp/out-data/tune-test.source.npz in 0 seconds
Found 726 unique unknown words (0.68% = 1148/168022):
2018-03-04 17:51:53 INFO Built a PackedIndexedCorpus from 14056 sentences in 0 seconds
2018-03-04 17:51:53 INFO Serialized PackedIndexedCorpus to /tmp/out-data/tune-test.target.npz in 0 seconds
Preparing file /home/urikz/data/personachat/train_none_original.txt
Parsed file /home/urikz/data/personachat/train_none_original.txt in 10 seconds
2018-03-04 17:52:05 INFO Built a PackedIndexedCorpus from 122499 sentences in 0 seconds
2018-03-04 17:52:05 INFO Serialized PackedIndexedCorpus to /tmp/out-data/train.source.npz in 0 seconds
2018-03-04 17:52:06 INFO Built a PackedIndexedCorpus from 122499 sentences in 0 seconds
2018-03-04 17:52:06 INFO Serialized PackedIndexedCorpus to /tmp/out-data/train.target.npz in 0 seconds
```
3. Train model
```
$ python ShaLab/engine.py -d <DATA_DIR> -o <MODEL_DIR> --gpu 0 --sort-batches --glove ~/word_vectors/glove.6B.300d.txt -lr 1.0 -emsz 300 -hsz 1024 -nlayers 1 -bs 128 --num-epochs 10 -dp 0.2
```
4. Run dialog model
```
python ShaLab/dialog.py -d <DATA_DIR> --model <MODEL_DIR>/model.checkpoint.epoch-6.pth.tar --gpu 0 --beam-size 10
```
