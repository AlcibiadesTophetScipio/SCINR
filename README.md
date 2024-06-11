# SCINR
A scene-conditional implicit neural regression model for mutli-scene visual relocalization.

## Setup

Install [TorchMeta](https://github.com/tristandeleu/pytorch-meta) and [DSACStar from ACE](https://github.com/nianticlabs/ace/tree/main/dsacstar).

Other required packages are list in 'requirements.txt'.

Download datasets:
- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Stanford 12-Scenes](https://graphics.stanford.edu/projects/reloc/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)

Make soft link to datasets.
```
mkdir datasets
# Put *.zip to ./datasets/Cambridge ./datasets/7scenes_source ./datasets/12scenes_source
ln -s ${dataset_path} ./datasets/${dataset_name}
```

Process datasets:
```
cd datasets
python ../dataset_process/setup_cambridge.py    # generate data in 'Cambridge' dir
python ../dataset_process/setup_7scenes.py      # generate data in '7scenes' dir
python ../dataset_process/setup_12scenes.py     # generate data in '12scenes' dir
# Note that you may need to link the 7scenes and 12scenes dirs in datasets manually
```

## ACE Experiments

ACE training and test
```
sh scripts/ace_train.sh 1 Cambridge together
sh scripts/ace_train.sh 0 7scenes together
sh scripts/ace_train.sh 0 12scenes together
```

ACE results output
```
sh scripts/ace_train.sh 1 Cambridge together print 2
sh scripts/ace_train.sh 0 7scenes together print 5
sh scripts/ace_train.sh 0 12scenes together print 5
```

## SCINR Experiments
SCINR training and test
```
sh scripts/hyper_train.sh 1 12scenes together RushH
sh scripts/hyper_train.sh 1 7scenes together RushH
sh scripts/hyper_train.sh 0 Cambridge together RushH
```

SCINR results output
```
sh scripts/hyper_train.sh 1 12scenes together RushH print 2
sh scripts/hyper_train.sh 1 7scenes together RushH print 2
sh scripts/hyper_train.sh 0 Cambridge together RushH print 2
```

## Ablation study

```
sh scripts/hyper_train.sh 1 12scenes together RHnoHyper print 2
sh scripts/hyper_train.sh 1 12scenes together RHnoM print 2
sh scripts/hyper_train.sh 0 12scenes together RHnoP print 2
sh scripts/hyper_train.sh 0 12scenes together RHnoC print 2
sh scripts/hyper_train.sh 0 12scenes together RHnoR print 2

sh scripts/hyper_train.sh 1 7scenes together RHnoM print 2
sh scripts/hyper_train.sh 1 7scenes together RHnoP print 2
sh scripts/hyper_train.sh 1 7scenes together RHnoC print 2
sh scripts/hyper_train.sh 1 7scenes together RHnoHyper print 2

sh scripts/hyper_train.sh 0 Cambridge together RHnoM print 2
sh scripts/hyper_train.sh 0 Cambridge together RHnoP print 2
sh scripts/hyper_train.sh 0 Cambridge together RHnoC print 2
sh scripts/hyper_train.sh 0 Cambridge together RHnoHyper print 2
```