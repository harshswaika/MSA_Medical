# MSA_Medical

## Victim models:
1. Radformer (data_msa_medical/victim_models/radformer/radformer.pkl)
2. POCUS-ResNet18 (data_msa_medical/victim_models/pocus_resnet18.pth.tar)

## Datasets:
1. GBC victim dataset - GBCU (data_msa_medical/GBCU-Shared) 
2. GBCthief dataset - GBUSV (data_msa_medical/GBUSV-Shared)
3. COVID victim dataset - POCUS (data_msa_medical/covid_5_fold)
4. COVID thief dataset - COVIDx-US (data_msa_medical/covidx_us)

## ActiveThief: 
Use the script `activethief_gbc.py` to train a thief model. Config files are named as `<victim_arch>_<thief_arch>_<thief_dataset>.yaml`
```
cd activethief
python activethief_gbc.py --c configs/gbc/radformer_resnet50_gbusv.yaml
```

For the covid dataset, use `activethief_pocus.py`

## SSL:
Use the script `train_proposed.py' to train a thief model using SSL. 
```
cd ssl
python train_proposed.py --c configs/gbc/selfkd.yaml
```

For the covid dataset, use script `train_proposed_pocus.py'. 
```
cd ssl
python train_proposed_pocus.py --c configs/covid/selfkd_resnet50.yaml
```
