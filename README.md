# THU-ACP DeepLearning 2021 Project

# Kaggle Competition 
## Shopee - Price Match Guarantee

Determine if two products are the same by their images

[Kaggle Link](https://www.kaggle.com/c/shopee-product-matching)
 
# Module
- Training
  - [x] CurricularFace Training
- Inference
  - [x] tdidf 
  - [ ] image feature

# Result

| Model              | KNN Threshold |Private |Public  |CV       |
| ------------------ |:-------------:| ------:|-------:|:-------:|
| TFIDF              | 0.6           | 0.562  | 0.574  | 0.60192 |
| TFIDF              | 0.8           | 0.523  | 0.531  | 0.63497 |
| NFnet+CuFace - E0  | 3.0           | 0.627  | 0.638  | 0.64980 |
# References

- code 
  - [ ] [Kaggle 37th](https://www.kaggle.com/takusid/37th-place-solution-version-76)
  - [ ] [Kaggle 2nd](https://www.kaggle.com/lyakaap/2nd-place-solution)
  - [ ] [Kaggle 5th](https://www.kaggle.com/aerdem4/shopee-v03)
  - [X] [Kaggle CurricularFace](https://www.kaggle.com/parthdhameliya77/curricularface-eca-nfnet-l0-pytorch-training)

- lib
  - [timm - vision model](https://github.com/rwightman/pytorch-image-models)
  - [albumentations - image augmentation](https://github.com/albumentations-team/albumentations)

