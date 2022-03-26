# image_deep_learning


data set 

훈련 및 검증 데이터 셋 : https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

테스트 데이터 셋 : https://aimi.stanford.edu/chexphoto-chest-x-rays

x-ray 이미지와 폐의 mask 이미지 대조하여 Unet을 통해 
폐를 segmentation하는 학습을 진행

![Unet](https://user-images.githubusercontent.com/80818827/160245520-63221a4e-4461-4acd-aa89-84a3eeae055d.png)


mask가 없는 새로운 x-ray 사진을 통해 검증
