모두가 사용할 수 있는 파이프라인을 만들기 위해서는 통일된 이미지 사이즈에 맞춰서 deep learning에 넣을 필요가 있다.  
그래서 deep learning을 학습할 때 사용한 이미지의 사이즈에 맞춰서 nilearn resampling을 해주고, UNet에 이 이미지를 넣어서 mask를 얻은 다음에는 mask를 native space로 되돌려 보낼 필요가 있다. 
