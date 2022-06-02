# 파이프라인 구성시 주의 사항 
모두가 사용할 수 있는 파이프라인을 만들기 위해서는 통일된 이미지 사이즈에 맞춰서 deep learning에 넣을 필요가 있다.  
그래서 deep learning을 학습할 때 사용한 이미지의 사이즈에 맞춰서 nilearn resampling을 해주고, UNet에 이 이미지를 넣어서 mask를 얻은 다음에는 mask를 native space로 되돌려 보낼 필요가 있다.  
이를 위해서 학습시와 파이프라인 사용시 들어가는 input 이미지를 MNI 사이즈와 좌표계에 맞춰서 resampling 해주고, mask를 얻어서 이를 다시 원래의 좌표계로 resampling 해주는 과정을 해준다.  
