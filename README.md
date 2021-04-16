# Caffe Face Light Field Project

## **Caffe**
- Berkeley vision의 순정 caffe (CUDA GPU버전) + python 3.5 사용
- 모델에 필요한 CUDA custom layer, Python layer를 Caffe 코어에 병합
- 위 환경을 Docker image를 통해 사용 가능[https://drive.google.com/file/d/1tCNuNCYubNiOdmBELJ0VYFFv09wEX1d5/view?usp=sharing](https://drive.google.com/file/d/1tCNuNCYubNiOdmBELJ0VYFFv09wEX1d5/view?usp=sharing)
## **Dataset**
- Chicago face 데이터셋으로 생성한 training set과 Inha VCL face dataset으로 생성한 test set으로 구성
- Spatial resolution: 256 x 256 / Angular resolution: 5 x 5
- Baseline: 2.5mm / 1.25mm
- [https://drive.google.com/drive/folders/1IN2ALGk3GbHz-vOxWNQHgU3t-5_A6QQa?usp=sharing](https://drive.google.com/drive/folders/1IN2ALGk3GbHz-vOxWNQHgU3t-5_A6QQa?usp=sharing)
## **Source code**
- datas 디렉토리에 다운로드한 데이터셋 추가
- 쉘 스크립트 제공 예정
- https://github.com/JunhyeongBak/Caffe_Light_Field
## **Pretrained model**
- Caffe prototxt
- Caffe model (baseline 2.5mm 데이터셋으로 batch size 4에 130000 iteration학습)
- 쉘 스크립트 제공 예정
- https://drive.google.com/file/d/1xtT_ycUhbTD847kkBndd9ar_LVRc6Nj3/view?usp=sharing
