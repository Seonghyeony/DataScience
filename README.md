# DataScience
학교 데이터사이언스 강의

* Final Project

Image Segmentation에서 다루었던 Oxford-IIIT_Pet_Dataset을 이용하여 이미지를 고양이와 개로 분류하는 신경망을 구성하라.
1. CNN을 사용하라.
2. 데이터셋은 Python 소스코드가 위치한 디렉토리에 Oxford-IIIT_Pet_Dataset이라는 이름의 디
렉토리를 만들고 그 안에 images와 annotations 디렉토리를 위치하라. 각 디렉토리의 파일들은
다운로드 받아 압축을 푼 상태 그대로 유지하라. (단, 손상된 jpeg 파일이 있다면 제거하라.)
3. Dataset에 포함된 ReadMe 파일을 읽어보면 이미지들이 어떻게 고양이와 개로 분류되어 있는지 알
수 있다.
4. trainval.txt에 수록된 파일들을 train 데이터로, test.txt 파일에 수록된 파일들을 test 데이터
로 사용하라.
5. 신경망의 train이 완료된 후 모든 테스트 데이터에 대해서 평균 정확도(accuracy)를 계산하여 출력
하도록 코딩 한다.
