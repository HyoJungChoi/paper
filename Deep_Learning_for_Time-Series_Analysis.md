[TOC]



### Deep Learning for Time-Series Analysis

##### John Gamboa https://arxiv.org/pdf/1701.01887.pdf

> ###### Abstract: In many real-world application, e.g., speech recognition or sleep stage classification, data are captured over the course of time, constituting a Time-Series. With the advent of Deep Learning new models of unsupervised learning of features for Time-series analysis and forecast have been developed.





#### 1. Introduction

- 깊은 구조가 얕은 구조보다 좋을거란 직관에도 불구하고 한,두 층을 가진 모델과 비교해봤을때 비슷하거나 더 나쁜 결과를 발견했다. 훈련이 어렵거나 비효율적이었기 때문이다. (너무 깊은 층은  되려 training error가 높아질 수 있다.)

- 기울기가 사라지는 것을 해결하기 위해 요즘은  각각 나누어 훈련하는 모듈을 사용하여 전층의 아웃풋을 다음층의 인풋이 되게 한다.

- 주식시장 가격부터 전염병 확산, 오디오 신호 녹음부터 수면 모니터링에 이르기까지  시간의 개념이 포함된 실제 세계 데이터를 흔히 볼 수 있다.
- 이 paper는 최근의 시계열 딥러닝 구조에 대해 리뷰한다. 

> Section 1.1 introduces basic types of Neural Network (NN) modules that are often used to build deep neural structures. Section 2 describes how the present paper relates to other works in the literature. Sections 3, 4 and 5 describe some approaches using Deep Learning to perform Modeling, Classification and Anomaly Detection in Time-Series data, respectively. Finally, Section 6 concludes the paper.



#### 1.1 Artificial Neural Network

- ANN은 기본적으로 지시된 연결에 의해 연결된 계산 단위의 네트워크다.

- 연결은 대부분 두 unit이 얼마나 강한 관련이 있는지에 대한 가중치를 갖는다. 

  

  <img src="C:\Users\chlgy\Documents\papers\image\function.png" alt="function" style="zoom: 67%;" />

- 일반적으로, unit 에 의해 수행되는 연산은 aggregation function과 activation function이라는 두 개의 그룹으로 구분된다.

- input layer과 output layer를 제외한 모든 층을 hidden layer라고 한다.

  <img src="C:\Users\chlgy\Documents\papers\image\ann.png" alt="ann" style="zoom:67%;" />

<img src="C:\Users\chlgy\Documents\papers\image\ann2.png" alt="ann2" style="zoom:67%;" />



학습 알고리즘의 초점은 주로 네트워크가 어떤 가중치로 인해 예상 값을 출력할 것인지를 결정하는 데 있다. 널리 사용되는 학습 알고리즘은 오류 함수의 기울기를 계산하고 오류를 최소화하기 위해 가중치를 반복적으로 설정하는 역전파 알고리즘이다.

#### CNN

- A network that is too big and with layers that are fully connected can become infeasible to train.
- 역전파 알고리즘으로 훈련된 CNN은 영상 처리 작업에 공통적으로 사용되며, 숨겨진 층에 있는 뉴런의 연결 수를 입력 뉴런의 일부(즉, 입력 이미지의 지역적 영역)로만 제한하여 학습할 파라미터의 수를 줄인다.
- 풀링으로 계층이 충분히 작아지면, 출력 계층 이전에 완전히 연결된 계층이 있는 것이 일반적이다.



#### RNN

- 네트워크가 루프를 가지고 있으면, 이를 RNN이라고 한다.
- 역전파 알고리즘을 적용하여 반복되는 네트워크를 훈련시킬 수 있다. 네트워크를 시간에 따라 "개폐"하고 연결부의 일부를 항상 동일한 가중치를 가지도록 제한한다.

##### LSTM(Long Short-Term Memory)

RNN의 전개에서 발생하는 한 가지 문제는 네트워크가 너무 많은 시간 단계에 걸쳐 펼쳐지면 일부 가중치의 기울기가 너무 작거나 너무 커지기 시작한다는 것이다. 이것을 *vanishing gradients* 문제라고 한다. 이 문제를 해결하는 네트워크 아키텍처의 한 종류는 LSTM이다. 일반적인 구현에서는 숨겨진 층을 블록의 오류를 가두어 놓는 게이트로 구성된 컴퓨팅 유닛의 복잡한 블록(그림 1c 참조)으로 대체하여 이른바 "error carrousel"을 형성한다.



### 2. Literature Review

딥러닝과는 별개로, 타임 시리즈 데이터의 분석은 경제, 엔지니어링 및 의학 같은 다른 분야에서 인기 있는 주제였다. ANN을 사용하여 Time-Series 데이터를 조작하는 대부분의 작업은 모델링 및 예측에 초점을 맞춘다. 최근의 접근법으로는 혼란스러운 TimeSeries를 예측하기 위한 Elman RNN의 사용, 인터넷 트래픽을 예측하기 위한 ANN의 합주 방법 사용, 북해의 쓰레기 양을 모델링하기 위한 단순한 다층 인식론 사용이 있다. 비교적 새로운 분야임에도 불구하고 딥러닝 분야는 지난 몇 년간 많은 관심을 끌었다.  우리는 시계열 데이터에 대한 딥러닝 적용 검토를 진행한다.

#### classification

모든 유형의 데이터를 분류하는 작업은 CNN의 출현으로 인해 이익을 얻었다. 과거의 분류 방법은 일반적으로 인간 전문가들에 의해 수작업으로 조작되는 도메인별 특징의 사용에 의존했다. 가장 좋은 특징을 찾는 것은 많은 연구의 대상이었고 분류자의 성과는 그 품질에 크게 의존하고 있었다. CNN의 장점은 스스로 그러한 특징들을 배울 수 있어 인간 전문가의 필요성을 줄일 수 있다는 것이다.

 

#### Forecasting

예측 과제를 수행하기 위한 문헌에서 몇 가지 다른 딥러닝 접근법을 찾을 수 있다. 예를 들어, Deep Belief Networks는 RBM과 함께 [33]의 작업에 사용된다.[58] 또한 Deep Belief Networks의 성능을 Stacked Denoising Autoencoders의 성능과 비교한다. 이 마지막 유형의 네트워크는 실내 환경의 온도를 예측하기 위해 [50]에도 사용된다. Time-Series 예측의 또 다른 애플리케이션은 [43]에서 찾을 수 있으며, 이 애플리케이션은 누적된 자동 코더를 사용하여 빅 데이터 데이터 세트의 트래픽 흐름을 예측한다.

Time-Series 예측 작업에 대한 일반적인 적용은 날씨 예측에 관한 것이다. [41]에서 홍콩 천문대가 제공하는 기상 데이터에 대한 일부 예비 예측은 스택형 오토엔코더의 사용을 통해 이루어진다. 후속 작업에서 저자들은 유사한 아이디어를 사용하여 빅 데이터에 대한 예측을 수행한다[40]. 오토엔코더 대신 [20]은 ANN이 날씨 예측 변수 사이의 공동 분포를 모델링하는 하이브리드 모델을 구축하기 위해 심층 신뢰 네트워크를 사용한다.



#### Anomaly Detection

Time-Series 데이터의 이상 징후 탐지에 딥러닝 기법을 적용하는 작업은 문헌에 그리 풍부하지 않다. 저준위 추적 알고리즘에서 얻은 궤적의 이상 징후 탐지를 수행하기 위해 적층 거부 자동 코더스를 사용하는 [16]과 같은 작품은 여전히 찾기 어렵다.



> 그러나 이상 징후 검출과 이전의 두 가지 과제에는 많은 유사성이 있다. 예를 들어, 이상 징후를 식별하는 것은 [35]에서와 같이 분류 작업으로 변환될 수 있다. 또는 예측값이 실제 값과 너무 다른 타임 시리즈에서 이상 징후를 감지하는 것은 영역을 찾는 것과 동일한 것으로 간주될 수 있다.

