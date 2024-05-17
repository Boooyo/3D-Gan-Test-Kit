## Self-Attention과 WGAN-GP를 적용한 3D GAN

이 프로젝트는 Self-Attention 메커니즘과 Gradient Penalty를 사용한 Wasserstein GAN (WGAN-GP)을 적용한 3D 생성적 적대 신경망 (GAN)을 구현합니다. 이 모델은 3D 볼륨 데이터를 생성하는 데 목적이 있습니다.

## 목차

- [소개](#소개)
- [요구 사항](#요구-사항)
- [모델 아키텍처](#모델-아키텍처)
- [사용법](#사용법)
- [훈련](#훈련)
- [결과](#결과)
- [참조](#참조)

## 소개

이 저장소는 Self-Attention 레이어와 WGAN-GP 방식을 사용하여 3D 볼륨 데이터를 생성하는 3D GAN 모델의 구현을 포함하고 있습니다. Self-Attention은 모델이 중요한 부분에 집중할 수 있게 하며, WGAN-GP는 안정적인 훈련을 제공합니다.

## 요구 사항

이 프로젝트를 실행하려면 다음 라이브러리가 필요합니다:
- TensorFlow 2.12.0
- NumPy 1.22.0
- Matplotlib 3.4.3

다음 명령어를 사용하여 필요한 라이브러리를 설치할 수 있습니다:


```
bash
pip install tensorflow==2.12.0 numpy==1.22.0 matplotlib==3.4.3
```

## 모델 아키텍처

Self-Attention 레이어
Self-Attention 레이어는 Generator와 Discriminator 모델 모두에 추가되어 3D 데이터의 중요한 특징에 집중할 수 있도록 도와줍니다.

## Generator
Generator 모델은 100차원의 노이즈 벡터를 입력으로 받아 16x16x16 3D 볼륨을 생성합니다. 이 모델은 Conv3DTranspose 레이어와 Self-Attention 레이어로 구성되어 있으며, 생성된 출력을 세밀하게 조정합니다.

## Discriminator
Discriminator 모델은 16x16x16 3D 볼륨을 입력으로 받아 실제 데이터와 생성된 데이터를 구분합니다. 이 모델 역시 Conv3D 레이어와 Self-Attention 레이어로 구성되어 있습니다.

## WGAN-GP
WGAN-GP는 그래디언트 페널티를 포함하여 안정적인 훈련을 제공합니다. 이를 통해 모델이 입력 노이즈에서 생성된 출력으로의 매핑을 부드럽게 학습할 수 있습니다.

## 사용법

1. 저장소 클론

```
git clone https://github.com/yourusername/3d-gan-wgan-gp.git
cd 3d-gan-wgan-gp
```
2. 훈련 스크립트 실행

```
python gan.py
```
## 훈련

### 데이터 준비
이 예제에서는 훈련 데이터로 임의의 3D 볼륨을 생성합니다. 실제 응용에서는 적절한 3D 데이터셋을 사용해야 합니다.

### 훈련 루프
훈련 루프는 다음과 같은 단계로 구성됩니다:

노이즈에서 가짜 3D 볼륨 생성
WGAN-GP 손실 함수를 사용하여 Generator와 Discriminator의 손실 계산
그래디언트 페널티를 적용하여 안정적인 훈련 유도
Adam 최적화를 사용하여 모델 가중치 업데이트

## 샘플 코드


```
batch_size = 32
epochs = 50
train_dataset = tf.data.Dataset.from_tensor_slices(generate_real_samples(1000)).shuffle(1000).batch(batch_size)

train(train_dataset, epochs)
```
## 결과
훈련 과정에서 각 에포크마다 3D 이미지를 생성하고 저장합니다. Matplotlib를 사용하여 이러한 이미지를 시각화하여 모델의 진행 상황을 확인할 수 있습니다.

## 예제 출력

```
def generate_and_save_images(model, epoch):
    noise = tf.random.normal([1, 100])
    predictions = model(noise, training=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(predictions[0, :, :, :, 0] > 0.5, edgecolor='k')
    plt.title(f'Epoch {epoch}')
    plt.show()

generate_and_save_images(generator, epochs)
```
## 참조

- Wasserstein GAN: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)   

- Self-Attention GAN: [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)   

