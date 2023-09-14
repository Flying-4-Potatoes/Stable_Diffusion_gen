# Stable_Diffusion_gen


## experiments

이미지를 생성 할 때, text prompt는 결과물에 많은 영향을 미칩니다.
저희가 직접 크롤링하고 전처리한 tsv 형식의 데이터셋을 불러와 동화 일러스트를 생성합니다.

순서는 다음과 같습니다.

1. 초기 이미지를 생성하고 저장
2. 프롬프트를 생성하고 이미지를 생성
3. 이미지를 표시하고 요약모델을 이용해 프롬프트를 요약한 후 해당 이미지를 생성
4. 반복

## 예시 이미지

![image](https://github.com/Flying-4-Potatoes/Stable_Diffusion_gen/assets/79971467/56a9c4c5-a1af-4bf7-a2c4-f984da63f673)


![image](https://github.com/Flying-4-Potatoes/Stable_Diffusion_gen/assets/79971467/a079131e-bdb6-4e3c-9a1c-c12852427248)
