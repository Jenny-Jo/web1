import numpy as np
#데이터 생성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from keras.models import Sequential
from keras.layers import Dense

#케라스 회귀모델, 순차적 모델/ Dense layer 추가
model = Sequential()
model.add(Dense(1,input_dim=1,activation='relu'))

#머신이 어떻게 모델을 돌릴 지 지정=컴파일
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

#모델 실행 fit / 최종 결과에 대한 평가
model.fit(x, y, epochs=500, batch_size=1)
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("acc : ",acc)
