#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])
#predict
#One Data [1~15]에서 train 1~10, test 11~15. train과 test 분리를 알아서 해야 함.


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.
model.fit(x_train, y_train, epochs=30, batch_size=5)


#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size=5) 
    #같은 값으로 예측하면 안됨. 훈련데이타, 평가 데이타는 달라야 함. 여기서 predict 값 생성.

print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import mean_squared_error
#함수정의 def ,함수이름은 사용자 정의
def RMSE(y_test, y_prdict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt 는 루트
print("RMSE : ", RMSE (y_test, y_predict))

