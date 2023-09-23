import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Загрузите данные из JSON-файла
with open('annotations.json', 'r') as file:
    data = json.load(file)
    print(data)

# Извлеките входные данные (input) и выходные данные (output) из JSON-объектов

X = [['input'] for entry in data]
y = [entry['output'] for entry in data]

# Преобразуйте данные в массивы NumPy
X = np.array(X)
y = np.array(y)

# Создайте модель нейронной сети
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Скомпилируйте модель
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Обучите модель
model.fit(X, y, epochs=100, batch_size=16)

# Оцените модель
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Сохраните модель
model.save('модель.h5')
