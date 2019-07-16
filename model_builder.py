from utils import load_data
from my_CNN_model import *
import cv2


X_train, y_train = load_data()

my_model = get_my_CNN_model_architecture()

compile_my_CNN_model(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

hist = train_my_CNN_model(my_model, X_train, y_train)

save_my_CNN_model(my_model, 'my_model')

