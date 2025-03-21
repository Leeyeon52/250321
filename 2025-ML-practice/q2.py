import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 
"""
1. data 폴더 내에 있는 dataset.csv파일을 불러오고, 
   학습용 데이터와 테스트용 데이터를 분리하여 
   반환하는 함수를 구현합니다.
"""
def load_data():
    
    data = pd.read_csv('data/dataset.csv')
    
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print(X, y)
    return train_X, test_X, train_y, test_y
    
"""
2. SVM 모델을 불러오고,
   학습용 데이터에 맞추어 학습시킨 후, 
   테스트 데이터에 대한 예측 결과를 반환하는 함수를
   구현합니다.
"""
def SVM(train_X, test_X, train_y, test_y):
    
    svm = SVC(kernel='linear', random_state=0)
    
    svm.fit(train_X, train_y)
    
    pred_y = svm.predict(test_X)
    
    return pred_y
    
# 데이터를 불러오고, 모델 예측 결과를 확인하는 main 함수입니다.
def main():
    
    train_X, test_X, train_y, test_y = load_data()
    
    pred_y = SVM(train_X, test_X, train_y, test_y)
    
    # SVM 분류 결과값을 출력합니다.
    print("\nConfusion matrix : \n",confusion_matrix(test_y,pred_y))  
    print("\nReport : \n",classification_report(test_y,pred_y)) 

    cm = confusion_matrix(test_y, pred_y)
    cm_df = pd.DataFrame(cm, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    
    # Classification report를 pandas DataFrame으로 변환
    report = classification_report(test_y, pred_y, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # 결과를 엑셀로 저장
    with pd.ExcelWriter('svm_results.xlsx') as writer:
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')  # Confusion matrix 저장
        report_df.to_excel(writer, sheet_name='Classification Report')  # Classification report 저장


if __name__ == "__main__":
    main()
