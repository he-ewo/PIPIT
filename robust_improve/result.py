from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_membership_inference(y_true, membership_predictions):
    # 예측된 멤버십 레이블
    y_pred = (membership_predictions > 0.5).astype(int)
    
    # print(len(y_pred), y_pred)
    # print(len(y_true), y_true)

    # 성능 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 결과를 인덱스 순서대로 출력
    # print("Membership Inference Results:")
    # for i in range(len(y_true)):
    #     actual_status = "Trained (leaked)" if y_true[i] == 1 else "Not included in Target Training Dataset"
    #     predicted_status = "Trained (leaked)" if y_pred[i] == 1 else "Not included in Target Training Dataset"
    #     correctness = "Correctly" if y_true[i] == y_pred[i] else "Incorrectly"
    #     print(f"Sample {i}: {correctness} identified. Actual: {actual_status}, Predicted: {predicted_status}")

    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
