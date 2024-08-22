import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

FILE_NAME = 'healthcare_dataset.csv'
PICKLE_DIR = 'pickle'

# Load or create GAN datasets
def exist_dataset(file_path):
    if os.path.exists(file_path):
        print(f"Loading datasets from {file_path}")
        return True
    else:
        return False
    
if __name__ == '__main__':
    # 현재 시간을 얻고 형식화하기
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 디렉터리 및 파일 경로 설정
    output_dir = '../tests'
    output_file = f'stdout_{current_time}.txt'
    output_path = os.path.join(output_dir, output_file)

    # 디렉터리 존재 확인 및 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # stdout을 새로운 파일로 리다이렉트
    sys.stdout = open(output_path, 'w')

    print('Started testing RGAN-TL Classifier')
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)

    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)
    
    print("============ RandomForest ============")
    src.classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
        # Create pickle directory if not exists
    
    if not os.path.exists(PICKLE_DIR):
        os.makedirs(PICKLE_DIR)
        
    PICKLE_FILE = os.path.join(PICKLE_DIR, 'gan_dataset.p')

    if exist_dataset(PICKLE_FILE):
        # PICKLE_FILE에서 pickle 파일 로드
        with open(PICKLE_FILE, 'rb') as f:
            gan_dataset, wgan_dataset, wgangp_dataset, sngan_dataset = pickle.load(f)
    else:
        # GAN 데이터셋 생성
        gan_dataset = src.utils.get_gan_dataset(src.gans.GAN())
        wgan_dataset = src.utils.get_gan_dataset(src.gans.WGAN())
        wgangp_dataset = src.utils.get_gan_dataset(src.gans.WGANGP())
        sngan_dataset = src.utils.get_gan_dataset(src.gans.SNGAN())
        
        # PICKLE_FILE에 pickle 파일로 저장
        with open(PICKLE_FILE, 'wb') as f:
            pickle.dump((gan_dataset, wgan_dataset, wgangp_dataset, sngan_dataset), f)
        

    ############ GAN ############
    print("============ RF with GAN ============")
    src.classifier.RandomForest(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ RF with WGAN ============")
    src.classifier.RandomForest(wgan_dataset.samples, wgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ RF with WGANGP ============")
    src.classifier.RandomForest(wgangp_dataset.samples, wgangp_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ RF with SNGANs ============")
    src.classifier.RandomForest(sngan_dataset.samples, sngan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)


