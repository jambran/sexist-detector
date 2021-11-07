import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    log_path = Path('./logs/split_dataset.log')
    logging.basicConfig(handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path),
                                  ],
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        )
    data_dir = Path('./data')
    kaggle_data_file = data_dir / 'ISEP Sexist Data labeling.xlsx'
    logging.info(f'reading in data from {kaggle_data_file}')
    df = pd.read_excel(kaggle_data_file)
    logging.info('data read in')

    # split the data into train, val, and test randomly
    train, evaluation = train_test_split(df,
                                         test_size=.2,
                                         stratify=df['Label'],
                                         )
    val, test = train_test_split(evaluation,
                                 test_size=.5,
                                 stratify=evaluation['Label'],
                                 )

    train.to_csv(data_dir / 'train.csv', index=False)
    val.to_csv(data_dir / 'val.csv', index=False)
    test.to_csv(data_dir / 'test.csv', index=False)
