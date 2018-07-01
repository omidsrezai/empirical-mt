import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':
    alov300_df = pd.read_csv('../../data/alov300++/alov300.csv')

    train_df, test_df = model_selection.train_test_split(alov300_df, test_size=0.1, random_state=1234)

    train_df.to_csv('../../data/alov300++/alov300_train.csv', index=False)
    test_df.to_csv('../../data/alov300++/alov300_test.csv', index=False)