import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':
    alov300_pairwise_df = pd.read_json('../../data/alov300++/pairwise.json')

    train_df, test_df = model_selection.train_test_split(alov300_pairwise_df, test_size=0.2, random_state=1234)

    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    train_df.to_json('../../data/alov300++/pairwise_train.json', orient='records')
    test_df.to_json('../../data/alov300++/pairwise_test.json', orient='records')