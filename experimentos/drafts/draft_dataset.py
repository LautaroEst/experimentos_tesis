from datasets import load_dataset
import pandas as pd



def main():
    dataset = load_dataset("amazon_reviews_multi","es")
    df_train = pd.DataFrame(dataset['train']).loc[:,['review_body','review_title','stars']]
    df_train = df_train.rename(columns={'review_body':'review_content', 'stars': 'review_rate'})
    df_train['review_rate'] = df_train['review_rate'] - 1
    
    df_dev = pd.DataFrame(dataset['validation']).loc[:,['review_body','review_title','stars']]
    df_dev = df_dev.rename(columns={'review_body':'review_content', 'stars': 'review_rate'})
    df_dev['review_rate'] = df_dev['review_rate'] - 1
    
    df_test = pd.DataFrame(dataset['test']).loc[:,['review_body','review_title','stars']]
    df_test = df_test.rename(columns={'review_body':'review_content', 'stars': 'review_rate'})
    df_test['review_rate'] = df_test['review_rate'] - 1
    
    print(df_train)
    print()
    print(df_dev)
    print()
    print(df_test)
    # print(dataset['review_body'].str.len().describe())
    # print()
    # print(dataset['review_summary'].str.len().describe())

if __name__ == "__main__":
    main()