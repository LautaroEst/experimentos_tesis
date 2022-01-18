from datasets import load_dataset



def main():
    dataset = load_dataset('spanish_billion_words')
    train_dataset = dataset["train"]
    # print(train_dataset)
    # print(train_dataset.info)
    # print(train_dataset.shape)
    # print(train_dataset.features)
    print(train_dataset[:3])
    print(train_dataset[[1,3,5]])


if __name__ == "__main__":
    main()
    