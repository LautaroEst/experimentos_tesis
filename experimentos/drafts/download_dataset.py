from datasets import load_dataset



def main():
    dataset = load_dataset(
    'spanish_billion_words')
    
    print(dataset['train'][:10])

if __name__ == "__main__":
    main()
    