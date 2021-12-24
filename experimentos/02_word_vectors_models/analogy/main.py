import pandas as pd

files = [
    "capital-common-countries.csv",
    "capital-world.csv",
    "city-in-state.csv",
    "currency.csv",
    "family.csv",
    "gram1-adjective-to-adverb.csv",
    "gram2-opposite.csv",
    "gram5-present-participle.csv",
    "gram6-nationality-adjective.csv",
    "gram7-past-tense.csv",
    "gram8-plural.csv",
    "gram9-plural-verbs.csv"
]

def main():
    df = pd.read_csv("./data/{}".format(files[-1]))
    print(df.sample(n=20,random_state=17283))


if __name__ == "__main__":
    main()