import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('acc_mae.csv')
    fig, ax = plt.subplots()
    ax.plot(df['acc'][:22],df['mae'][:22],'o',fillstyle="none",label="2 clases")
    ax.plot(df['acc'][22:44],df['mae'][22:44],'o',fillstyle="none",label="3 clases")
    ax.plot(df['acc'][44:],df['mae'][44:],'o',fillstyle="none",label="5 clases")
    ax.set_xlabel("accuracy")
    ax.set_ylabel("MAE")
    ax.grid(True)
    ax.set_xlim(0,200)
    ax.set_ylim(0,200)
    ax.legend()
    plt.savefig("acc_mae.png")

    df = pd.read_csv('f1_mae.csv')
    fig, ax = plt.subplots()
    ax.plot(df['f1'][:22],df['mae'][:22],'o',fillstyle="none",label="2 clases")
    ax.plot(df['f1'][22:44],df['mae'][22:44],'o',fillstyle="none",label="3 clases")
    ax.plot(df['f1'][44:],df['mae'][44:],'o',fillstyle="none",label="5 clases")
    ax.set_xlim(0,200)
    ax.set_ylim(0,200)
    # ax.set_xticks(list(range(0,201,25)))
    ax.set_xlabel("f1-score")
    ax.set_ylabel("MAE")
    ax.grid(True)
    ax.legend()
    plt.savefig("f1_mae.png")



if __name__ == "__main__":
    main()