import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_hist(df, colname):
    df.hist(column=colname, density=True)
    plt.title('{}, mean={}, std={}'.format(colname, "{:.2f}".format(np.mean(df[colname])), "{:.2f}".format(np.std(df[colname]))))
    plt.savefig('{}.png'.format(colname))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualize results.')
    parser.add_argument('--df_path', type=str,
                        help='path to df')

    args = parser.parse_args()

    compos = pd.read_csv(args.df_path)
    for colname in compos.columns:
        plot_hist(compos, colname)
