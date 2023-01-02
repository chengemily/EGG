import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle


def plot_hist(df, colname):
    df.hist(column=colname, density=True)
    plt.title('{}, mean={}, std={}, range=[{},{}]'.format(
        colname,
        "{:.2f}".format(np.mean(df[colname])),
        "{:.2f}".format(np.std(df[colname])),
        "{:.2f}".format(np.min(df[colname])),
        "{:.2f}".format(np.max(df[colname]))
    ))
    plt.savefig('{}.png'.format(colname))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualize results.')
    parser.add_argument('--df_path', type=str,
                        help='path to df')

    args = parser.parse_args()

    # compos = pd.read_csv(args.df_path)

    import os
    all_checkpoints = []
    for file in os.listdir('saved_models/'):
        f = os.path.join('saved_models/', file)
        with open(f, 'rb') as f_:
            all_checkpoints.append(pickle.load(f_)['last_validation_compo_metrics'])

    df = pd.DataFrame(all_checkpoints)
    for colname in df.columns:
        plot_hist(df, colname)