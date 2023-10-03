""" Plot the result of all attacks, i.e. Figure 5 in the paper """
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
import seaborn as sns

from utils import setup_pyplot_font

setup_pyplot_font(25)

COLORS = {"l2coe": 'b', "random": 'o', "gadget-pdp": 'b', "pfi" : "g"}

if __name__ == "__main__":

    # Load the results
    df_local = pd.read_csv(os.path.join("local_disagreements.csv"))
    df_global = pd.read_csv(os.path.join("global_disagreements.csv"))
    df_local["local"] = True
    df_global["local"] = False
    df = pd.concat((df_local, df_global), ignore_index=True)


    ##################################################################################
    # Box plots of the disagreement for various depths and LoA. Here we aggregate over
    # the dataset, model type and random seed. We also conduct paired Student-t tests
    # comparing each of the LoA with the Random Splits as a sanity check
    ##################################################################################
    for local, group_ in df.groupby(['local']):
        df_ = []
        for (seed, model, data), group__ in \
                                            group_.groupby(['seed', 'model', 'dataset']):
            # Normalize so that the disagreement on total dataset is 1
            temp = group__[["partition", "disagreement", "max_depth"]].copy(deep=True)
            factor =  temp[temp["max_depth"]==0]["disagreement"].max()
            temp["disagreement"] = temp["disagreement"] / factor
            temp = temp[temp["max_depth"] >= 1]
            df_.append(temp)
        df_ = pd.concat(df_, ignore_index=True)
        
        order = [1, 2, 3]
        hue_order = ["random", "l2coe", "pfi", "gadget-pdp"]
        plt.figure()
        sns.boxplot(x="max_depth", y="disagreement", hue="partition", data=df_, width=0.45, 
                                                        order=order, hue_order=hue_order)
        plt.legend('',frameon=False)
        plt.ylabel("Normalized Disagreement")
        plt.xlabel("Maximum Depth")
        plt.ylim(0, 1)
        plt.savefig(os.path.join("Images", f"results_local_{local}.pdf"), bbox_inches='tight')

        # Paired student-t tests
        test = ttest_rel
        t_1, p_val_1 = test(df_[df_["partition"]=="l2coe"]["disagreement"], 
                            df_[df_["partition"]=="random"]["disagreement"], alternative='less')
        t_2, p_val_2 = test(df_[df_["partition"]=="gadget-pdp"]["disagreement"], 
                            df_[df_["partition"]=="random"]["disagreement"], alternative='less')
        t_3, p_val_3 = test(df_[df_["partition"]=="pfi"]["disagreement"], 
                            df_[df_["partition"]=="random"]["disagreement"], alternative='less')
        
        print(f"#### Dataset-Local-{local} #####")
        print(f"L2CoE vs Random  t-stat {t_1:1.2e} p-val {100*p_val_1:.2e}")
        print(f"GADGET vs Random  t-stat {t_2:1.2e} p-val {100*p_val_2:.2e}")
        print(f"PFI vs Random  t-stat {t_3:1.2e} p-val {100*p_val_3:.2e}\n")


    # Plot the legend separately
    plt.figure(figsize=(5, 0.6))
    partitions = ["Random", "CoE", "PFI-PDP", "GADGET-PDP"]
    for p in range(4):
        plt.scatter(0, 0, label=partitions[p])
    plt.legend(loc='center', ncol=4, prop={"size": 10}, framealpha=1)
    plt.axis('off')
    plt.savefig(os.path.join("Images", f"results_legend.pdf"), bbox_inches='tight', pad_inches=0)




    ##################################################################################
    # Repeat-Measure ANOVA tests where we compare all three LoA simultanously instead of
    # conducting a paired test for each pair of LoA. Here we aggregate over
    # the model type, max_depth, and random seed. Hence, for every dataset and locality we
    # provide p-values for significant differences in explanation disagreements.
    ##################################################################################
    subject = 0
    for (dataset, local), group_ in df.groupby(['dataset', 'local']):
        df_ = []
        for (seed, max_depth, model), group__ in \
                                            group_.groupby(['seed', 'max_depth', 'model']):
            if max_depth >= 1:
                temp_ = group__.assign(subject=subject)
                subject += 1
                df_.append(temp_[["subject", "partition", "disagreement"]])
        df_ = pd.concat(df_, ignore_index=True)
        df_ = df_[~(df_["partition"]=="random")]

        # Repeated Measure ANOVA test for simultaneously comparing 
        # L2CoE, GADGET-PDP and PFI
        print(f"#### Dataset-{dataset} Local-{local} #####")
        print(AnovaRM(data=df_, depvar='disagreement',
                    subject='subject', within=['partition']).fit())
        disagreements = df_.groupby("partition")["disagreement"].mean()
        print(disagreements)
        print("\n\n")