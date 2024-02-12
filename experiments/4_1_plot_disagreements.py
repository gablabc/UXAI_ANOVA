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


if __name__ == "__main__":

    # Load the results
    df = []
    temp = None
    for seed in range(5):
        temp = pd.read_csv(os.path.join(f"local_disagreements_{seed}.csv"))
        temp["seed"] = seed
        temp["local"] = True
        df.append(temp)
        temp = pd.read_csv(os.path.join(f"global_disagreements_{seed}.csv"))
        temp["seed"] = seed
        temp["local"] = False
        df.append(temp)
    df = pd.concat(df, ignore_index=True)
    del temp

    ##################################################################################
    # Box plots of the disagreement for various depths and LoA. Here we aggregate over
    # the dataset, model type and random seed. We also conduct paired Student-t tests
    # comparing each of the LoA with the Random Splits and CART as a sanity check
    ##################################################################################

    # L2 disagreement
    for local, group_ in df.groupby('local'):
        df_ = []
        for (seed, model, data), group__ in group_.groupby(['seed', 'model', 'dataset']):
            # Normalize so that the disagreement on total dataset is 1
            temp = group__[["partition", "l2disagreement", "max_depth"]].copy(deep=True)
            factor =  temp[temp["max_depth"]==0]["l2disagreement"].max()
            temp["l2disagreement"] = temp["l2disagreement"] / factor
            temp = temp[temp["max_depth"] >= 1]
            df_.append(temp)
        df_ = pd.concat(df_, ignore_index=True)
        
        order = [1, 2, 3]
        hue_order = ["random", "cart", "gadget-pdp", "l2coe", "pfi"]
        plt.figure()
        sns.boxplot(x="max_depth", y="l2disagreement", hue="partition", data=df_, width=0.45, 
                                                    order=order, hue_order=hue_order)
        plt.legend('',frameon=False)
        plt.ylabel("Explanation Disagreement")
        plt.xlabel("Maximum Depth")
        plt.ylim(0, 1)
        plt.savefig(os.path.join("Images", f"results_local_{local}.pdf"), bbox_inches='tight')

        # Paired student-t tests
        test = ttest_rel
        print(f"\n#### Dataset-Local-{local} #####\n")
        for baseline in ["random", "cart"]:
            t_1, p_val_1 = test(df_[df_["partition"]=="l2coe"]["l2disagreement"], 
                                df_[df_["partition"]==baseline]["l2disagreement"], alternative='less')
            t_2, p_val_2 = test(df_[df_["partition"]=="gadget-pdp"]["l2disagreement"], 
                                df_[df_["partition"]==baseline]["l2disagreement"], alternative='less')
            t_3, p_val_3 = test(df_[df_["partition"]=="pfi"]["l2disagreement"], 
                                df_[df_["partition"]==baseline]["l2disagreement"], alternative='less')
            
            print(f"L2CoE vs {baseline}  t-stat {t_1:1.2e} p-val {100*p_val_1:.2e}")
            print(f"GADGET vs {baseline}  t-stat {t_2:1.2e} p-val {100*p_val_2:.2e}")
            print(f"PFI vs {baseline}  t-stat {t_3:1.2e} p-val {100*p_val_3:.2e}\n")


    # Plot the legend separately
    plt.figure(figsize=(5, 0.6))
    partitions = ["Random", "CART", "GADGET-PDP", "CoE", "PDP-PFI"]
    for p in range(len(partitions)):
        plt.scatter(0, 0, label=partitions[p])
    plt.legend(loc='center', ncol=5, prop={"size": 10}, framealpha=1)
    plt.axis('off')
    plt.savefig(os.path.join("Images", f"results_legend.pdf"), bbox_inches='tight', pad_inches=0)


    # L2 norm
    for local, group_ in df.groupby('local'):
        df_ = []
        for (seed, model, data), group__ in group_.groupby(['seed', 'model', 'dataset']):
            # Normalize so that the disagreement on total dataset is 1
            temp = group__[["partition", "l2norm", "max_depth"]].copy(deep=True)
            factor =  temp[temp["max_depth"]==0]["l2norm"].max()
            temp["l2norm"] = temp["l2norm"] / factor
            temp = temp[temp["max_depth"] >= 1]
            df_.append(temp)
        df_ = pd.concat(df_, ignore_index=True)
        
        order = [1, 2, 3]
        hue_order = ["random", "cart", "gadget-pdp", "l2coe", "pfi"]
        plt.figure()
        sns.boxplot(x="max_depth", y="l2norm", hue="partition", data=df_, width=0.45, 
                                                        order=order, hue_order=hue_order)
        plt.legend('',frameon=False)
        plt.ylabel("Explanation Amplitude")
        plt.xlabel("Maximum Depth")
        plt.ylim(0, 1)
        plt.savefig(os.path.join("Images", f"results_l2norm_local_{local}.pdf"), bbox_inches='tight')


    # Pearson disagreement
    for local, group_ in df.groupby('local'):
        df_ = []
        for (seed, model, data), group__ in group_.groupby(['seed', 'model', 'dataset']):
            # Normalize so that the disagreement on total dataset is 1
            temp = group__[["partition", "pearson", "max_depth"]].copy(deep=True)
            factor =  temp[temp["max_depth"]==0]["pearson"].max()
            temp["pearson"] = temp["pearson"] / factor
            temp = temp[temp["max_depth"] >= 1]
            df_.append(temp)
        df_ = pd.concat(df_, ignore_index=True)
        
        order = [1, 2, 3]
        hue_order = ["random", "cart", "gadget-pdp", "l2coe", "pfi"]
        plt.figure()
        sns.boxplot(x="max_depth", y="pearson", hue="partition", data=df_, width=0.45, 
                                                        order=order, hue_order=hue_order)
        plt.legend('',frameon=False)
        plt.ylabel("Pearson")
        plt.xlabel("Maximum Depth")
        plt.ylim(0, 1)
        plt.savefig(os.path.join("Images", f"results_pearson_local_{local}.pdf"), bbox_inches='tight')

    
    # Spearman disagreement
    for local, group_ in df.groupby('local'):
        df_ = []
        for (seed, model, data), group__ in group_.groupby(['seed', 'model', 'dataset']):
            # Normalize so that the disagreement on total dataset is 1
            temp = group__[["partition", "spearman", "max_depth"]].copy(deep=True)
            factor =  temp[temp["max_depth"]==0]["spearman"].max()
            temp["spearman"] = temp["spearman"] / factor
            temp = temp[temp["max_depth"] >= 1]
            df_.append(temp)
        df_ = pd.concat(df_, ignore_index=True)
        
        order = [1, 2, 3]
        hue_order = ["random", "cart", "gadget-pdp", "l2coe", "pfi"]
        plt.figure()
        sns.boxplot(x="max_depth", y="spearman", hue="partition", data=df_, width=0.45, 
                                                        order=order, hue_order=hue_order)
        plt.legend('',frameon=False)
        plt.ylabel("Spearman")
        plt.xlabel("Maximum Depth")
        plt.ylim(0, 1)
        plt.savefig(os.path.join("Images", f"results_spearman_local_{local}.pdf"), bbox_inches='tight')
    


    ##################################################################################
    # Repeat-Measure ANOVA tests where we compare all three LoA simultaneously instead of
    # conducting a paired test for each pair of LoA. Each `subject` in the test is a
    # combination of the model type, max_depth, and random seed. Hence, for every dataset 
    # and locality we provide p-values for differences in explanation disagreements.
    ##################################################################################
    subject = 0
    for (dataset, local), group_ in df.groupby(['dataset', 'local']):
        df_ = []
        for (seed, max_depth, model), group__ in \
                                        group_.groupby(['seed', 'max_depth', 'model']):
            if max_depth >= 1:
                temp_ = group__.assign(subject=subject)
                subject += 1
                df_.append(temp_[["subject", "partition", "l2disagreement"]])
        df_ = pd.concat(df_, ignore_index=True)
        df_ = df_[~(df_["partition"]=="random")]
        df_ = df_[~(df_["partition"]=="cart")]

        # Repeated Measure ANOVA test for simultaneously comparing 
        # CoE, GADGET-PDP and PFI
        print(f"#### Dataset-{dataset} Local-{local} #####")
        print(AnovaRM(data=df_, depvar='l2disagreement',
                    subject='subject', within=['partition']).fit())
        disagreements = df_.groupby("partition")["l2disagreement"].mean()
        print(disagreements)
        print("\n\n")