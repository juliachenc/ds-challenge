
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.metrics import classification_report, recall_score, precision_score, auc, precision_recall_curve, make_scorer, confusion_matrix


##########################################################################
####################### Load Data  #######################################
##########################################################################
def load_json_txt_to_df(file_path):

    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip removes any extraneous whitespace/newline characters
            data_dict = json.loads(line.strip())  
            # Replace empty strings or None with NaN
            data_dict = {k: (v if v not in ["", None] else np.nan) for k, v in data_dict.items()}
            data_list.append(data_dict)

    df = pd.DataFrame(data_list)
    
    return df

##########################################################################
############################# Plot #######################################
##########################################################################
def camel_to_normal(s):
    result = ""
    for char in s:
        if char.isupper():
            result += " " + char.lower()
        else:
            result += char
    return result.strip().title()


def add_box_labels(ax, fmt='.2f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    #print(lines_per_box)
    for min in lines[2:len(lines):lines_per_box]:
        x, y = (data.mean() for data in min.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (min.get_xdata()[1] - min.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='normal', color='white', fontsize=10)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground=min.get_color()),
            path_effects.Normal(),
        ])
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                        fontweight='normal', color='white', fontsize=10)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground=median.get_color()),
            path_effects.Normal(),
        ])
    for max in lines[3:len(lines):lines_per_box]:
        x, y = (data.mean() for data in max.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (max.get_xdata()[1] - max.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='normal', color='white', fontsize=10)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground=min.get_color()),
            path_effects.Normal(),
        ])

def side_by_side_num(df, cat):
    cat_title = camel_to_normal(cat)
    title_1 = f'Density Plot of {cat_title}'
    title_2 = f'Box Plot for Fraud Status on {cat_title} '


    hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), )
    sns.distplot(df[cat][df.isFraud == True], label='Fraud True', 
                ax=ax[0], hist_kws=hist_kws)
    sns.distplot(df[cat][df.isFraud == False], label='Fraud False', 
                ax=ax[0], hist_kws=hist_kws)
    ax[0].set_title(title_1, fontsize=16)
    ax[0].set_xlabel(cat_title)
    ax[0].legend()
    custom_palette = ["#1f77b4", "#ff7f0e"] 

    # Plot with custom colors
    b = sns.boxplot(data=df, y=cat, x="isFraud", palette=custom_palette, ax=ax[1])
    add_box_labels(b)
    ax[1].set_xlabel('Fraud Status')
    ax[1].set_title(title_2, fontsize=16)
    ax[1].set_ylabel(cat_title)

    plt.show()

def side_by_side_cag(df, cat):
    x_label = camel_to_normal(cat)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    ax[0] = sns.countplot(data=df, x=cat, palette="Set3", order = df[cat].value_counts().index,ax=ax[0])
    for p in ax[0].patches:
        ax[0].annotate('{}'.format(p.get_height()), (p.get_x()+0.27, p.get_height()+1), fontsize=12)

    ax[0].set_xlabel(x_label, fontsize=12)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].set_title(f'Count Plot of {x_label}', fontsize=16)


    cross_tab_prop = pd.crosstab(index=df[cat], columns=df['isFraud'], normalize="index")
    cross_tab_prop.sort_values(by = cat, ascending=False, inplace=True)
    cross_tab_prop.plot(kind='bar',  stacked = True, colormap="Pastel2_r", ax=ax[1])
    for n, x in enumerate([*cross_tab_prop.index.values]):
        for (proportion, y_loc) in zip(cross_tab_prop.loc[x], cross_tab_prop.loc[x].cumsum()):
                plt.text(x = n - 0.17, y=y_loc, s=f'{np.round(proportion * 100, 1)}%', fontsize=12)

    ax[1].set_title(f'Percentage of Fraud on {x_label}', fontsize=16)
    ax[1].legend(['Not Fraud','Fraud'], loc=3)
    ax[1].set_xlabel(f"{cat}", fontsize=12)
    ax[1].set_ylabel("Percentage", fontsize=12)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

def side_by_side_long_cag(df, cat):
    normal = camel_to_normal(cat)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 6))
    top_10= df[cat].value_counts().head(10)

    # Plot the top 10 merchant names with count labels on top
    bars = top_10.plot(kind='bar', color='skyblue', ax=ax[0])

    # Add title and labels
    ax[0].set_title(f'Top 10 {normal} by Frequency', fontsize=14)
    ax[0].set_xlabel(normal, fontsize=12)
    ax[0].set_ylabel('Count', fontsize=12)

    # Add count labels on top of the bars
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'), 
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)


    df_fraud = df[df['isFraud'] == True]
    top_10 = df_fraud[cat].value_counts().head(10)
    bars = top_10.plot(kind='bar', color='skyblue')

    ax[1].set_title(f'Top 10 Fraud {normal} by Frequency', fontsize=14)
    ax[1].set_xlabel(normal, fontsize=12)
    ax[1].set_ylabel('Count', fontsize=12)

    # Add count labels on top of the bars
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'), 
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points')
        
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)


    plt.tight_layout()
    plt.show()

##########################################################################
############################# Model ######################################
##########################################################################

def get_model_result1(classifier, threshold, x_test, y_test, x_train, y_train, output_df, title):
    # after oct
    y_pred = (classifier.predict_proba(x_test)[:,1] >= threshold).astype(bool)
    print(f"Test Recall : {recall_score(y_test, y_pred):.2f}")
    print(f"Test Precision : {precision_score(y_test, y_pred):.2f}")

    print(classification_report(y_test, y_pred, zero_division=1))
    print(confusion_matrix(y_test, y_pred))

    y_pred_train=(classifier.predict_proba(x_train)[:,1] >= threshold).astype(bool)
    print(f"Training Recall : {recall_score(y_train, y_pred_train):.2f}")
    print(f"Training Precision : {precision_score(y_train, y_pred_train):.2f}")
    print(classification_report(y_train, y_pred_train, zero_division=1))
    
    output_df[title] = [# score on testing set 
                        recall_score(y_test, y_pred),
                        precision_score(y_test, y_pred),
                        # score on training  
                        recall_score(y_train, y_pred_train),
                        precision_score(y_train, y_pred_train),
                        ]