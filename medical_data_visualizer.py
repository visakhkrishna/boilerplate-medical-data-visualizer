import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(df['weight']/((df['height']/100)**2)>24.9,1,0)
#df['BMI'] = df['weight']/((df['height']/100)**2)

# 3
df['gluc'] = np.where(df['gluc']>1,1,0)
df['cholesterol'] = np.where(df['cholesterol']>1,1,0)

# 4
def draw_cat_plot():
    # 5
    df_cat= pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])
    df_cat2= df_cat

    # 6
    df_cat =  pd.DataFrame(df_cat.groupby(by=['cardio', 'variable', 'value']).size().reset_index(name='total'))
    

    # 7



    # 8
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975)) &
        (df['ap_lo'] <= df['ap_hi'])
    ]
    heatsize = len(df.index)-len(df_heat.index)

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(corr)



    # 14
    fig, ax = plt.subplots()

    # 15

    ax=sns.heatmap(corr,mask=mask,annot=True,square=True,fmt='0.1f')

    # 16
    fig.savefig('heatmap.png')
    return fig
