import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_date(fname:str):
    """
    Parse the date from the filename
    """
    date = fname.split('_')[1].split('.')[0]
    date = pd.to_datetime(date, format='ISO8601', errors='coerce').strftime('%Y-%m-%d')
    return date


def main():
    st.title("\"Tiger 5\" Analysis")
    
    # File uploader
    uploaded_files = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=True)

    attribute_columns = ["3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors", "holes", "score"]
    all_columns = ["date", *attribute_columns]

    if uploaded_files:
        print("*"*40)
        global_df = pd.DataFrame(columns=all_columns, index=range(len(uploaded_files)))
        #st.write(global_df)
        for i, uploaded_file in enumerate(uploaded_files):
            stats_date = parse_date(uploaded_file.name)
            df = pd.read_csv(uploaded_file, header=None, index_col=0)
            global_df.loc[i]["date"] = stats_date#, *df.iloc[:, 0].values]
            global_df.loc[i][attribute_columns] = df.iloc[:, 0].values
            # # df.index.name = 'Attribute'
            # # df.columns = ['Count']
            # st.write(df)
        #global_df['date'] = global_df['date'].dt.strftime('%Y-%m-%d')
        for col in attribute_columns:
            global_df[col] =  global_df[col].astype(float)#pd.to_datetime(global_df['date'])
        global_df['date'] = pd.to_datetime(global_df['date'], format='ISO8601', errors='coerce').dt.date
        global_df = global_df.sort_values(by='date', ascending=True)
        global_df.reset_index(drop=True, inplace=True)
        global_df["18H score"] = np.ceil(global_df["score"]/global_df["holes"] * 18.).astype(int)
        global_df.drop(columns=["holes", "score"], inplace=True)
        global_df["error_sum"] = global_df.drop(columns=["18H score"]).sum(axis=1, numeric_only=True)
        
        st.write(global_df)
        mean_and_goal = pd.DataFrame(global_df.mean(axis=0, numeric_only=True))
        mean_and_goal.columns = ["mean"]
        mean_and_goal["mean_last_five"] = global_df.iloc[-5:, 1:].mean(axis=0)
        mean_and_goal["target"] = pd.Series([1, 2.5, 0.25, 1, 0.25, 0.5, 5., 5.5], index=mean_and_goal.index)
        st.write(mean_and_goal)
        
        for ci, col in enumerate(global_df.columns):
            if col == "date":
                continue
            plot_df = global_df[["date", col]]
            rolling_mean = plot_df[col].rolling(window=5).mean()
            global_mean = plot_df[col].mean()
            fig, ax = plt.subplots()
            ax.bar(x=plot_df["date"], height=plot_df[col], color='blue')
            sns.lineplot(x="date", y=rolling_mean, 
                        data=plot_df, ax=ax,color='red',
                        label=f"Rolling mean (5 rounds):{rolling_mean.mean():.1f}", linestyle='--')
            sns.lineplot(x="date", y=global_mean, 
                        data=plot_df, ax=ax,color='black', linestyle=':', label=f"Global mean:{global_mean.mean():.1f}")
            sns.lineplot(x="date", y=mean_and_goal.loc[col, "target"], 
                        data=plot_df, ax=ax,color='gold', linestyle='-.', 
                        label=f"Target:{mean_and_goal.loc[col, 'target']:.1f}", linewidth=3.)
            ax.legend()
            ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            st.pyplot(fig)
        
        fig, ax = plt.subplots()
        ax.bar(x=plot_df["date"], height=global_df["18H score"], color='blue',label='18H score')
        sns.lineplot(x="date", y="error_sum", data=global_df, ax=ax, color='red', label='Errors per round', linewidth=0.5)
        sns.lineplot(x="date", y=global_df["error_sum"].rolling(window=5).mean(), 
                        data=global_df, ax=ax,color='red', label=f"Rolling mean", linestyle='--')
        
        ax.tick_params(axis='x', labelrotation=45)
        st.pyplot(fig)

    
        fig1, ax1 = plt.subplots()
        global_df[["date", "3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors"]].plot(
            kind='bar', stacked=True, ax=ax1, alpha=0.5) 
        plot_df = global_df[["date", '18H score']]
        rolling_mean = plot_df['18H score'].rolling(window=5).mean()
        error_rolling_std = plot_df["18H score"].rolling(window=5).std()
        ax1.plot(global_df["18H score"], color='black',label='18H score', alpha=1.0)
        ax1.plot(rolling_mean, color='red',label='rolling mean', alpha=1.0)
        ax1.plot(rolling_mean+error_rolling_std, color='black', linestyle=':', label=f"Rolling std", alpha=0.5)
        ax1.plot(rolling_mean-error_rolling_std, color='black', linestyle=':', alpha=0.5)
        ax1.grid()

        ax1.set_xticklabels(global_df["date"])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        global_df_normalise = global_df[["date", "3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors"]].copy()
        global_df_normalise[['3 putt', '<150 miss', 'p5 bogey', 'double bogey', '2 chip', 'mental errors']] = global_df_normalise[['3 putt', '<150 miss', 'p5 bogey', 'double bogey', '2 chip', 'mental errors']].div(global_df_normalise[['3 putt', '<150 miss', 'p5 bogey', 'double bogey', '2 chip', 'mental errors']].sum(axis=1), axis=0)
        global_df_normalise[["date", "3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors"]].plot(
            kind='bar', stacked=True, ax=ax2, alpha=0.5) 
        ax2.set_xticklabels(global_df["date"])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Fraction of total errors')
        st.pyplot(fig2)
        
        
        
    
if __name__ == "__main__":
    main()
