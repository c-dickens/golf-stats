import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def parse_date(fname:str):
    """
    Parse the date from the filename
    """
    date = fname.split('_')[1].split('.')[0]
    date = pd.to_datetime(date, format='ISO8601', errors='coerce').strftime('%Y-%m-%d')
    return date


def calculate_best_eight_of_twenty(series):
    """
    Calculate the average of the best 8 values from the last 20 values
    """
    if len(series) < 20:
        return pd.Series([np.nan] * len(series))
    
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < 19:  # Not enough data for first 19 points
            result.iloc[i] = np.nan
        else:
            last_20 = series.iloc[i-19:i+1]
            best_8 = last_20.nsmallest(8)
            result.iloc[i] = best_8.mean()
    return result


def apply_smoothing(series, method='exponential', **kwargs):
    """
    Apply various smoothing methods to reduce noise in the data
    
    Parameters:
    - series: pandas Series to smooth
    - method: 'exponential', 'savgol', 'gaussian', 'rolling', or 'combined'
    - **kwargs: method-specific parameters
    """
    if len(series) < 3:
        return series
    
    if method == 'exponential':
        alpha = kwargs.get('alpha', 0.3)
        return series.ewm(alpha=alpha).mean()
    
    elif method == 'savgol':
        window_length = min(kwargs.get('window_length', 5), len(series) - 1)
        if window_length % 2 == 0:
            window_length += 1  # Savitzky-Golay requires odd window length
        polyorder = min(kwargs.get('polyorder', 2), window_length - 1)
        return pd.Series(savgol_filter(series.dropna(), window_length, polyorder), 
                        index=series.dropna().index)
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return pd.Series(gaussian_filter1d(series.dropna(), sigma=sigma), 
                        index=series.dropna().index)
    
    elif method == 'rolling':
        window = kwargs.get('window', 5)
        return series.rolling(window=window, center=True).mean()
    
    elif method == 'combined':
        # Combine exponential and Savitzky-Golay for best results
        exp_smooth = series.ewm(alpha=0.3).mean()
        window_length = min(5, len(exp_smooth) - 1)
        if window_length % 2 == 0:
            window_length += 1
        return pd.Series(savgol_filter(exp_smooth.dropna(), window_length, 2), 
                        index=exp_smooth.dropna().index)
    
    else:
        return series


def main():
    st.title("\"Tiger 5\" Analysis")
    
    # Smoothing configuration
    st.sidebar.header("Smoothing Settings")
    smoothing_method = st.sidebar.selectbox(
        "Smoothing Method",
        ['exponential', 'savgol', 'gaussian', 'rolling', 'combined'],
        help="Choose the smoothing method for trend lines"
    )
    
    # Method-specific parameters
    if smoothing_method == 'exponential':
        alpha = st.sidebar.slider("Alpha (smoothing factor)", 0.1, 0.9, 0.3, 0.1,
                                 help="Lower values = more smoothing")
    elif smoothing_method == 'savgol':
        window_length = st.sidebar.slider("Window Length", 3, 11, 5, 2,
                                         help="Must be odd number, higher = more smoothing")
        polyorder = st.sidebar.slider("Polynomial Order", 1, 3, 2, 1,
                                     help="Higher order preserves more features")
    elif smoothing_method == 'gaussian':
        sigma = st.sidebar.slider("Sigma", 0.5, 3.0, 1.0, 0.1,
                                 help="Higher values = more smoothing")
    elif smoothing_method == 'rolling':
        window = st.sidebar.slider("Window Size", 3, 15, 5, 1,
                                  help="Number of points to average")
    
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
            # # st.write(df)
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
            
            # Apply improved smoothing
            if smoothing_method == 'exponential':
                smoothed_data = apply_smoothing(plot_df[col], method=smoothing_method, alpha=alpha)
            elif smoothing_method == 'savgol':
                smoothed_data = apply_smoothing(plot_df[col], method=smoothing_method, 
                                              window_length=window_length, polyorder=polyorder)
            elif smoothing_method == 'gaussian':
                smoothed_data = apply_smoothing(plot_df[col], method=smoothing_method, sigma=sigma)
            elif smoothing_method == 'rolling':
                smoothed_data = apply_smoothing(plot_df[col], method=smoothing_method, window=window)
            else:  # combined
                smoothed_data = apply_smoothing(plot_df[col], method=smoothing_method)
            
            rolling_mean = plot_df[col].rolling(window=5).mean()
            best_eight_avg = calculate_best_eight_of_twenty(plot_df[col])
            global_mean = plot_df[col].mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x=plot_df["date"], height=plot_df[col], color='lightblue', alpha=0.7, label='Raw Data')
            
            # Plot smoothed trend line
            if not smoothed_data.empty:
                sns.lineplot(x="date", y=smoothed_data, 
                            data=plot_df, ax=ax, color='red', linewidth=2,
                            label=f"{smoothing_method.title()} Smoothing", linestyle='-')
            
            sns.lineplot(x="date", y=rolling_mean, 
                        data=plot_df, ax=ax, color='orange',
                        label=f"Rolling mean (5 rounds):{rolling_mean.mean():.1f}", linestyle='--')
            sns.lineplot(x="date", y=best_eight_avg,
                        data=plot_df, ax=ax, color='green',
                        label=f"Best 8 of 20 avg:{best_eight_avg.mean():.1f}", linestyle='-')
            sns.lineplot(x="date", y=global_mean, 
                        data=plot_df, ax=ax, color='black', linestyle=':', 
                        label=f"Global mean:{global_mean.mean():.1f}")
            sns.lineplot(x="date", y=mean_and_goal.loc[col, "target"], 
                        data=plot_df, ax=ax, color='gold', linestyle='-.', 
                        label=f"Target:{mean_and_goal.loc[col, 'target']:.1f}", linewidth=3.)
            ax.axvline(x=datetime(2025, 1, 1), color='red', linestyle='--', alpha=0.5, label='2025')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{col} - {smoothing_method.title()} Smoothing")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Summary plot with improved smoothing
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x=plot_df["date"], height=global_df["18H score"], color='lightblue', alpha=0.7, label='18H score')
        
        # Apply smoothing to error sum
        if smoothing_method == 'exponential':
            error_smoothed = apply_smoothing(global_df["error_sum"], method=smoothing_method, alpha=alpha)
        elif smoothing_method == 'savgol':
            error_smoothed = apply_smoothing(global_df["error_sum"], method=smoothing_method, 
                                           window_length=window_length, polyorder=polyorder)
        elif smoothing_method == 'gaussian':
            error_smoothed = apply_smoothing(global_df["error_sum"], method=smoothing_method, sigma=sigma)
        elif smoothing_method == 'rolling':
            error_smoothed = apply_smoothing(global_df["error_sum"], method=smoothing_method, window=window)
        else:  # combined
            error_smoothed = apply_smoothing(global_df["error_sum"], method=smoothing_method)
        
        if not error_smoothed.empty:
            sns.lineplot(x="date", y=error_smoothed, 
                        data=global_df, ax=ax, color='red', linewidth=2,
                        label=f'Errors ({smoothing_method.title()} Smoothing)', linestyle='-')
        
        sns.lineplot(x="date", y=global_df["error_sum"].rolling(window=5).mean(), 
                    data=global_df, ax=ax, color='orange', label=f"Rolling mean", linestyle='--')
        best_eight_avg = calculate_best_eight_of_twenty(global_df["error_sum"])
        sns.lineplot(x="date", y=best_eight_avg,
                    data=global_df, ax=ax, color='green',
                    label=f"Best 8 of 20 avg:{best_eight_avg.mean():.1f}", linestyle='-')
        ax.axvline(x=datetime(2025, 1, 1), color='red', linestyle='--', alpha=0.5, label='2025')
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Summary - {smoothing_method.title()} Smoothing")
        plt.tight_layout()
        st.pyplot(fig)

        # Stacked bar chart with improved styling
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        global_df[["date", "3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors"]].plot(
            kind='bar', stacked=True, ax=ax1, alpha=0.7) 
        plot_df = global_df[["date", '18H score']]
        
        # Apply smoothing to 18H score
        if smoothing_method == 'exponential':
            score_smoothed = apply_smoothing(plot_df['18H score'], method=smoothing_method, alpha=alpha)
        elif smoothing_method == 'savgol':
            score_smoothed = apply_smoothing(plot_df['18H score'], method=smoothing_method, 
                                           window_length=window_length, polyorder=polyorder)
        elif smoothing_method == 'gaussian':
            score_smoothed = apply_smoothing(plot_df['18H score'], method=smoothing_method, sigma=sigma)
        elif smoothing_method == 'rolling':
            score_smoothed = apply_smoothing(plot_df['18H score'], method=smoothing_method, window=window)
        else:  # combined
            score_smoothed = apply_smoothing(plot_df['18H score'], method=smoothing_method)
        
        rolling_mean = plot_df['18H score'].rolling(window=5).mean()
        best_eight_avg = calculate_best_eight_of_twenty(plot_df['18H score'])
        error_rolling_std = plot_df["18H score"].rolling(window=5).std()
        
        ax1.plot(global_df["18H score"], color='black', label='18H score', alpha=1.0, linewidth=1)
        if not score_smoothed.empty:
            ax1.plot(score_smoothed, color='red', label=f'{smoothing_method.title()} Smoothing', alpha=1.0, linewidth=2)
        ax1.plot(rolling_mean, color='orange', label='rolling mean', alpha=1.0)
        ax1.plot(best_eight_avg, color='green', label='Best 8 of 20 avg', alpha=1.0)
        ax1.plot(rolling_mean+error_rolling_std, color='black', linestyle=':', label=f"Rolling std", alpha=0.5)
        ax1.plot(rolling_mean-error_rolling_std, color='black', linestyle=':', alpha=0.5)
        ax1.axvline(x=datetime(2025, 1, 1), color='red', linestyle='--', alpha=0.5, label='2025')
        ax1.grid(True, alpha=0.3)

        ax1.set_xticklabels(global_df["date"])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f"Error Breakdown with {smoothing_method.title()} Smoothing")
        plt.tight_layout()
        st.pyplot(fig1)

        # Normalized stacked bar chart
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        global_df_normalise = global_df[["date", "3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors"]].copy()
        global_df_normalise[['3 putt', '<150 miss', 'p5 bogey', 'double bogey', '2 chip', 'mental errors']] = global_df_normalise[['3 putt', '<150 miss', 'p5 bogey', 'double bogey', '2 chip', 'mental errors']].div(global_df_normalise[['3 putt', '<150 miss', 'p5 bogey', 'double bogey', '2 chip', 'mental errors']].sum(axis=1), axis=0)
        global_df_normalise[["date", "3 putt", "<150 miss", "p5 bogey", "double bogey", "2 chip", "mental errors"]].plot(
            kind='bar', stacked=True, ax=ax2, alpha=0.7) 
        ax2.axvline(x=datetime(2025, 1, 1), color='red', linestyle='--', alpha=0.5, label='2025')
        ax2.set_xticklabels(global_df["date"])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Fraction of total errors')
        ax2.set_title('Error Distribution (Normalized)')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        
        
    
if __name__ == "__main__":
    main()
