# main.py
import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from itertools import combinations
import statsmodels.api as sm


# --- Configuration ---
class Config:
    DATA_DIR = "data"

    # --- Phase 2: Pair Finding Config ---
    # We now enforce that a pair must have BOTH high correlation AND cointegration
    CORRELATION_THRESHOLD = 0.95  # Increased for higher quality pairs
    P_VALUE_THRESHOLD = 0.05
    MIN_DATAPOINTS = 500

    # --- Phase 3: Backtesting Config ---
    PAIRS_FILE = "high_quality_pairs.csv"  # New output file name
    ZSCORE_WINDOW = 21
    ENTRY_THRESHOLD = 2.0
    EXIT_THRESHOLD = 0.5


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log", mode='w'),
        logging.StreamHandler()
    ]
)


# --- UPGRADED: Phase 2 ---
class PairFinder:
    """
    Finds high-quality pairs that meet BOTH correlation and cointegration thresholds.
    """

    def __init__(self, config):
        self.config = config
        self.output_file = self.config.PAIRS_FILE
        self.master_df = self._load_all_data()
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def _load_all_data(self):
        """Loads all symbol data into a single DataFrame."""
        logging.info("Loading all symbol data into memory...")
        files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('.csv')]
        all_series = [
            pd.read_csv(os.path.join(self.config.DATA_DIR, f), index_col='open_time', parse_dates=True)['close'].rename(
                f.replace('.csv', ''))
            for f in tqdm(files, desc="Loading data files")
        ]
        if not all_series: return pd.DataFrame()
        master_df = pd.concat(all_series, axis=1).ffill().astype(np.float32)
        return master_df

    @staticmethod
    def check_cointegration(series1, series2):
        """Performs the cointegration test."""
        from statsmodels.tsa.stattools import adfuller
        try:
            ols_result = sm.OLS(series1, sm.add_constant(series2)).fit()
            # **FIX**: Using .iloc[1] to explicitly access by position and remove warning.
            hedge_ratio = ols_result.params.iloc[1]
            residuals = series1 - hedge_ratio * series2
            p_value = adfuller(residuals)[1]
            return p_value
        except Exception:
            return 1.0  # Return a high p-value on error

    def find_pairs(self):
        """Finds pairs that are both highly correlated and cointegrated."""
        if self.master_df.empty: return

        logging.info("Step 1: Filtering for highly correlated pairs...")
        corr_matrix = self.master_df.corr()
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) < corr_pairs.index.get_level_values(1)]
        candidate_pairs = corr_pairs[corr_pairs > self.config.CORRELATION_THRESHOLD]

        logging.info(
            f"Found {len(candidate_pairs)} pairs with correlation > {self.config.CORRELATION_THRESHOLD}. Now testing for cointegration.")

        high_quality_pairs = []
        pbar = tqdm(candidate_pairs.items(), total=len(candidate_pairs), desc="Testing Candidates")
        for (s1, s2), corr_value in pbar:
            series1, series2 = self.master_df[s1].dropna(), self.master_df[s2].dropna()
            aligned_s1, aligned_s2 = series1.align(series2, join='inner')

            if len(aligned_s1) < self.config.MIN_DATAPOINTS: continue

            p_value = self.check_cointegration(aligned_s1, aligned_s2)
            if p_value < self.config.P_VALUE_THRESHOLD:
                high_quality_pairs.append({
                    'symbol1': s1, 'symbol2': s2, 'p_value': p_value, 'correlation': corr_value
                })

        if high_quality_pairs:
            results_df = pd.DataFrame(high_quality_pairs).sort_values('p_value')
            results_df.to_csv(self.output_file, index=False)
            logging.info(
                f"--- Pair Finding Finished. Found {len(results_df)} high-quality pairs. Saved to {self.output_file} ---")
        else:
            logging.info("--- Pair Finding Finished. No pairs met both criteria. ---")


# --- UPGRADED: Phase 3 ---
class Backtester:
    """
    Backtests pairs with robust equity calculation.
    """

    def __init__(self, config):
        self.config = config
        self.pairs_df = self._load_pairs()
        self.summary_results = []

    def _load_pairs(self):
        try:
            df = pd.read_csv(self.config.PAIRS_FILE)
            logging.info(f"Loaded {len(df)} pairs to backtest from {self.config.PAIRS_FILE}")
            return df
        except FileNotFoundError:
            logging.error(f"Error: Could not find pairs file at {self.config.PAIRS_FILE}. Please run Phase 2 first.")
            return pd.DataFrame()

    def _load_pair_data(self, s1, s2):
        s1_path = os.path.join(Config.DATA_DIR, f"{s1}.csv")
        s2_path = os.path.join(Config.DATA_DIR, f"{s2}.csv")
        df1 = pd.read_csv(s1_path, index_col='open_time', parse_dates=True)['close'].rename(s1)
        df2 = pd.read_csv(s2_path, index_col='open_time', parse_dates=True)['close'].rename(s2)
        merged = pd.concat([df1, df2], axis=1).dropna()
        return merged[s1], merged[s2]

    def run_backtest(self):
        if self.pairs_df.empty: return
        logging.info("--- Starting Backtesting Process ---")

        pbar = tqdm(self.pairs_df.iterrows(), total=len(self.pairs_df), desc="Backtesting Pairs")
        for _, row in pbar:
            s1, s2 = row['symbol1'], row['symbol2']
            pbar.set_description(f"Backtesting {s1}-{s2}")

            series1, series2 = self._load_pair_data(s1, s2)

            model = sm.OLS(series1, sm.add_constant(series2)).fit()
            # **FIX**: Using .iloc[1] to explicitly access by position and remove warning.
            hedge_ratio = model.params.iloc[1]
            spread = series1 - hedge_ratio * series2

            spread_mean = spread.rolling(window=self.config.ZSCORE_WINDOW).mean()
            spread_std = spread.rolling(window=self.config.ZSCORE_WINDOW).std()
            z_score = (spread - spread_mean) / spread_std

            equity = [1.0]
            in_position = None
            trades, wins = 0, 0

            for i in range(1, len(z_score)):
                # **ROBUST EQUITY CALCULATION**
                pnl = 0
                if in_position is not None:
                    risk_unit = spread_std.iloc[i - 1]
                    if risk_unit > 1e-9:
                        pnl = (spread.iloc[i] - spread.iloc[i - 1]) / risk_unit
                    if in_position == 'short':
                        pnl = -pnl

                period_return = 0.01 * pnl
                equity.append(equity[-1] * (1 + period_return))

                # Trading Logic
                current_z = z_score.iloc[i]
                if not in_position:
                    if current_z < -self.config.ENTRY_THRESHOLD:
                        in_position = 'long';
                        entry_price = spread.iloc[i]
                    elif current_z > self.config.ENTRY_THRESHOLD:
                        in_position = 'short';
                        entry_price = spread.iloc[i]
                elif (in_position == 'long' and current_z >= -self.config.EXIT_THRESHOLD) or \
                        (in_position == 'short' and current_z <= self.config.EXIT_THRESHOLD):
                    trades += 1
                    if (in_position == 'long' and spread.iloc[i] > entry_price) or \
                            (in_position == 'short' and spread.iloc[i] < entry_price):
                        wins += 1
                    in_position = None

            returns = pd.Series(equity).pct_change().dropna()
            total_return = (equity[-1] - 1) * 100
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24) if returns.std() > 1e-9 else 0

            peak = pd.Series(equity).expanding(min_periods=1).max()
            drawdown = ((pd.Series(equity) - peak) / peak).min() * 100
            win_rate = (wins / trades) * 100 if trades > 0 else 0

            self.summary_results.append({
                'Symbol 1': s1, 'Symbol 2': s2, 'P-Value': row['p_value'],
                'Correlation': f"{row['correlation']:.4f}", 'Total Return (%)': f"{total_return:.2f}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}", 'Max Drawdown (%)': f"{drawdown:.2f}",
                'Win Rate (%)': f"{win_rate:.2f}", 'Total Trades': trades
            })

        self._generate_report()

    def _generate_report(self):
        if not self.summary_results: return
        summary_df = pd.DataFrame(self.summary_results)
        numeric_cols = ['Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)']
        for col in numeric_cols:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        summary_df.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)

        output_file = 'backtest_summary.xlsx'
        # Install 'openpyxl' to write to .xlsx files
        summary_df.to_excel(output_file, sheet_name='Backtest Summary', index=False)
        logging.info(f"Successfully generated backtest report: {output_file}")


if __name__ == '__main__':
    config = Config()

    # --- STEP 1: Run this part first to find the pairs ---
    # Make sure this is uncommented to generate the new high_quality_pairs.csv
    # finder = PairFinder(config)
    # finder.find_pairs()

    # --- STEP 2: Once Step 1 is done, comment it out and run this part ---
    # Make sure this is commented out on the first run, then uncomment for the second run.
    backtester = Backtester(config)
    backtester.run_backtest()
