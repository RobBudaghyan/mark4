# main.py
import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from itertools import product
import statsmodels.api as sm
from datetime import datetime, timedelta


# --- Configuration ---
class Config:
    DATA_DIR = "data"
    BACKTEST_SUMMARY_FILE = "backtest_summary.xlsx"
    PLAYBOOK_FILE = "trading_playbook.xlsx"

    # --- Optimization Parameter Ranges ---
    OPTIMIZER_RANGES = {
        'z_window': range(14, 49, 7),  # Windows: 14, 21, 28, 35, 42
        'entry_z': np.arange(1.5, 2.75, 0.25),  # Entries: 1.5, 1.75, 2.0, 2.25, 2.5
        'exit_z': np.arange(0.0, 1.0, 0.25)  # Exits: 0.0, 0.25, 0.5, 0.75
    }

    # --- Watchlist Filtering Criteria ---
    # These rules will be applied to the backtest_summary file
    # to select which pairs to include in the final playbook.
    WATCHLIST_CRITERIA = {
        # Tier 1: Elite, extremely high-quality strategies
        'Tier 1': {'min_sharpe': 15.0, 'min_trades': 10, 'max_drawdown': -20},
        # Tier 2: Core, very strong and reliable strategies
        'Tier 2': {'min_sharpe': 7.0, 'min_trades': 15, 'max_drawdown': -30},
        # Tier 3: Good, solid strategies worth monitoring
        'Tier 3': {'min_sharpe': 5.0, 'min_trades': 20, 'max_drawdown': -40},
    }


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log", mode='w'),
        logging.StreamHandler()
    ]
)


# --- Re-usable Functions ---
def load_pair_data(s1, s2, start_date=None):
    """Loads and aligns price data for a given pair, with an optional start date."""
    try:
        s1_path = os.path.join(Config.DATA_DIR, f"{s1}.csv")
        s2_path = os.path.join(Config.DATA_DIR, f"{s2}.csv")
        df1 = pd.read_csv(s1_path, index_col='open_time', parse_dates=True)['close'].rename(s1)
        df2 = pd.read_csv(s2_path, index_col='open_time', parse_dates=True)['close'].rename(s2)

        merged = pd.concat([df1, df2], axis=1)

        if start_date:
            merged = merged[merged.index >= pd.to_datetime(start_date)]

        merged.dropna(inplace=True)
        return merged[s1], merged[s2]
    except FileNotFoundError:
        return None, None


class Backtester:
    """Runs a single backtest with a given set of parameters."""

    @staticmethod
    def run(series1, series2, z_window, entry_z, exit_z):
        """Runs the core backtest logic and returns key metrics."""
        try:
            model = sm.OLS(series1, sm.add_constant(series2)).fit()
            hedge_ratio = model.params.iloc[1]
            spread = series1 - hedge_ratio * series2

            spread_mean = spread.rolling(window=z_window).mean()
            spread_std = spread.rolling(window=z_window).std()
            z_score = (spread - spread_mean) / spread_std

            equity = [1.0]
            in_position = None
            trades = 0

            for i in range(1, len(z_score)):
                pnl = 0
                if in_position is not None:
                    risk_unit = spread_std.iloc[i - 1]
                    if risk_unit > 1e-9: pnl = (spread.iloc[i] - spread.iloc[i - 1]) / risk_unit
                    if in_position == 'short': pnl = -pnl

                period_return = 0.01 * np.clip(pnl, -5, 5)
                equity.append(equity[-1] * (1 + period_return))

                current_z = z_score.iloc[i]
                if not in_position:
                    if current_z < -entry_z:
                        in_position = 'long'
                    elif current_z > entry_z:
                        in_position = 'short'
                elif (in_position == 'long' and current_z >= -exit_z) or \
                        (in_position == 'short' and current_z <= exit_z):
                    trades += 1;
                    in_position = None

            returns = pd.Series(equity).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24) if returns.std() > 1e-9 else 0

            return sharpe_ratio, trades
        except Exception:
            return -99, 0


class PlaybookGenerator:
    """
    Filters the backtest summary to create a tiered watchlist, then optimizes
    each pair on the list to generate a final trading playbook.
    """

    def __init__(self, config):
        self.config = config
        self.param_ranges = self.config.OPTIMIZER_RANGES
        self.watchlist = self._create_watchlist()
        if os.path.exists(self.config.PLAYBOOK_FILE):
            os.remove(self.config.PLAYBOOK_FILE)

    def _create_watchlist(self):
        """Filters the backtest summary based on tiered criteria."""
        try:
            df = pd.read_excel(self.config.BACKTEST_SUMMARY_FILE)
            logging.info(f"Loaded {len(df)} backtested pairs from {self.config.BACKTEST_SUMMARY_FILE}")
        except FileNotFoundError:
            logging.error(f"Error: Backtest summary file not found at {self.config.BACKTEST_SUMMARY_FILE}")
            return pd.DataFrame()

        df_filtered = pd.DataFrame()
        for tier, criteria in self.config.WATCHLIST_CRITERIA.items():
            tier_df = df[
                (df['Sharpe Ratio'] >= criteria['min_sharpe']) &
                (df['Total Trades'] >= criteria['min_trades']) &
                (df['Max Drawdown (%)'] >= criteria['max_drawdown'])
                ].copy()
            tier_df['Tier'] = tier
            df_filtered = pd.concat([df_filtered, tier_df])

        # Remove duplicates, keeping the one from the highest tier
        df_filtered.drop_duplicates(subset=['Symbol 1', 'Symbol 2'], keep='first', inplace=True)

        logging.info(
            f"Created watchlist with {len(df_filtered)} high-quality pairs across {len(self.config.WATCHLIST_CRITERIA)} tiers.")
        return df_filtered

    def generate_playbook(self):
        """
        Loops through the watchlist, optimizes each pair, and saves the results.
        """
        if self.watchlist.empty:
            logging.warning("Watchlist is empty. Cannot generate playbook.")
            return

        playbook_results = []

        pbar = tqdm(self.watchlist.iterrows(), total=len(self.watchlist), desc="Generating Playbook")
        for _, row in pbar:
            s1, s2 = row['Symbol 1'], row['Symbol 2']
            pbar.set_description(f"Optimizing {s1}-{s2}")

            series1, series2 = load_pair_data(s1, s2)
            if series1 is None: continue

            param_combinations = list(product(
                self.param_ranges['z_window'], self.param_ranges['entry_z'], self.param_ranges['exit_z']
            ))

            best_sharpe = -float('inf')
            best_params = {}

            for params in param_combinations:
                z_win, entry_z, exit_z = params
                sharpe, _ = Backtester.run(series1, series2, z_win, entry_z, exit_z)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'z_window': z_win, 'entry_z': entry_z, 'exit_z': exit_z}

            if not best_params: continue

            # --- Validation Step ---
            six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            val_s1, val_s2 = load_pair_data(s1, s2, start_date=six_months_ago)
            val_sharpe, val_trades = -99, 0
            if val_s1 is not None and len(val_s1) > best_params['z_window']:
                val_sharpe, val_trades = Backtester.run(val_s1, val_s2, **best_params)

            playbook_results.append({
                'Tier': row['Tier'],
                'Symbol 1': s1, 'Symbol 2': s2,
                'Optimal Z-Window': best_params['z_window'],
                'Optimal Entry-Z': best_params['entry_z'],
                'Optimal Exit-Z': best_params['exit_z'],
                'Best Sharpe (Full History)': f"{best_sharpe:.2f}",
                'Validation Sharpe (6m)': f"{val_sharpe:.2f}",
                'Validation Trades (6m)': val_trades
            })

        # Save the final playbook
        playbook_df = pd.DataFrame(playbook_results)
        playbook_df.to_excel(self.config.PLAYBOOK_FILE, index=False, sheet_name="Trading Playbook")
        logging.info(f"--- Playbook Generation Complete. Saved to {self.config.PLAYBOOK_FILE} ---")


if __name__ == '__main__':
    config = Config()

    # This script now runs the entire final step automatically.
    # It assumes 'backtest_summary.xlsx' exists.
    playbook_generator = PlaybookGenerator(config)
    playbook_generator.generate_playbook()
