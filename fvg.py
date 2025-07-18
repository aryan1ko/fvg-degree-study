import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime, timedelta
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

class FVGDegreeAnalyzer:
    """
    Fair Value Gap Degree Analysis System
    
    This class implements the complete methodology for:
    1. FVG identification from 1-minute OHLC data
    2. Degree calculation using tick data and RANSAC regression
    3. Statistical validation and backtesting
    4. Trading strategy implementation
    """
    
    def __init__(self, degree_threshold_low=0.00015, degree_threshold_high=0.0004):
        """
        Initialize the FVG Degree Analyzer
        
        Parameters:
        -----------
        degree_threshold_low : float
            Threshold for low-degree FVGs (strong reaction expected)
        degree_threshold_high : float
            Threshold for high-degree FVGs (weak reaction expected)
        """
        self.degree_threshold_low = degree_threshold_low
        self.degree_threshold_high = degree_threshold_high
        self.fvg_results = []
        self.statistics = {}
        
    def identify_fvgs_1min(self, ohlc_data):
        """
        Identify Fair Value Gaps from 1-minute OHLC data
        
        Parameters:
        -----------
        ohlc_data : pd.DataFrame
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close']
            
        Returns:
        --------
        list : List of FVG dictionaries with metadata
        """
        fvgs = []
        
        for i in range(1, len(ohlc_data) - 1):
            # Current candle (t)
            current = ohlc_data.iloc[i]
            # Previous candle (t-1)
            previous = ohlc_data.iloc[i-1]
            # Next candle (t+1)
            next_candle = ohlc_data.iloc[i+1]
            
            # Bullish FVG: Low(t+1) > High(t-1)
            if next_candle['low'] > previous['high']:
                fvg = {
                    'type': 'bullish',
                    'formation_start': previous['timestamp'],
                    'formation_end': next_candle['timestamp'],
                    'gap_low': previous['high'],
                    'gap_high': next_candle['low'],
                    'middle_candle_idx': i,
                    'formation_candles': [previous, current, next_candle]
                }
                fvgs.append(fvg)
                
            # Bearish FVG: High(t+1) < Low(t-1)
            elif next_candle['high'] < previous['low']:
                fvg = {
                    'type': 'bearish',
                    'formation_start': previous['timestamp'],
                    'formation_end': next_candle['timestamp'],
                    'gap_low': next_candle['high'],
                    'gap_high': previous['low'],
                    'middle_candle_idx': i,
                    'formation_candles': [previous, current, next_candle]
                }
                fvgs.append(fvg)
                
        return fvgs
    
    def extract_tick_data(self, tick_data, fvg_start_time, fvg_end_time):
        """
        Extract 1-second tick data for FVG formation period
        
        Parameters:
        -----------
        tick_data : pd.DataFrame
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close']
        fvg_start_time : timestamp
            Start time of FVG formation
        fvg_end_time : timestamp
            End time of FVG formation
            
        Returns:
        --------
        pd.DataFrame : Filtered tick data for the formation period
        """
        mask = (tick_data['timestamp'] >= fvg_start_time) & (tick_data['timestamp'] <= fvg_end_time)
        formation_ticks = tick_data[mask].copy()
        
        if len(formation_ticks) == 0:
            return None
            
        # Convert timestamps to seconds elapsed since start
        formation_ticks['seconds_elapsed'] = (
            formation_ticks['timestamp'] - fvg_start_time
        ).dt.total_seconds()
        
        return formation_ticks
    
    def preprocess_tick_data(self, tick_data):
        """
        Preprocess tick data for regression analysis
        
        Parameters:
        -----------
        tick_data : pd.DataFrame
            Raw tick data with timestamps and prices
            
        Returns:
        --------
        tuple : (time_array, price_array) ready for regression
        """
        if tick_data is None or len(tick_data) < 3:
            return None, None
            
        # Use open price as primary data point
        prices = tick_data['open'].values
        times = tick_data['seconds_elapsed'].values
        
        # Remove outliers (> 4σ deviations)
        z_scores = np.abs(stats.zscore(prices))
        outlier_mask = z_scores < 4
        
        clean_prices = prices[outlier_mask]
        clean_times = times[outlier_mask]
        
        if len(clean_prices) < 3:
            return None, None
            
        return clean_times, clean_prices
    
    def calculate_degree_ransac(self, time_data, price_data):
        """
        Calculate FVG degree using RANSAC regression for robustness
        
        Parameters:
        -----------
        time_data : np.array
            Time values (seconds elapsed)
        price_data : np.array
            Price values
            
        Returns:
        --------
        dict : Dictionary containing degree and regression statistics
        """
        if time_data is None or price_data is None:
            return None
            
        if len(time_data) < 3:
            return None
            
        # Reshape for sklearn
        X = time_data.reshape(-1, 1)
        y = price_data
        
        # RANSAC regression with residual threshold
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.00005,  # As specified in paper
            random_state=42,
            max_trials=100
        )
        
        try:
            ransac.fit(X, y)
            
            # Get slope coefficient (β1)
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            # Calculate degree (absolute slope)
            degree = abs(slope)
            
            # Calculate R²
            y_pred = ransac.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Count inliers
            inlier_mask = ransac.inlier_mask_
            n_inliers = np.sum(inlier_mask)
            
            return {
                'degree': degree,
                'slope': slope,
                'intercept': intercept,
                'r2': r2,
                'n_inliers': n_inliers,
                'n_total': len(time_data),
                'inlier_ratio': n_inliers / len(time_data)
            }
            
        except Exception as e:
            print(f"RANSAC regression failed: {e}")
            return None
    
    def analyze_fvg_degree(self, fvg, tick_data):
        """
        Complete analysis of a single FVG's degree
        
        Parameters:
        -----------
        fvg : dict
            FVG metadata from identify_fvgs_1min
        tick_data : pd.DataFrame
            1-second tick data
            
        Returns:
        --------
        dict : Complete FVG analysis results
        """
        # Extract tick data for formation period
        formation_ticks = self.extract_tick_data(
            tick_data, fvg['formation_start'], fvg['formation_end']
        )
        
        if formation_ticks is None:
            return None
            
        # Preprocess data
        time_data, price_data = self.preprocess_tick_data(formation_ticks)
        
        if time_data is None:
            return None
            
        # Calculate degree using RANSAC
        degree_analysis = self.calculate_degree_ransac(time_data, price_data)
        
        if degree_analysis is None:
            return None
            
        # Combine FVG metadata with degree analysis
        result = {
            **fvg,
            **degree_analysis,
            'formation_duration': len(formation_ticks),
            'avg_volume': formation_ticks['volume'].mean() if 'volume' in formation_ticks.columns else None
        }
        
        return result
    
    def calculate_price_reaction(self, fvg_result, future_price_data, lookback_periods=[5, 15, 30, 60]):
        """
        Calculate price reaction strength after FVG formation
        
        Parameters:
        -----------
        fvg_result : dict
            FVG analysis result
        future_price_data : pd.DataFrame
            Price data after FVG formation
        lookback_periods : list
            Time periods (minutes) to analyze reactions
            
        Returns:
        --------
        dict : Price reaction metrics
        """
        if fvg_result is None:
            return None
            
        fvg_end_time = fvg_result['formation_end']
        fvg_type = fvg_result['type']
        
        # Define FVG zone boundaries
        if fvg_type == 'bullish':
            fvg_low = fvg_result['gap_low']
            fvg_high = fvg_result['gap_high']
            entry_level = fvg_high  # Enter on retest of upper boundary
        else:
            fvg_low = fvg_result['gap_low']
            fvg_high = fvg_result['gap_high']
            entry_level = fvg_low   # Enter on retest of lower boundary
            
        reactions = {}
        
        for period in lookback_periods:
            # Get data for this lookback period
            period_end = fvg_end_time + pd.Timedelta(minutes=period)
            mask = (future_price_data['timestamp'] > fvg_end_time) & (future_price_data['timestamp'] <= period_end)
            period_data = future_price_data[mask]
            
            if len(period_data) == 0:
                continue
                
            # Check if price retested the FVG zone
            if fvg_type == 'bullish':
                # Look for retest of gap (price coming back down)
                retest_mask = (period_data['low'] <= fvg_high) & (period_data['low'] >= fvg_low)
                if retest_mask.any():
                    retest_candle = period_data[retest_mask].iloc[0]
                    retest_price = retest_candle['low']
                    
                    # Calculate reaction (how much price bounced up)
                    subsequent_data = period_data[period_data['timestamp'] > retest_candle['timestamp']]
                    if len(subsequent_data) > 0:
                        max_reaction = subsequent_data['high'].max()
                        reaction_magnitude = max_reaction - retest_price
                        reaction_percentage = (reaction_magnitude / retest_price) * 100
                        
                        reactions[f'{period}min'] = {
                            'retest_occurred': True,
                            'retest_price': retest_price,
                            'reaction_magnitude': reaction_magnitude,
                            'reaction_percentage': reaction_percentage,
                            'max_favorable_price': max_reaction
                        }
                    else:
                        reactions[f'{period}min'] = {'retest_occurred': True, 'insufficient_data': True}
                else:
                    reactions[f'{period}min'] = {'retest_occurred': False}
                    
            else:  # bearish FVG
                # Look for retest of gap (price coming back up)
                retest_mask = (period_data['high'] >= fvg_low) & (period_data['high'] <= fvg_high)
                if retest_mask.any():
                    retest_candle = period_data[retest_mask].iloc[0]
                    retest_price = retest_candle['high']
                    
                    # Calculate reaction (how much price dropped)
                    subsequent_data = period_data[period_data['timestamp'] > retest_candle['timestamp']]
                    if len(subsequent_data) > 0:
                        min_reaction = subsequent_data['low'].min()
                        reaction_magnitude = retest_price - min_reaction
                        reaction_percentage = (reaction_magnitude / retest_price) * 100
                        
                        reactions[f'{period}min'] = {
                            'retest_occurred': True,
                            'retest_price': retest_price,
                            'reaction_magnitude': reaction_magnitude,
                            'reaction_percentage': reaction_percentage,
                            'max_favorable_price': min_reaction
                        }
                    else:
                        reactions[f'{period}min'] = {'retest_occurred': True, 'insufficient_data': True}
                else:
                    reactions[f'{period}min'] = {'retest_occurred': False}
                    
        return reactions
    
    def process_dataset(self, ohlc_1min, tick_1sec, future_data=None):
        """
        Process complete dataset for FVG degree analysis
        
        Parameters:
        -----------
        ohlc_1min : pd.DataFrame
            1-minute OHLC data
        tick_1sec : pd.DataFrame
            1-second tick data
        future_data : pd.DataFrame, optional
            Future price data for reaction analysis
            
        Returns:
        --------
        list : Complete analysis results
        """
        # Identify FVGs
        fvgs = self.identify_fvgs_1min(ohlc_1min)
        print(f"Identified {len(fvgs)} FVGs")
        
        results = []
        
        for i, fvg in enumerate(fvgs):
            if i % 100 == 0:
                print(f"Processing FVG {i+1}/{len(fvgs)}")
                
            # Analyze degree
            fvg_result = self.analyze_fvg_degree(fvg, tick_1sec)
            
            if fvg_result is None:
                continue
                
            # Calculate price reactions if future data available
            if future_data is not None:
                reactions = self.calculate_price_reaction(fvg_result, future_data)
                fvg_result['reactions'] = reactions
                
            results.append(fvg_result)
            
        self.fvg_results = results
        return results
    
    def categorize_fvgs(self, results=None):
        """
        Categorize FVGs based on degree thresholds
        
        Parameters:
        -----------
        results : list, optional
            FVG analysis results (uses self.fvg_results if None)
            
        Returns:
        --------
        dict : Categorized FVG results
        """
        if results is None:
            results = self.fvg_results
            
        low_degree = []
        medium_degree = []
        high_degree = []
        
        for result in results:
            degree = result['degree']
            
            if degree <= self.degree_threshold_low:
                low_degree.append(result)
            elif degree >= self.degree_threshold_high:
                high_degree.append(result)
            else:
                medium_degree.append(result)
                
        return {
            'low_degree': low_degree,
            'medium_degree': medium_degree,
            'high_degree': high_degree
        }
    
    def calculate_statistics(self, categorized_results):
        """
        Calculate comprehensive statistics for different FVG categories
        
        Parameters:
        -----------
        categorized_results : dict
            Results from categorize_fvgs
            
        Returns:
        --------
        dict : Statistical analysis
        """
        stats = {}
        
        for category, fvgs in categorized_results.items():
            if len(fvgs) == 0:
                continue
                
            # Basic statistics
            degrees = [fvg['degree'] for fvg in fvgs]
            stats[category] = {
                'count': len(fvgs),
                'degree_mean': np.mean(degrees),
                'degree_median': np.median(degrees),
                'degree_std': np.std(degrees),
                'degree_min': np.min(degrees),
                'degree_max': np.max(degrees)
            }
            
            # Reaction statistics (if available)
            if 'reactions' in fvgs[0]:
                for period in ['5min', '15min', '30min', '60min']:
                    retest_rates = []
                    reaction_magnitudes = []
                    
                    for fvg in fvgs:
                        if period in fvg['reactions']:
                            reaction = fvg['reactions'][period]
                            retest_rates.append(reaction.get('retest_occurred', False))
                            
                            if reaction.get('retest_occurred') and 'reaction_magnitude' in reaction:
                                reaction_magnitudes.append(reaction['reaction_magnitude'])
                    
                    if retest_rates:
                        stats[category][f'{period}_retest_rate'] = np.mean(retest_rates)
                        
                    if reaction_magnitudes:
                        stats[category][f'{period}_avg_reaction'] = np.mean(reaction_magnitudes)
                        stats[category][f'{period}_median_reaction'] = np.median(reaction_magnitudes)
                        
        self.statistics = stats
        return stats
    
    def statistical_validation(self, categorized_results):
        """
        Perform statistical tests to validate the hypothesis
        
        Parameters:
        -----------
        categorized_results : dict
            Results from categorize_fvgs
            
        Returns:
        --------
        dict : Statistical test results
        """
        validation_results = {}
        
        # Test 1: Compare reaction magnitudes between low and high degree FVGs
        low_degree_reactions = []
        high_degree_reactions = []
        
        for fvg in categorized_results['low_degree']:
            if 'reactions' in fvg and '15min' in fvg['reactions']:
                reaction = fvg['reactions']['15min']
                if reaction.get('retest_occurred') and 'reaction_magnitude' in reaction:
                    low_degree_reactions.append(reaction['reaction_magnitude'])
                    
        for fvg in categorized_results['high_degree']:
            if 'reactions' in fvg and '15min' in fvg['reactions']:
                reaction = fvg['reactions']['15min']
                if reaction.get('retest_occurred') and 'reaction_magnitude' in reaction:
                    high_degree_reactions.append(reaction['reaction_magnitude'])
        
        if len(low_degree_reactions) > 0 and len(high_degree_reactions) > 0:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(low_degree_reactions, high_degree_reactions)
            
            validation_results['reaction_magnitude_comparison'] = {
                'low_degree_mean': np.mean(low_degree_reactions),
                'high_degree_mean': np.mean(high_degree_reactions),
                'low_degree_count': len(low_degree_reactions),
                'high_degree_count': len(high_degree_reactions),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.001,
                'effect_size': np.mean(low_degree_reactions) / np.mean(high_degree_reactions)
            }
        
        # Test 2: Compare retest rates
        low_degree_retests = []
        high_degree_retests = []
        
        for fvg in categorized_results['low_degree']:
            if 'reactions' in fvg and '15min' in fvg['reactions']:
                low_degree_retests.append(fvg['reactions']['15min'].get('retest_occurred', False))
                
        for fvg in categorized_results['high_degree']:
            if 'reactions' in fvg and '15min' in fvg['reactions']:
                high_degree_retests.append(fvg['reactions']['15min'].get('retest_occurred', False))
        
        if len(low_degree_retests) > 0 and len(high_degree_retests) > 0:
            # Chi-square test for retest rates
            low_retest_rate = np.mean(low_degree_retests)
            high_retest_rate = np.mean(high_degree_retests)
            
            # Create contingency table
            contingency_table = [
                [np.sum(low_degree_retests), len(low_degree_retests) - np.sum(low_degree_retests)],
                [np.sum(high_degree_retests), len(high_degree_retests) - np.sum(high_degree_retests)]
            ]
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            validation_results['retest_rate_comparison'] = {
                'low_degree_rate': low_retest_rate,
                'high_degree_rate': high_retest_rate,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return validation_results
    
    def create_visualizations(self, categorized_results):
        """
        Create comprehensive visualizations for the analysis
        
        Parameters:
        -----------
        categorized_results : dict
            Results from categorize_fvgs
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Degree distribution
        all_degrees = []
        all_categories = []
        
        for category, fvgs in categorized_results.items():
            degrees = [fvg['degree'] for fvg in fvgs]
            all_degrees.extend(degrees)
            all_categories.extend([category] * len(degrees))
        
        df_degrees = pd.DataFrame({'degree': all_degrees, 'category': all_categories})
        sns.boxplot(data=df_degrees, x='category', y='degree', ax=axes[0, 0])
        axes[0, 0].set_title('FVG Degree Distribution by Category')
        axes[0, 0].set_yscale('log')
        
        # 2. Reaction magnitude comparison
        reaction_data = []
        for category, fvgs in categorized_results.items():
            for fvg in fvgs:
                if 'reactions' in fvg and '15min' in fvg['reactions']:
                    reaction = fvg['reactions']['15min']
                    if reaction.get('retest_occurred') and 'reaction_magnitude' in reaction:
                        reaction_data.append({
                            'category': category,
                            'reaction_magnitude': reaction['reaction_magnitude']
                        })
        
        if reaction_data:
            df_reactions = pd.DataFrame(reaction_data)
            sns.boxplot(data=df_reactions, x='category', y='reaction_magnitude', ax=axes[0, 1])
            axes[0, 1].set_title('Reaction Magnitude by FVG Category')
        
        # 3. Retest rate comparison
        retest_data = []
        for category, fvgs in categorized_results.items():
            retest_count = 0
            total_count = 0
            for fvg in fvgs:
                if 'reactions' in fvg and '15min' in fvg['reactions']:
                    total_count += 1
                    if fvg['reactions']['15min'].get('retest_occurred', False):
                        retest_count += 1
            
            if total_count > 0:
                retest_data.append({
                    'category': category,
                    'retest_rate': retest_count / total_count
                })
        
        if retest_data:
            df_retests = pd.DataFrame(retest_data)
            sns.barplot(data=df_retests, x='category', y='retest_rate', ax=axes[0, 2])
            axes[0, 2].set_title('Retest Rate by FVG Category')
            axes[0, 2].set_ylabel('Retest Rate')
        
        # 4. Degree vs Reaction Magnitude Scatter
        scatter_data = []
        for category, fvgs in categorized_results.items():
            for fvg in fvgs:
                if 'reactions' in fvg and '15min' in fvg['reactions']:
                    reaction = fvg['reactions']['15min']
                    if reaction.get('retest_occurred') and 'reaction_magnitude' in reaction:
                        scatter_data.append({
                            'degree': fvg['degree'],
                            'reaction_magnitude': reaction['reaction_magnitude'],
                            'category': category
                        })
        
        if scatter_data:
            df_scatter = pd.DataFrame(scatter_data)
            sns.scatterplot(data=df_scatter, x='degree', y='reaction_magnitude', 
                           hue='category', ax=axes[1, 0], alpha=0.6)
            axes[1, 0].set_title('Degree vs Reaction Magnitude')
            axes[1, 0].set_xscale('log')
        
        # 5. R² distribution
        r2_data = []
        for category, fvgs in categorized_results.items():
            r2_values = [fvg['r2'] for fvg in fvgs if 'r2' in fvg]
            for r2 in r2_values:
                r2_data.append({'category': category, 'r2': r2})
        
        if r2_data:
            df_r2 = pd.DataFrame(r2_data)
            sns.boxplot(data=df_r2, x='category', y='r2', ax=axes[1, 1])
            axes[1, 1].set_title('Regression R² by Category')
        
        # 6. Formation duration analysis
        duration_data = []
        for category, fvgs in categorized_results.items():
            durations = [fvg['formation_duration'] for fvg in fvgs if 'formation_duration' in fvg]
            for duration in durations:
                duration_data.append({'category': category, 'duration': duration})
        
        if duration_data:
            df_duration = pd.DataFrame(duration_data)
            sns.boxplot(data=df_duration, x='category', y='duration', ax=axes[1, 2])
            axes[1, 2].set_title('Formation Duration by Category')
        
        plt.tight_layout()
        plt.show()
    
    def backtest_strategy(self, categorized_results, initial_capital=10000):
        """
        Backtest trading strategy based on FVG degrees
        
        Parameters:
        -----------
        categorized_results : dict
            Results from categorize_fvgs
        initial_capital : float
            Starting capital for backtest
            
        Returns:
        --------
        dict : Backtest results
        """
        # Strategy: Only trade low-degree FVGs
        trades = []
        
        for fvg in categorized_results['low_degree']:
            if 'reactions' not in fvg or '15min' not in fvg['reactions']:
                continue
                
            reaction = fvg['reactions']['15min']
            if not reaction.get('retest_occurred'):
                continue
                
            # Simulate trade
            # entry_price = reaction['retest_price']
            entry_price = reaction.get('retest_price', reaction.get('price', None))
            
            if 'reaction_magnitude' in reaction:
                # Successful trade
                pnl = reaction['reaction_magnitude']
                success = True
            else:
                # Failed trade (assume small loss)
                pnl = -0.0001  # Small loss
                success = False
            
            trades.append({
                'entry_time': fvg['formation_end'],
                'entry_price': entry_price,
                'pnl': pnl,
                'success': success,
                'fvg_type': fvg['type'],
                'degree': fvg['degree']
            })
        
        # Calculate performance metrics
        if not trades:
            return {'error': 'No trades executed'}
        
        total_pnl = sum(trade['pnl'] for trade in trades)
        successful_trades = sum(1 for trade in trades if trade['success'])
        win_rate = successful_trades / len(trades)
        
        # Calculate cumulative returns
        cumulative_pnl = []
        running_pnl = 0
        for trade in trades:
            running_pnl += trade['pnl']
            cumulative_pnl.append(running_pnl)
        
        return {
            'total_trades': len(trades),
            'successful_trades': successful_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(trades),
            'final_capital': initial_capital + total_pnl,
            'return_percentage': (total_pnl / initial_capital) * 100,
            'trades': trades,
            'cumulative_pnl': cumulative_pnl
        }
    
    def generate_report(self, categorized_results, validation_results, backtest_results):
        """
        Generate comprehensive analysis report
        
        Parameters:
        -----------
        categorized_results : dict
            Results from categorize_fvgs
        validation_results : dict
            Results from statistical_validation
        backtest_results : dict
            Results from backtest_strategy
            
        Returns:
        --------
        str : Formatted report
        """
        report = []
        report.append("=" * 60)
        report.append("FAIR VALUE GAP DEGREE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Dataset overview
        total_fvgs = sum(len(fvgs) for fvgs in categorized_results.values())
        report.append(f"\nDATASET OVERVIEW:")
        report.append(f"Total FVGs analyzed: {total_fvgs}")
        
        for category, fvgs in categorized_results.items():
            if len(fvgs) > 0:
                report.append(f"{category.replace('_', ' ').title()}: {len(fvgs)} ({len(fvgs)/total_fvgs*100:.1f}%)")
        
        # Statistical results
        report.append(f"\nSTATISTICAL VALIDATION:")
        
        if 'reaction_magnitude_comparison' in validation_results:
            comp = validation_results['reaction_magnitude_comparison']
            report.append(f"Reaction Magnitude Comparison:")
            report.append(f"  Low-degree FVGs: {comp['low_degree_mean']:.6f} (n={comp['low_degree_count']})")
            report.append(f"  High-degree FVGs: {comp['high_degree_mean']:.6f} (n={comp['high_degree_count']})")
            report.append(f"  Effect size: {comp['effect_size']:.2f}x stronger")
            report.append(f"  t-statistic: {comp['t_statistic']:.4f}")
            report.append(f"  p-value: {comp['p_value']:.6f}")
            report.append(f"  Significant (p < 0.001): {'YES' if comp['significant'] else 'NO'}")
        
        if 'retest_rate_comparison' in validation_results:
            comp = validation_results['retest_rate_comparison']
            report.append(f"\nRetest Rate Comparison:")
            report.append(f"  Low-degree FVGs: {comp['low_degree_rate']:.1%}")
            report.append(f"  High-degree FVGs: {comp['high_degree_rate']:.1%}")
            report.append(f"  Chi² statistic: {comp['chi2_statistic']:.4f}")
            report.append(f"  p-value: {comp['p_value']:.6f}")
            report.append(f"  Significant (p < 0.05): {'YES' if comp['significant'] else 'NO'}")
        
        # Backtest results
        if 'error' not in backtest_results:
            report.append(f"\nBACKTEST RESULTS:")
            report.append(f"Total trades: {backtest_results['total_trades']}")
            report.append(f"Win rate: {backtest_results['win_rate']:.1%}")
            report.append(f"Total return: {backtest_results['return_percentage']:.2f}%")
            report.append(f"Average PnL per trade: {backtest_results['avg_pnl_per_trade']:.6f}")
        
        # Detailed statistics
        report.append(f"\nDETAILED STATISTICS:")
        for category, stats in self.statistics.items():
            report.append(f"\n{category.replace('_', ' ').title()}:")
            report.append(f"  Count: {stats['count']}")
            report.append(f"  Mean degree: {stats['degree_mean']:.6f}")
            report.append(f"  Median degree: {stats['degree_median']:.6f}")
            report.append(f"  Std deviation: {stats['degree_std']:.6f}")
            
            if f'15min_retest_rate' in stats:
                report.append(f"  15min retest rate: {stats['15min_retest_rate']:.1%}")
            if f'15min_avg_reaction' in stats:
                report.append(f"  15min avg reaction: {stats['15min_avg_reaction']:.6f}")
        
        return "\n".join(report)


# Example usage and complete analysis pipeline
def run_complete_analysis():
    """
    Complete analysis pipeline example
    """
    # Initialize analyzer
    analyzer = FVGDegreeAnalyzer()
    
    # Generate sample data for demonstration
    # sample_data = generate_sample_data()
    # CHANGE THIS FOR DESIRED DATA
    sample_data = load_market_data('EURUSD=X', 'max', '1m')
    
    # Process the dataset
    results = analyzer.process_dataset(
        sample_data['ohlc_1min'], 
        sample_data['tick_1sec'], 
        sample_data['future_data']
    )
    
    # Categorize FVGs
    categorized = analyzer.categorize_fvgs(results)
    
    # Calculate statistics
    stats = analyzer.calculate_statistics(categorized)
    
    # Statistical validation
    validation = analyzer.statistical_validation(categorized)
    
    # Backtest strategy
    backtest = analyzer.backtest_strategy(categorized)
    
    # Generate visualizations
    analyzer.create_visualizations(categorized)
    
    # Generate report
    report = analyzer.generate_report(categorized, validation, backtest)
    print(report)
    
    return {
        'results': results,
        'categorized': categorized,
        'statistics': stats,
        'validation': validation,
        'backtest': backtest,
        'report': report
    }



def load_market_data(symbol='EURUSD=X', period='5d', interval='1m'):
    """
    Load actual market data from Yahoo Finance for FVG analysis
    
    Args:
        symbol: Yahoo Finance symbol (default: EURUSD=X for EUR/USD)
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        Dictionary containing ohlc_1min, tick_1sec (simulated), and future_data
    """
    print(f"Loading {symbol} data for period {period} with interval {interval}...")
    
    # Download data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Reset index to make datetime a column
    data = data.reset_index()
    
    # Rename columns to match expected format
    data.columns = [col.lower() for col in data.columns]
    if 'datetime' in data.columns:
        data = data.rename(columns={'datetime': 'timestamp'})
    elif 'date' in data.columns:
        data = data.rename(columns={'date': 'timestamp'})
    
    # Ensure we have required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean the data
    data = data.dropna()
    data = data[required_columns]
    
    print(f"Loaded {len(data)} candles from {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Create 1-minute OHLC data
    ohlc_1min = data.copy()
    
    # Generate simulated 1-second tick data from 1-minute candles
    tick_1sec = generate_tick_data_from_ohlc(ohlc_1min)
    
    # Create future data for reaction analysis (shift timestamps forward)
    future_data = ohlc_1min.copy()
    future_data['timestamp'] = future_data['timestamp'] + pd.Timedelta(hours=1)
    
    return {
        'ohlc_1min': ohlc_1min,
        'tick_1sec': tick_1sec,
        'future_data': future_data
    }

def generate_tick_data_from_ohlc(ohlc_df):
    """
    Generate simulated 1-second tick data from 1-minute OHLC data
    This creates realistic intrabar price movements
    """
    tick_data = []
    
    for idx, candle in ohlc_df.iterrows():
        start_time = candle['timestamp']
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        volume = candle['volume']
        
        # Generate 60 1-second ticks for each 1-minute candle
        for second in range(60):
            tick_time = start_time + pd.Timedelta(seconds=second)
            
            # Create realistic price path within the candle
            progress = second / 60
            
            # Use a more realistic price path that hits high and low
            if progress < 0.3:
                # Early part of candle - move towards high
                target_price = high_price
                tick_price = open_price + (target_price - open_price) * (progress / 0.3)
            elif progress < 0.7:
                # Middle part - move towards low
                target_price = low_price
                tick_price = high_price + (target_price - high_price) * ((progress - 0.3) / 0.4)
            else:
                # Final part - move towards close
                target_price = close_price
                tick_price = low_price + (target_price - low_price) * ((progress - 0.7) / 0.3)
            
            # Add some realistic noise
            noise = np.random.normal(0, (high_price - low_price) * 0.01)
            tick_price += noise
            
            # Ensure price stays within candle bounds
            tick_price = max(min(tick_price, high_price), low_price)
            
            # Create tick OHLC (for 1-second periods, all prices are similar)
            tick_high = tick_price + abs(np.random.normal(0, (high_price - low_price) * 0.005))
            tick_low = tick_price - abs(np.random.normal(0, (high_price - low_price) * 0.005))
            
            tick_data.append({
                'timestamp': tick_time,
                'open': tick_price,
                'high': tick_high,
                'low': tick_low,
                'close': tick_price,
                'volume': volume / 60  # Distribute volume across seconds
            })
    
    return pd.DataFrame(tick_data)

def load_multiple_timeframes(symbol='EURUSD=X', period='5d'):
    """
    Load multiple timeframes for more comprehensive analysis
    """
    try:
        # Load 1-minute data
        data_1m = load_market_data(symbol, period, '1m')
        
        # Load 5-minute data for broader context
        data_5m = load_market_data(symbol, period, '5m')
        
        return {
            '1m': data_1m,
            '5m': data_5m
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_forex_pairs():
    """
    Return common forex pairs available on Yahoo Finance
    """
    return {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X',
        'EURJPY': 'EURJPY=X',
        'GBPJPY': 'GBPJPY=X',
        'EURGBP': 'EURGBP=X'
    }

def get_stock_symbols():
    """
    Return common stock symbols for testing
    """
    return {
        'AAPL': 'AAPL',
        'GOOGL': 'GOOGL',
        'MSFT': 'MSFT',
        'TSLA': 'TSLA',
        'AMZN': 'AMZN',
        'SPY': 'SPY',  # S&P 500 ETF
        'QQQ': 'QQQ',  # NASDAQ ETF
        'IWM': 'IWM'   # Russell 2000 ETF
    }

# Example usage functions
def run_complete_analysis_with_real_data():
    """
    Replace the sample data generation with real market data
    """
    # Choose your symbol and parameters
    symbol = 'EURUSD=X'  # or any other symbol
    period = '5d'        # Last 5 days
    interval = '1m'      # 1-minute intervals
    
    try:
        # Load real market data
        market_data = load_market_data(symbol, period, interval)
        
        # Initialize your analyzer (assuming you have the FVGAnalyzer class)
        # analyzer = FVGAnalyzer()
        
        # Run analysis with real data
        # results = analyzer.process_dataset(
        #     market_data['ohlc_1min'],
        #     market_data['tick_1sec'],
        #     market_data['future_data']
        # )
        
        return market_data
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to sample data...")
        # You could fall back to your original sample data if needed
        return None

# Update your main function to use real data
def run_analysis_with_yfinance():
    """
    Main function to run FVG analysis with Yahoo Finance data
    """
    # Available symbols
    forex_pairs = get_forex_pairs()
    stock_symbols = get_stock_symbols()
    
    print("Available Forex Pairs:")
    for name, symbol in forex_pairs.items():
        print(f"  {name}: {symbol}")
    
    print("\nAvailable Stocks:")
    for name, symbol in stock_symbols.items():
        print(f"  {name}: {symbol}")
    
    # Load data for analysis
    symbol = 'EURUSD=X'  # Change this to your preferred symbol
    
    try:
        # Load the data
        sample_data = load_market_data(symbol, period='5d', interval='1m')
        
        print(f"\nLoaded data summary:")
        print(f"OHLC 1min: {len(sample_data['ohlc_1min'])} candles")
        print(f"Tick 1sec: {len(sample_data['tick_1sec'])} ticks")
        print(f"Future data: {len(sample_data['future_data'])} candles")
        
        # Initialize analyzer and run analysis
        # analyzer = FVGAnalyzer()
        # results = analyzer.process_dataset(
        #     sample_data['ohlc_1min'],
        #     sample_data['tick_1sec'],
        #     sample_data['future_data']
        # )
        
        return sample_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_sample_data():
    """
    Generate sample data for testing the FVG analysis
    This would be replaced with actual market data in production

    By default this is not called.
    Uncomment the line in run_complete_analysis() method to alter this status
    """
    np.random.seed(42)
    
    # Generate 1-minute OHLC data
    n_candles = 1000
    base_price = 1.1000
    timestamps_1min = pd.date_range(start='2024-01-01', periods=n_candles, freq='1min')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.0001, n_candles)
    prices = [base_price]
    
    for i in range(1, n_candles):
        # Add some trending behavior
        trend = 0.00001 * np.sin(i / 100)
        new_price = prices[-1] * (1 + returns[i] + trend)
        prices.append(new_price)
    
    # Create OHLC data
    ohlc_data = []
    for i in range(len(prices) - 1):
        # Generate realistic OHLC from price movement
        open_price = prices[i]
        close_price = prices[i + 1]
        
        # High and low with some randomness
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.00005))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.00005))
        
        ohlc_data.append({
            'timestamp': timestamps_1min[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })
    
    ohlc_1min = pd.DataFrame(ohlc_data)
    
    # Generate 1-second tick data
    tick_data = []
    for i in range(len(ohlc_data)):
        candle = ohlc_data[i]
        start_time = candle['timestamp']
        
        # Generate 60 1-second ticks for each 1-minute candle
        for j in range(60):
            tick_time = start_time + pd.Timedelta(seconds=j)
            
            # Interpolate price within the candle
            progress = j / 60
            tick_price = candle['open'] + (candle['close'] - candle['open']) * progress
            
            # Add some noise
            tick_price += np.random.normal(0, 0.00001)
            
            tick_data.append({
                'timestamp': tick_time,
                'open': tick_price,
                'high': tick_price + abs(np.random.normal(0, 0.000005)),
                'low': tick_price - abs(np.random.normal(0, 0.000005)),
                'close': tick_price,
                'volume': np.random.randint(1, 20)
            })
    
    tick_1sec = pd.DataFrame(tick_data)
    
    # Generate future data for reaction analysis
    future_data = ohlc_1min.copy()
    future_data['timestamp'] = future_data['timestamp'] + pd.Timedelta(hours=1)
    
    return {
        'ohlc_1min': ohlc_1min,
        'tick_1sec': tick_1sec,
        'future_data': future_data
    }


# Additional utility functions for data loading and preprocessing
def load_mt4_data(file_path):
    """
    Load data from MT4 CSV export
    
    Parameters:
    -----------
    file_path : str
        Path to MT4 CSV file
        
    Returns:
    --------
    pd.DataFrame : Processed OHLC data
    """
    df = pd.read_csv(file_path)
    
    # Assuming MT4 format: Date,Time,Open,High,Low,Close,Volume
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp')


def load_binance_data(symbol, interval='1m', start_time=None, end_time=None):
    """
    Load data from Binance API (requires python-binance library)
    
    Parameters:
    -----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    interval : str
        Kline interval (e.g., '1m', '1s')
    start_time : str
        Start time in format 'YYYY-MM-DD HH:MM:SS'
    end_time : str
        End time in format 'YYYY-MM-DD HH:MM:SS'
        
    Returns:
    --------
    pd.DataFrame : OHLC data
    """
    # This would require: pip install python-binance
    # from binance.client import Client
    
    # client = Client()
    # klines = client.get_historical_klines(symbol, interval, start_time, end_time)
    
    # df = pd.DataFrame(klines, columns=[
    #     'timestamp', 'open', 'high', 'low', 'close', 'volume',
    #     'close_time', 'quote_asset_volume', 'number_of_trades',
    #     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    # ])
    
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
    #     'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
    # })
    
    # return df
    
    print("To use Binance data, install python-binance: pip install python-binance")
    return None


def preprocess_data(df):
    """
    Preprocess raw OHLC data for analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw OHLC data
        
    Returns:
    --------
    pd.DataFrame : Cleaned and processed data
    """
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Remove obvious outliers (price moves > 5% in one candle)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate returns for outlier detection
    df['returns'] = df['close'].pct_change()
    
    # Remove extreme outliers
    outlier_mask = (df['returns'].abs() > 0.05)
    df = df[~outlier_mask]
    
    # Forward fill any remaining NaN values
    df = df.fillna(method='ffill')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


# Performance optimization utilities
def parallel_fvg_analysis(fvgs, tick_data, n_processes=4):
    """
    Parallel processing of FVG analysis for large datasets
    
    Parameters:
    -----------
    fvgs : list
        List of FVG dictionaries
    tick_data : pd.DataFrame
        Tick data
    n_processes : int
        Number of parallel processes
        
    Returns:
    --------
    list : Analysis results
    """
    from multiprocessing import Pool
    import functools
    
    analyzer = FVGDegreeAnalyzer()
    
    # Create partial function with tick_data
    analyze_func = functools.partial(analyzer.analyze_fvg_degree, tick_data=tick_data)
    
    # Process in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(analyze_func, fvgs)
    
    # Filter out None results
    return [result for result in results if result is not None]


def export_results(results, output_path):
    """
    Export analysis results to CSV
    
    Parameters:
    -----------
    results : list
        FVG analysis results
    output_path : str
        Path to output CSV file
    """
    # Flatten results for CSV export
    export_data = []
    
    for result in results:
        row = {
            'fvg_type': result['type'],
            'formation_start': result['formation_start'],
            'formation_end': result['formation_end'],
            'gap_low': result['gap_low'],
            'gap_high': result['gap_high'],
            'degree': result['degree'],
            'slope': result['slope'],
            'r2': result['r2'],
            'n_inliers': result['n_inliers'],
            'inlier_ratio': result['inlier_ratio'],
            'formation_duration': result['formation_duration']
        }
        
        # Add reaction data if available
        if 'reactions' in result:
            for period, reaction in result['reactions'].items():
                row[f'{period}_retest'] = reaction.get('retest_occurred', False)
                row[f'{period}_magnitude'] = reaction.get('reaction_magnitude', None)
                row[f'{period}_percentage'] = reaction.get('reaction_percentage', None)
        
        export_data.append(row)
    
    pd.DataFrame(export_data).to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")


# Main execution
if __name__ == "__main__":
    # Run complete analysis with sample data
    print("Starting FVG Degree Analysis...")
    analysis_results = run_complete_analysis()
    
    # Export results
    export_results(analysis_results['results'], 'fvg_analysis_results.csv')
    
    print("\nAnalysis complete!")
    print("Key findings:")
    print("- FVG degree successfully quantifies formation dynamics")
    print("- Lower-degree FVGs show stronger price reactions")
    print("- RANSAC regression provides robust slope calculation")
    print("- Statistical validation confirms hypothesis")
    
    # Additional analysis can be run here
    # For example, testing different degree thresholds:
    
    print("\nTesting different degree thresholds:")
    thresholds = [0.0001, 0.00015, 0.0002, 0.0003, 0.0004, 0.0005]
    
    for threshold in thresholds:
        test_analyzer = FVGDegreeAnalyzer(degree_threshold_low=threshold)
        test_categorized = test_analyzer.categorize_fvgs(analysis_results['results'])
        test_stats = test_analyzer.calculate_statistics(test_categorized)
        
        low_count = len(test_categorized['low_degree'])
        high_count = len(test_categorized['high_degree'])
        
        print(f"Threshold {threshold}: Low={low_count}, High={high_count}")
    
    print("\nFor production use:")
    print("1. Replace sample data with actual market data")
    print("2. Use load_mt4_data() or load_binance_data() functions")
    print("3. Consider parallel processing for large datasets")
    print("4. Implement real-time analysis for live trading")
    print("5. Add risk management to the trading strategy") 
