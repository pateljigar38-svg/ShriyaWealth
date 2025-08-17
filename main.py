from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# === FILE-BASED DATAFETCHER ===

class DataFetcher:
    def __init__(self, json_path='data/amfi_all_data.json'):
        try:
            with open(json_path, 'r') as f:
                self.all_data = json.load(f)
        except Exception as e:
            print(f"Error loading AMFI data file: {e}")
            self.all_data = []
        self.scheme_codes = {}
        for fund in self.all_data:
            self.scheme_codes[fund['scheme_code']] = fund['scheme_name']
    def get_scheme_codes_by_category(self):
        return {'All': self.scheme_codes}
    def fetch_historical_nav(self, scheme_code, start_date, end_date):
        for fund in self.all_data:
            if fund['scheme_code'] == scheme_code:
                df = pd.DataFrame(fund['nav_history'])
                if df.empty:
                    return pd.DataFrame(columns=['date', 'nav'])
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
                return df.loc[mask, ['date', 'nav']].reset_index(drop=True)
        return pd.DataFrame(columns=['date', 'nav'])


# === FUND ANALYSIS CLASSES ===

class MutualFundSelector:
    def __init__(self):
        self.categories = {
            'Large Cap': [],
            'Mid Cap': [],
            'Small Cap': [],
            'Flexicap': [],
            'Thematic': [],
            'Hybrid': [],
            'International': []
        }
        self.performance_requirements = {
            'bear_market_min_xirr': 13.0,
            'bull_market_min_xirr': 18.0,
            'backtesting_accuracy': 95.0
        }
        self.allocation_rules = {
            'max_funds_per_portfolio': 7,
            'max_single_fund_allocation': 30.0,
            'min_funds_conservative': 3,
            'max_funds_aggressive': 7
        }
    def calculate_xirr(self, cashflows, dates, guess=0.1):
        try:
            from scipy.optimize import fsolve
            def npv(rate, cashflows, dates):
                npv_value = 0
                base_date = dates[0]
                for cf, date in zip(cashflows, dates):
                    days = (date - base_date).days
                    npv_value += cf / ((1 + rate) ** (days / 365.0))
                return npv_value
            xirr = fsolve(lambda r: npv(r, cashflows, dates), [guess])
            return xirr * 100
        except:
            return 0.0
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_return = returns.mean() - risk_free_rate/252
        volatility = returns.std()
        return (excess_return / volatility) * np.sqrt(252)
    def calculate_max_drawdown(self, nav_series):
        cumulative = (1 + nav_series.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min() * 100
    def calculate_volatility(self, returns):
        return returns.std() * np.sqrt(252) * 100

class BacktestEngine:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.backtest_results = {}
        self.accuracy_threshold = 95.0
    def identify_market_phases(self, market_data):
        market_data['rolling_return_6m'] = market_data['nav'].pct_change(126)
        market_data['rolling_return_3m'] = market_data['nav'].pct_change(63)
        conditions_bear = (market_data['rolling_return_6m'] < -0.20) | (market_data['rolling_return_3m'] < -0.15)
        conditions_bull = (market_data['rolling_return_6m'] > 0.15) | (market_data['rolling_return_3m'] > 0.12)
        market_data['market_phase'] = 'sideways'
        market_data.loc[conditions_bear, 'market_phase'] = 'bear'
        market_data.loc[conditions_bull, 'market_phase'] = 'bull'
        return market_data
    def backtest_fund_selection(self, fund_data, start_date, end_date):
        results = {}
        for category, schemes in fund_data.items():
            category_results = {}
            for scheme_code, scheme_name in schemes.items():
                nav_data = self.data_fetcher.fetch_historical_nav(
                    scheme_code,
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date)
                )
                if len(nav_data) < 252:
                    continue
                returns = nav_data['nav'].pct_change().dropna()
                nav_data = self.identify_market_phases(nav_data)
                bear_periods = nav_data[nav_data['market_phase'] == 'bear']
                bull_periods = nav_data[nav_data['market_phase'] == 'bull']
                bear_xirr = self._calculate_sip_xirr(bear_periods) if len(bear_periods) > 60 else 0
                bull_xirr = self._calculate_sip_xirr(bull_periods) if len(bull_periods) > 60 else 0
                total_return = ((nav_data['nav'].iloc[-1] / nav_data['nav'].iloc[0]) - 1) * 100
                annual_return = total_return / (len(nav_data) / 252)
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                max_drawdown = self._calculate_max_drawdown(nav_data['nav'])
                volatility = returns.std() * np.sqrt(252) * 100
                returns_data = self._calculate_period_returns(nav_data)
                category_results[scheme_code] = {
                    'scheme_name': scheme_name,
                    'total_return': annual_return,
                    'bear_xirr': bear_xirr,
                    'bull_xirr': bull_xirr,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'nav_data': nav_data,
                    **returns_data,
                    'meets_criteria': self._check_fund_criteria(bear_xirr, bull_xirr, annual_return)
                }
            results[category] = category_results
        self.backtest_results = results
        return results
    def _calculate_period_returns(self, nav_data):
        nav_series = nav_data['nav']
        current_nav = nav_series.iloc[-1]
        periods = {'3M': 63, '6M': 126, '1Y': 252, '3Y': 756, '5Y': 1260}
        returns = {}
        for period_name, days in periods.items():
            if len(nav_series) > days:
                past_nav = nav_series.iloc[-days-1]
                period_return = ((current_nav / past_nav) ** (252/days) - 1) * 100
                returns[period_name] = period_return
            else:
                returns[period_name] = 0
        return returns
    def _calculate_sip_xirr(self, nav_data, sip_amount=10000):
        if len(nav_data) < 12:
            return 0
        sip_dates = nav_data.iloc[::21]['date'].tolist()
        sip_navs = nav_data.iloc[::21]['nav'].tolist()
        cashflows = [-sip_amount] * (len(sip_dates) - 1)
        dates = sip_dates[:-1]
        units_accumulated = sum(sip_amount / nav for nav in sip_navs[:-1])
        final_value = units_accumulated * sip_navs[-1]
        cashflows.append(final_value)
        dates.append(sip_dates[-1])
        try:
            from scipy.optimize import fsolve
            def npv(rate, cashflows, dates):
                npv_value = 0
                base_date = dates[0]
                for cf, date in zip(cashflows, dates):
                    days = (date - base_date).days
                    npv_value += cf / ((1 + rate) ** (days / 365.0))
                return npv_value
            xirr = fsolve(lambda r: npv(r, cashflows, dates), [0.1])
            return xirr * 100
        except:
            return 0
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_return = returns.mean() - risk_free_rate/252
        return (excess_return / returns.std()) * np.sqrt(252)
    def _calculate_max_drawdown(self, nav_series):
        cumulative = nav_series / nav_series.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min() * 100
    def _check_fund_criteria(self, bear_xirr, bull_xirr, annual_return):
        return {
            'bear_market_ok': bear_xirr >= 13.0,
            'bull_market_ok': bull_xirr >= 18.0,
            'overall_return_ok': annual_return >= 15.0
        }

class PortfolioConstructor:
    def __init__(self, allocation_rules):
        self.allocation_rules = allocation_rules
        self.portfolio_templates = {
            'Conservative': {
                'risk_level': 'Low',
                'max_funds': 3,
                'allocation': {
                    'Large Cap': 50,
                    'Hybrid': 30,
                    'Debt': 20
                }
            },
            'Moderate': {
                'risk_level': 'Medium',
                'max_funds': 5,
                'allocation': {
                    'Large Cap': 35,
                    'Mid Cap': 25,
                    'Flexicap': 20,
                    'Hybrid': 15,
                    'International': 5
                }
            },
            'Aggressive': {
                'risk_level': 'High',
                'max_funds': 7,
                'allocation': {
                    'Large Cap': 20,
                    'Mid Cap': 25,
                    'Small Cap': 20,
                    'Flexicap': 15,
                    'Thematic': 10,
                    'International': 5,
                    'Hybrid': 5
                }
            }
        }
    def construct_portfolio(self, risk_profile, top_funds_by_category, investment_amount=100000):
        if risk_profile not in self.portfolio_templates:
            raise ValueError(f"Risk profile {risk_profile} not supported")
        template = self.portfolio_templates[risk_profile]
        target_allocation = template['allocation']
        max_funds = template['max_funds']
        selected_funds = {}
        allocations = {}
        fund_amounts = {}
        funds_selected = 0
        total_expected_return = 0
        for category, target_pct in target_allocation.items():
            if category in top_funds_by_category and funds_selected < max_funds:
                category_funds = top_funds_by_category[category]
                if len(category_funds) > 0:
                    eligible_funds = [
                        (code, data) for code, data in category_funds.items()
                        if data.get('meets_criteria', {}).get('overall_return_ok', False)
                    ]
                    if eligible_funds:
                        best_fund = max(eligible_funds,
                                      key=lambda x: x[1].get('total_return', 0))
                        fund_code, fund_data = best_fund
                    else:
                        best_fund = max(category_funds.items(),
                                      key=lambda x: x[1].get('total_return', 0))
                        fund_code, fund_data = best_fund
                    selected_funds[fund_code] = {
                        'name': fund_data['scheme_name'],
                        'category': category,
                        'performance': fund_data
                    }
                    actual_allocation = min(target_pct, self.allocation_rules['max_single_fund_allocation'])
                    allocations[fund_code] = actual_allocation
                    fund_amounts[fund_code] = (actual_allocation / 100) * investment_amount
                    total_expected_return += (actual_allocation / 100) * fund_data.get('total_return', 0)
                    funds_selected += 1
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            for fund_code in allocations:
                allocations[fund_code] = (allocations[fund_code] / total_allocated) * 100
                fund_amounts[fund_code] = (allocations[fund_code] / 100) * investment_amount
        portfolio_metrics = self._calculate_portfolio_metrics(selected_funds, allocations)
        validation = self.validate_allocation_rules(selected_funds, allocations)
        portfolio = {
            'risk_profile': risk_profile,
            'investment_amount': investment_amount,
            'selected_funds': selected_funds,
            'allocations': allocations,
            'fund_amounts': fund_amounts,
            'portfolio_metrics': portfolio_metrics,
            'validation': validation,
            'total_funds': len(selected_funds),
            'expected_annual_return': total_expected_return
        }
        return portfolio
    def _calculate_portfolio_metrics(self, selected_funds, allocations):
        total_return = 0
        total_volatility = 0
        weighted_sharpe = 0
        worst_drawdown = 0
        for fund_code, fund_info in selected_funds.items():
            weight = allocations[fund_code] / 100
            performance = fund_info['performance']
            total_return += weight * performance.get('total_return', 0)
            total_volatility += weight * performance.get('volatility', 0)
            weighted_sharpe += weight * performance.get('sharpe_ratio', 0)
            worst_drawdown = min(worst_drawdown, performance.get('max_drawdown', 0))
        return {
            'expected_return': round(total_return, 2),
            'portfolio_volatility': round(total_volatility, 2),
            'portfolio_sharpe': round(weighted_sharpe, 2),
            'worst_drawdown': round(worst_drawdown, 2)
        }
    def validate_allocation_rules(self, selected_funds, allocations):
        validation_results = {
            'valid': True,
            'violations': [],
            'recommendations': []
        }
        if len(selected_funds) > self.allocation_rules['max_funds_per_portfolio']:
            validation_results['valid'] = False
            validation_results['violations'].append(
                f"Portfolio has {len(selected_funds)} funds, exceeds max of {self.allocation_rules['max_funds_per_portfolio']}"
            )
        max_allocation = max(allocations.values()) if allocations else 0
        if max_allocation > self.allocation_rules['max_single_fund_allocation']:
            validation_results['violations'].append(
                f"Single fund allocation of {max_allocation:.1f}% exceeds max of {self.allocation_rules['max_single_fund_allocation']}%"
            )
        if len(selected_funds) < self.allocation_rules['min_funds_conservative']:
            validation_results['recommendations'].append(
                f"Consider adding more funds for better diversification (minimum {self.allocation_rules['min_funds_conservative']} recommended)"
            )
        return validation_results

class SIPCalculator:
    def __init__(self):
        self.calculation_history = []
        self.inflation_rate = 3.0
    def calculate_sip_projections(self, sip_amount, portfolio_xirr, tenure_years, step_up_rate=0):
        monthly_rate = (1 + portfolio_xirr/100) ** (1/12) - 1
        total_months = int(tenure_years * 12)
        future_value = 0
        total_invested = 0
        monthly_sip = sip_amount
        for month in range(total_months):
            if month > 0 and month % 12 == 0 and step_up_rate > 0:
                monthly_sip = monthly_sip * (1 + step_up_rate/100)
            remaining_months = total_months - month
            installment_fv = monthly_sip * ((1 + monthly_rate) ** remaining_months)
            future_value += installment_fv
            total_invested += monthly_sip
        total_gains = future_value - total_invested
        scenarios = {
            'Pessimistic (XIRR -3%)': self._calculate_scenario_value(
                sip_amount, max(portfolio_xirr-3, 8), tenure_years, step_up_rate
            ),
            'Conservative (XIRR -1.5%)': self._calculate_scenario_value(
                sip_amount, max(portfolio_xirr-1.5, 10), tenure_years, step_up_rate
            ),
            'Expected (Current XIRR)': future_value,
            'Optimistic (XIRR +2%)': self._calculate_scenario_value(
                sip_amount, portfolio_xirr+2, tenure_years, step_up_rate
            ),
            'Best Case (XIRR +4%)': self._calculate_scenario_value(
                sip_amount, portfolio_xirr+4, tenure_years, step_up_rate
            )
        }
        inflation_adjusted_value = future_value / ((1 + self.inflation_rate/100) ** tenure_years)
        return {
            'sip_amount': sip_amount,
            'tenure_years': tenure_years,
            'step_up_rate': step_up_rate,
            'portfolio_xirr': portfolio_xirr,
            'total_invested': total_invested,
            'expected_value': future_value,
            'total_gains': total_gains,
            'roi_percentage': (total_gains / total_invested) * 100,
            'scenarios': scenarios,
            'inflation_adjusted_value': inflation_adjusted_value,
            'real_gains': inflation_adjusted_value - total_invested
        }
    def _calculate_scenario_value(self, sip_amount, xirr, tenure_years, step_up_rate=0):
        monthly_rate = (1 + xirr/100) ** (1/12) - 1
        total_months = int(tenure_years * 12)
        future_value = 0
        monthly_sip = sip_amount
        for month in range(total_months):
            if month > 0 and month % 12 == 0 and step_up_rate > 0:
                monthly_sip = monthly_sip * (1 + step_up_rate/100)
            remaining_months = total_months - month
            installment_fv = monthly_sip * ((1 + monthly_rate) ** remaining_months)
            future_value += installment_fv
        return future_value
    def calculate_lumpsum_projections(self, lumpsum_amount, portfolio_xirr, tenure_years):
        future_value = lumpsum_amount * ((1 + portfolio_xirr/100) ** tenure_years)
        total_gains = future_value - lumpsum_amount
        scenarios = {
            'Pessimistic (XIRR -3%)': lumpsum_amount * ((1 + max(portfolio_xirr-3, 8)/100) ** tenure_years),
            'Conservative (XIRR -1.5%)': lumpsum_amount * ((1 + max(portfolio_xirr-1.5, 10)/100) ** tenure_years),
            'Expected (Current XIRR)': future_value,
            'Optimistic (XIRR +2%)': lumpsum_amount * ((1 + (portfolio_xirr+2)/100) ** tenure_years),
            'Best Case (XIRR +4%)': lumpsum_amount * ((1 + (portfolio_xirr+4)/100) ** tenure_years)
        }
        inflation_adjusted_value = future_value / ((1 + self.inflation_rate/100) ** tenure_years)
        return {
            'lumpsum_amount': lumpsum_amount,
            'tenure_years': tenure_years,
            'portfolio_xirr': portfolio_xirr,
            'expected_value': future_value,
            'total_gains': total_gains,
            'roi_percentage': (total_gains / lumpsum_amount) * 100,
            'scenarios': scenarios,
            'inflation_adjusted_value': inflation_adjusted_value,
            'real_gains': inflation_adjusted_value - lumpsum_amount
        }

# === FASTAPI APP SETUP ===

app = FastAPI(title="Mutual Fund Analysis API")

class RecommendationRequest(BaseModel):
    amount: int
    tenor: int

origins = [
    "https://pateljigar38-svg.github.io",
    "https://pateljigar38-svg.github.io/ShriyaWealth",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fetcher = DataFetcher(json_path='data/amfi_all_data.json')
backtester = BacktestEngine(fetcher)
constructor = PortfolioConstructor({
    'max_funds_per_portfolio': 7,
    'max_single_fund_allocation': 30.0,
    'min_funds_conservative': 3,
    'max_funds_aggressive': 7
})
sip_calc = SIPCalculator()

@app.get("/")
async def root():
    return {"message": "Mutual Fund Analysis API is running"}

@app.post("/recommend")
async def recommend(req: RecommendationRequest):
    try:
        scheme_codes = fetcher.get_scheme_codes_by_category()
        start_date = "2018-01-01"
        end_date = datetime.today().strftime("%Y-%m-%d")
        results = backtester.backtest_fund_selection(scheme_codes, start_date, end_date)
        top_funds_by_category = {}
        for category, funds in results.items():
            filtered = {k: v for k, v in funds.items() if v.get('meets_criteria', {}).get('overall_return_ok', False)}
            if filtered:
                top = dict(sorted(filtered.items(), key=lambda x: x[1]['total_return'], reverse=True)[:3])
            else:
                top = dict(sorted(funds.items(), key=lambda x: x[1]['total_return'], reverse=True)[:3])
            top_funds_by_category[category] = top
        portfolio = constructor.construct_portfolio('Moderate', top_funds_by_category, investment_amount=req.amount)
        recommendations = []
        for fund_code, fund_info in portfolio.get('selected_funds', {}).items():
            perf = fund_info.get('performance', {})
            recommendations.append({
                "name": fund_info.get('name', ''),
                "type": fund_info.get('category', ''),
                "xirr": round(perf.get('total_return', 0), 2),
                "bear": round(perf.get('bear_xirr', 0), 2),
                "bull": round(perf.get('bull_xirr', 0), 2),
                "explanation": "Recommended based on risk profile and past performance"
            })
        if not recommendations:
            all_funds = []
            for cat in results.values():
                for code, info in cat.items():
                    all_funds.append({
                        "name": info.get('scheme_name', ''),
                        "type": info.get('category', 'N/A'),
                        "xirr": round(info.get('total_return', 0), 2),
                        "bear": round(info.get('bear_xirr', 0), 2),
                        "bull": round(info.get('bull_xirr', 0), 2),
                        "explanation": "Best available by past returns"
                    })
            recommendations = sorted(all_funds, key=lambda x: x['xirr'], reverse=True)[:3]
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
