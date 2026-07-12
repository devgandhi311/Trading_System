from pathlib import Path
import pandas as pd

DEBUG_DIR = Path('reports/debug')
BACKTEST_REPORT = Path('reports/backtests/final_backtest_V2_report.xlsx')
DAILY_LOG = DEBUG_DIR / 'daily_signal_log.parquet'

FILTER_ORDER = [
    ('regime', 'Signals after regime filter'),
    ('volatility', 'Signals after volatility filter'),
    ('ml_confidence', 'Signals after ML filter'),
    ('expected_move', 'Signals after expected move filter'),
    ('liquidity', 'Signals after liquidity filter'),
    ('news', 'Signals after news filter'),
    ('risk_management', 'Signals after risk management'),
    ('position_sizing', 'Signals after position sizing'),
]

REQUESTED_LABELS = {
    'regime': 'Regime filter',
    'volatility': 'Volatility filter',
    'ml_confidence': 'ML confidence filter',
    'expected_move': 'Expected move filter',
    'liquidity': 'Liquidity filter',
    'news': 'News filter',
    'risk_management': 'Risk management filter',
    'position_sizing': 'Position sizing filter',
    'insufficient_data': 'Insufficient data',
    'evaluation_schedule': 'Evaluation schedule',
    'no_candidate_unknown': 'No candidate / unclassified',
}


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(DAILY_LOG)
    rejected_candidates = pd.read_excel(BACKTEST_REPORT, sheet_name='Rejected Candidates')
    by_index = rejected_candidates.groupby('index')['reject_reason'].apply(lambda s: tuple(str(x) for x in s.dropna()))
    candidate_rows = daily[~daily['Final decision'].isin([
        'REJECTED_INSUFFICIENT_LOOKBACK',
        'REJECTED_INSUFFICIENT_FORWARD_EXIT_DATA',
    ])].copy()
    total_candidate_signals = len(candidate_rows)
    records = []
    for _, row in candidate_rows.iterrows():
        reasons = by_index.get(row['Index'], tuple())
        first_filter = first_blocker(row, reasons)
        records.append({
            'Date': row['Date'],
            'Index': row['Index'],
            'Final decision': row['Final decision'],
            'Entry signal': row['Entry signal'],
            'First blocking filter': first_filter,
            'First blocking filter label': REQUESTED_LABELS.get(first_filter, first_filter),
            'Rejection reason': row['Rejection reason'],
            'Candidate rejection reasons': '; '.join(reasons),
            'Regime': row.get('Regime'),
            'ML probability': row.get('ML probability'),
            'Expected move': row.get('Expected move'),
            'Volatility forecast': row.get('Volatility forecast'),
            'Liquidity score': row.get('Liquidity score'),
            'Candidate EV': row.get('Candidate EV'),
            'Candidate POP': row.get('Candidate POP'),
            'Candidate max loss': row.get('Candidate max loss'),
        })
    blockers = pd.DataFrame(records)
    waterfall_rows = []
    remaining = total_candidate_signals
    waterfall_rows.append({
        'stage': 'Total candidate signals',
        'signals_remaining': remaining,
        'rejected_at_stage': 0,
        'rejection_percent_of_total': 0.0,
        'rejection_percent_of_previous_stage': 0.0,
    })
    for key, label in FILTER_ORDER:
        count = int((blockers['First blocking filter'] == key).sum()) if not blockers.empty else 0
        previous = remaining
        remaining = max(0, remaining - count)
        waterfall_rows.append({
            'stage': label,
            'signals_remaining': remaining,
            'rejected_at_stage': count,
            'rejection_percent_of_total': count / total_candidate_signals * 100.0 if total_candidate_signals else 0.0,
            'rejection_percent_of_previous_stage': count / previous * 100.0 if previous else 0.0,
        })
    executed = int((candidate_rows['Final decision'] == 'EXECUTED').sum())
    waterfall_rows.append({
        'stage': 'Final executed trades',
        'signals_remaining': executed,
        'rejected_at_stage': max(0, remaining - executed),
        'rejection_percent_of_total': max(0, remaining - executed) / total_candidate_signals * 100.0 if total_candidate_signals else 0.0,
        'rejection_percent_of_previous_stage': max(0, remaining - executed) / remaining * 100.0 if remaining else 0.0,
    })
    waterfall = pd.DataFrame(waterfall_rows)
    ranking = blockers.groupby(['First blocking filter','First blocking filter label'], dropna=False).size().reset_index(name='rejected_count') if not blockers.empty else pd.DataFrame(columns=['First blocking filter','First blocking filter label','rejected_count'])
    ranking['rejected_percent_of_candidate_signals'] = ranking['rejected_count'] / total_candidate_signals * 100.0 if total_candidate_signals else 0.0
    ranking = ranking.sort_values('rejected_count', ascending=False).reset_index(drop=True)
    summary = pd.DataFrame([{
        'total_days_in_daily_log': len(daily),
        'total_candidate_signals': total_candidate_signals,
        'final_executed_trades': executed,
        'dominant_first_blocking_filter': ranking.iloc[0]['First blocking filter label'] if not ranking.empty else 'none',
        'dominant_first_blocking_filter_percent': ranking.iloc[0]['rejected_percent_of_candidate_signals'] if not ranking.empty else 0.0,
    }])
    with pd.ExcelWriter(DEBUG_DIR / 'filter_waterfall.xlsx') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        waterfall.to_excel(writer, sheet_name='Waterfall', index=False)
        blockers.to_excel(writer, sheet_name='Daily First Blocker', index=False)
    with pd.ExcelWriter(DEBUG_DIR / 'filter_rankings.xlsx') as writer:
        ranking.to_excel(writer, sheet_name='Rankings', index=False)
        blockers.to_excel(writer, sheet_name='Daily First Blocker', index=False)
    print(summary.to_string(index=False))
    print(waterfall.to_string(index=False))
    print(ranking.to_string(index=False))


def first_blocker(row: pd.Series, reasons: tuple[str, ...]) -> str:
    decision = str(row['Final decision'])
    if decision == 'EXECUTED':
        return 'executed'
    if decision == 'REJECTED_BY_EVALUATION_SCHEDULE':
        return 'evaluation_schedule'
    if 'POSITION_SIZING' in decision:
        return 'position_sizing'
    joined = ' | '.join(reasons) or str(row.get('Rejection reason', ''))
    categories = {categorize(reason) for reason in reasons}
    if not categories:
        categories = {categorize(joined)}
    for key, _label in FILTER_ORDER:
        if key in categories:
            return key
    if 'insufficient_data' in categories:
        return 'insufficient_data'
    return 'no_candidate_unknown'


def categorize(reason: str) -> str:
    text = str(reason).lower()
    if 'news' in text:
        return 'news'
    if 'regime' in text or 'trend' in text:
        return 'regime'
    if 'forecast volatility' in text or 'volatility is not rising' in text or 'iv percentile' in text or 'iv ' in text:
        return 'volatility'
    if 'confidence' in text or 'ml' in text:
        return 'ml_confidence'
    if 'expected move' in text:
        return 'expected_move'
    if 'liquidity' in text or 'bid-ask' in text:
        return 'liquidity'
    if 'quantity < 1' in text or 'position' in text:
        return 'position_sizing'
    if 'risk' in text or 'max loss' in text or 'expected value is negative' in text or 'stress' in text or 'reward' in text:
        return 'risk_management'
    if 'insufficient data' in text or 'far-expiry option chain' in text:
        return 'insufficient_data'
    return 'no_candidate_unknown'


if __name__ == '__main__':
    main()
