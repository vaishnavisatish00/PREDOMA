import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
print('\nGenerating explicit test data with clear dormancy patterns...\n')

transactions = []
transaction_id = 0
reference_date = datetime(2024, 1, 31)  # This is our "today" - far enough for 180+ day dormancy

# DORMANT MULE ACCOUNTS (clearly labeled, with post-dormancy activity, then final dormancy)
for acc_id in range(1, 26):  # Accounts 1-25: Dormant mules
    is_mule = True
    # Initial activity: Jan-May 2020
    for day in range(5, 150, 10):
        transactions.append({
            'account_id': acc_id,
            'transaction_id': transaction_id,
            'timestamp': datetime(2020, 1, 1) + timedelta(days=day),
            'amount': 5000 + acc_id * 100,
            'label': 'TF.18'
        })
        transaction_id += 1

    # DORMANCY: May 2020 to Nov 2020 (exactly 181 days to ensure > 180)
    # Gap is: May 25 to Nov 23 = 182 days
    # Post-dormancy reactivation: Nov 2020 to April 2023 (labeled period)
    # These reactivation txns are what we're looking for in labeling!
    reactivation_start = datetime(2020, 11, 23)
    reactivation_end = datetime(2023, 4, 20)  # Last post-dormancy activity
    for month in range(0, 30):  # Activity from Nov 2020 to Apr 2023
        month_date = reactivation_start + timedelta(days=month * 30)
        if month_date <= reactivation_end:
            transactions.append({
                'account_id': acc_id,
                'transaction_id': transaction_id,
                'timestamp': month_date,
                'amount': 8000 + acc_id * 100,
                'label': 'TF.18'
            })
            transaction_id += 1
    # FINAL DORMANCY: April 2023 to Oct 2023 (>180 days), so no more transactions

# DORMANT LEGITIMATE ACCOUNTS
for acc_id in range(26, 51):  # Accounts 26-50: Dormant legitimate
    is_mule = False
    # Initial activity: Jan-May 2020
    for day in range(5, 150, 10):
        transactions.append({
            'account_id': acc_id,
            'transaction_id': transaction_id,
            'timestamp': datetime(2020, 1, 1) + timedelta(days=day),
            'amount': 3000 + (acc_id * 50) % 2000,
            'label': '0'
        })
        transaction_id += 1

    # DORMANCY: May 2020 to Nov 2020 (182 days like mules)
    # Post-dormancy reactivation: Nov 2020 to April 2023 (labeled period)
    reactivation_start = datetime(2020, 11, 23)
    reactivation_end = datetime(2023, 4, 20)  # Last post-dormancy activity
    for month in range(0, 30):  # Activity from Nov 2020 to Apr 2023
        month_date = reactivation_start + timedelta(days=month * 30)
        if month_date <= reactivation_end:
            transactions.append({
                'account_id': acc_id,
                'transaction_id': transaction_id,
                'timestamp': month_date,
                'amount': 4000 + (acc_id * 50) % 2000,
                'label': '0'
            })
            transaction_id += 1
    # FINAL DORMANCY: April 2023 to Oct 2023 (>180 days), so no more transactions

# ACTIVE ACCOUNTS (continuous activity, no dormancy)
for acc_id in range(51, 300):  # Accounts 51-299: Always active
    is_mule = np.random.random() < 0.15

    # Continuous transactions from 2020-2024 to push reference_date to 2024-01
    max_days = (reference_date - datetime(2020, 1, 1)).days
    for month_offset in range(0, 49):  # 49 months to reach Jan 2024
        month_date = datetime(2020, 1, 1) + timedelta(days=month_offset * 30)
        if month_date <= reference_date:
            # 1-3 transactions per month
            for _ in range(np.random.randint(1, 4)):
                ts = month_date + timedelta(days=np.random.randint(0, 28))
                if ts <= reference_date:
                    transactions.append({
                        'account_id': acc_id,
                        'transaction_id': transaction_id,
                        'timestamp': ts,
                        'amount': 5000 + np.random.randint(-2000, 2000),
                        'label': 'TF.18' if is_mule else '0'
                    })
                    transaction_id += 1

# Add a few transactions in Jan 2024 to ensure reference_date moves forward
for acc_id in range(280, 299):
    for day in range(1, 28):
        transactions.append({
            'account_id': acc_id,
            'transaction_id': transaction_id,
            'timestamp': datetime(2024, 1, day),
            'amount': 5000,
            'label': '0'
        })
        transaction_id += 1

df = pd.DataFrame(transactions)
df = df.sort_values('timestamp').reset_index(drop=True)

os.makedirs('./data/raw/amlworld', exist_ok=True)
df.to_csv('./data/raw/amlworld/transactions.csv', index=False)

print('[OK] Explicit test data created!')
print(f'  Transactions: {len(df):,}')
print(f'  Accounts: {df["account_id"].nunique()}')
print(f'  Date range: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')
print(f'  Mule rate: {(df["label"] == "TF.18").sum() / len(df) * 100:.1f}%')
print(f'  Dormant accounts (explicit): 50 (IDs 1-50)')
print(f'    - Mules: 25 (IDs 1-25)')
print(f'    - Legitimate: 25 (IDs 26-50)')
print('\n[OK] Ready for Phase 1!\n')
