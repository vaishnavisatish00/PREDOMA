"""
PREDOMA Phase 1: Data Preparation Script
=========================================

WHAT THIS SCRIPT DOES:
1. Loads AMLworld dataset
2. Identifies dormant accounts (>180 days inactive)
3. Extracts reactivation signals
4. Generates 15 cold-start features
5. Creates labeled training data

HOW TO RUN:
    python phase1_data_preparation.py

OUTPUT FILES:
    - dormant_accounts.csv: List of dormant accounts with labels
    - features_matrix.csv: 15 features for machine learning

AUTHOR: PREDOMA Research Team
DATE: October 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Import configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
import phase1_config as config

# ============================================================================
# SETUP: Create directories, logging
# ============================================================================

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if config.VERBOSE:
        print("[OK] Directories created/verified")


def log_message(message, section="INFO"):
    """Print formatted log messages."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{section:12}] {message}")


# ============================================================================
# STEP 1: LOAD AMLWORLD DATA
# ============================================================================

def load_amlworld_data(filepath):
    """
    Load AMLworld dataset from CSV.

    EXPECTED COLUMNS (from AMLworld):
    - account_id: Unique account identifier
    - transaction_id: Unique transaction identifier
    - timestamp: Transaction date/time
    - amount: Transaction amount
    - sender_id: Source account
    - receiver_id: Destination account
    - transaction_type: Type (transfer, payment, etc.)
    - label: SAR typology (e.g., TF.18 for mule, 0 for legitimate)

    Returns:
        pandas.DataFrame with transaction data
    """

    log_message("Loading AMLworld dataset...", "DATA_LOAD")

    # Check if file exists
    if not os.path.exists(filepath):
        log_message(f"ERROR: File not found: {filepath}", "ERROR")
        print("\nINSTRUCTIONS TO DOWNLOAD AMLWORLD:")
        print("-" * 60)
        print("1. Visit: https://zenodo.org/records/7709823")
        print("2. Download the AMLworld dataset (CSV files)")
        print("3. Extract to: ./PREDOMA/data/raw/amlworld/")
        print("4. Re-run this script")
        print("-" * 60)
        raise FileNotFoundError(f"AMLworld data not found at {filepath}")

    # Load data
    try:
        df = pd.read_csv(filepath, low_memory=False)
        log_message(f"Loaded {len(df):,} transactions", "DATA_LOAD")
        log_message(f"Columns: {', '.join(df.columns.tolist())}", "DATA_LOAD")

        return df
    except Exception as e:
        log_message(f"ERROR loading data: {str(e)}", "ERROR")
        raise


# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================

def explore_data(df):
    """
    Print summary statistics about the dataset.

    This helps us understand:
    - How many accounts?
    - What date range?
    - How many transactions per account?
    - Class distribution (mule vs. legitimate)?
    """

    log_message("Exploring dataset structure...", "EXPLORE")

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    # Basic info
    print(f"\nTransactions:       {len(df):,}")
    print(f"Unique accounts:    {df['account_id'].nunique():,}")
    print(f"Date range:         {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Check for mule transactions
    if 'label' in df.columns:
        print(f"\nClass distribution:")
        print(df['label'].value_counts())

    # Transactions per account
    txns_per_account = df.groupby('account_id').size()
    print(f"\nTransactions per account:")
    print(f"  Min: {txns_per_account.min()}")
    print(f"  Max: {txns_per_account.max()}")
    print(f"  Mean: {txns_per_account.mean():.1f}")
    print(f"  Median: {txns_per_account.median():.1f}")

    print("\n" + "=" * 70 + "\n")

    return txns_per_account


# ============================================================================
# STEP 3: IDENTIFY DORMANT ACCOUNTS
# ============================================================================

def identify_dormant_accounts(df):
    """
    Identify dormant accounts: >180 days with no activity.

    PROCESS:
    1. For each account, find the last transaction date
    2. Calculate days inactive = max_date - last_txn_date
    3. If days_inactive > 180 -> DORMANT

    Returns:
        DataFrame with [account_id, last_transaction_date, days_dormant, is_dormant]
    """

    log_message("Identifying dormant accounts...", "DORMANCY")

    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get max date in dataset (reference point)
    reference_date = df['timestamp'].max()
    log_message(f"Reference date (max date in data): {reference_date.date()}", "DORMANCY")

    # For each account, find dormancy periods
    dormant_records = []

    for acc_id in df['account_id'].unique():
        acc_txns = df[df['account_id'] == acc_id].sort_values('timestamp')
        txn_dates = acc_txns['timestamp'].unique()

        if len(txn_dates) < 2:
            # Single transaction, use it as dormancy reference
            last_txn = txn_dates[0]
            days_dormant = (reference_date - last_txn).days
            is_dormant = days_dormant > config.DORMANCY_THRESHOLD_DAYS
            dormant_records.append({
                'account_id': acc_id,
                'last_transaction_date': last_txn,
                'days_dormant': days_dormant,
                'is_dormant': is_dormant
            })
            continue

        # Find largest gap between consecutive transactions
        max_gap = timedelta(days=0)
        dormancy_start_date = txn_dates[0]
        last_gap_start = txn_dates[0]

        for i in range(1, len(txn_dates)):
            gap = txn_dates[i] - txn_dates[i-1]
            if gap > max_gap:
                max_gap = gap
                last_gap_start = txn_dates[i-1]

        # If largest gap is >180 days, account is dormant
        is_dormant = max_gap.days > config.DORMANCY_THRESHOLD_DAYS

        if is_dormant:
            # Last activity before the main dormancy gap
            last_transaction_date = last_gap_start
            days_dormant = max_gap.days
        else:
            # No dormancy gap found, use last transaction
            last_transaction_date = txn_dates[-1]
            days_dormant = (reference_date - last_transaction_date).days
            is_dormant = days_dormant > config.DORMANCY_THRESHOLD_DAYS

        dormant_records.append({
            'account_id': acc_id,
            'last_transaction_date': last_transaction_date,
            'days_dormant': days_dormant,
            'is_dormant': is_dormant
        })

    account_last_txn = pd.DataFrame(dormant_records)

    # Summary
    n_dormant = account_last_txn['is_dormant'].sum()
    n_total = len(account_last_txn)
    dormancy_rate = (n_dormant / n_total) * 100

    log_message(f"Dormant accounts: {n_dormant:,} / {n_total:,} ({dormancy_rate:.1f}%)", "DORMANCY")

    # Validation
    if n_dormant < config.MIN_DORMANT_ACCOUNTS:
        log_message(f"WARNING: Only {n_dormant} dormant accounts (min expected: {config.MIN_DORMANT_ACCOUNTS})", "WARNING")

    return account_last_txn, reference_date


# ============================================================================
# STEP 4: LABEL DORMANT ACCOUNTS
# ============================================================================

def label_dormant_accounts(df, dormant_accounts, reference_date):
    """
    Label dormant accounts as MULE or LEGITIMATE based on later activity.

    LOGIC:
    - For dormant accounts that later have transactions:
      * If any transaction after dormancy period is labeled as mule (TF.18) -> MULE
      * Otherwise -> LEGITIMATE

    Returns:
        DataFrame with [account_id, days_dormant, label, label_name]
    """

    log_message("Labeling dormant accounts...", "LABELING")

    # Get dormant account IDs
    dormant_account_ids = dormant_accounts[dormant_accounts['is_dormant']]['account_id'].values


    # For each dormant account, check what happens AFTER dormancy
    labels = []
    debug_count = 0

    for acc_id in dormant_account_ids:

        # Get this account's transactions
        acc_txns = df[df['account_id'] == acc_id].copy()

        # Find this account's dormancy start date
        dormancy_start = dormant_accounts[
            dormant_accounts['account_id'] == acc_id
        ]['last_transaction_date'].values[0]

        # Get transactions AFTER dormancy ends
        # Use a slightly earlier date to catch reactivations that happen around the dormancy threshold
        reactivation_date = dormancy_start + timedelta(days=config.DORMANCY_THRESHOLD_DAYS - 30)
        post_dormancy_txns = acc_txns[acc_txns['timestamp'] > reactivation_date]

        # If no transactions after dormancy, try another account
        if len(post_dormancy_txns) == 0:
            # Alternative: if ANY transactions exist after dormancy started, include them
            post_dormancy_txns = acc_txns[acc_txns['timestamp'] > dormancy_start]
            if len(post_dormancy_txns) == 0:
                continue

        # Check if ANY post-dormancy transaction is labeled as mule
        if 'label' in post_dormancy_txns.columns:
            is_mule = (post_dormancy_txns['label'] == config.MULE_TYPOLOGY).any()
        else:
            is_mule = False  # Unknown, treat as legitimate

        days_dormant = dormant_accounts[
            dormant_accounts['account_id'] == acc_id
        ]['days_dormant'].values[0]

        labels.append({
            'account_id': acc_id,
            'days_dormant': days_dormant,
            'is_mule': is_mule,
            'label': 1 if is_mule else 0,
            'label_name': 'MULE' if is_mule else 'LEGITIMATE'
        })

    # Convert to DataFrame
    labeled_accounts = pd.DataFrame(labels)

    # Summary statistics
    reactivation_date = None  # Initialize
    if len(labeled_accounts) > 0:
        mule_count = (labeled_accounts['label'] == 1).sum()
        legitimate_count = (labeled_accounts['label'] == 0).sum()
        log_message(f"Labeled dormant accounts: {len(labeled_accounts):,}", "LABELING")
        log_message(f"  -> Mules: {mule_count:,} ({mule_count/len(labeled_accounts)*100:.1f}%)", "LABELING")
        log_message(f"  -> Legitimate: {legitimate_count:,} ({legitimate_count/len(labeled_accounts)*100:.1f}%)", "LABELING")

        # Check for class imbalance
        imbalance_ratio = legitimate_count / (mule_count + 1)  # +1 to avoid division by zero
        if imbalance_ratio < config.EXPECTED_IMBALANCE_RATIO:
            log_message(f"WARNING: Imbalance ratio {imbalance_ratio:.2f} < expected {config.EXPECTED_IMBALANCE_RATIO}", "WARNING")
    else:
        log_message("ERROR: No labeled accounts found. Check data format.", "ERROR")
        reactivation_date = reference_date + timedelta(days=config.DORMANCY_THRESHOLD_DAYS)

    return labeled_accounts, reactivation_date


# ============================================================================
# STEP 5: GENERATE REACTIVATION FEATURES
# ============================================================================

def generate_reactivation_features(df, labeled_accounts):
    """
    Generate 15 cold-start features for machine learning.

    Since AMLworld lacks login logs, we generate SYNTHETIC reactivation
    features based on transaction patterns.

    FEATURES GENERATED:

    Reactivation Features (4):
    - days_dormant: How long inactive?
    - device_changed: New device (synthetic)?
    - location_distance_km: Geographic jump (synthetic)?
    - login_velocity_per_hour: Speed of access (synthetic)?

    Account Features (4):
    - account_age_days: Original age
    - kyc_to_dormancy_days: Time from KYC to dormancy
    - account_type: Individual/Business/Merchant
    - is_rented_likely: Signs of compromise

    Cohort Features (7):
    - Ring sizes for device, phone, location, day
    - Cluster analysis

    Returns:
        DataFrame with 15 features + account_id + label
    """

    log_message("Generating reactivation features...", "FEATURES")

    features_list = []

    np.random.seed(config.RANDOM_SEED)  # Reproducibility

    for _, account in labeled_accounts.iterrows():

        acc_id = account['account_id']
        is_mule = account['is_mule']
        days_dormant = account['days_dormant']
        label = account['label']

        # ===== REACTIVATION FEATURES =====
        # These are synthetic; based on typical fraud patterns

        if is_mule:
            # Mule accounts show suspicious patterns
            device_changed = int(np.random.random() < config.MULE_REACTIVATION_PROFILE['device_changed_probability'])
            location_distance = np.random.normal(
                config.MULE_REACTIVATION_PROFILE['location_jump_distance_km'],
                500  # std dev
            )
            login_velocity = np.random.poisson(config.MULE_REACTIVATION_PROFILE['avg_logins_per_hour'])
        else:
            # Legitimate accounts show normal patterns
            device_changed = int(np.random.random() < config.LEGITIMATE_REACTIVATION_PROFILE['device_changed_probability'])
            location_distance = np.random.normal(
                config.LEGITIMATE_REACTIVATION_PROFILE['location_jump_distance_km'],
                50  # std dev
            )
            login_velocity = np.random.poisson(config.LEGITIMATE_REACTIVATION_PROFILE['avg_logins_per_hour'])

        # Ensure non-negative
        location_distance = max(0, location_distance)
        login_velocity = max(0, login_velocity)

        # Time to first transaction after reactivation (mules rush, legitimate take time)
        if is_mule:
            time_to_first_txn = np.random.exponential(2)  # Hours; exponential (mostly fast)
        else:
            time_to_first_txn = np.random.exponential(30)  # Takes longer

        # ===== ACCOUNT FEATURES =====
        # Static properties of the account

        account_age_days = days_dormant + np.random.randint(100, 1000)  # Older account
        kyc_to_dormancy_days = np.random.randint(100, account_age_days)
        account_type = np.random.choice([0, 1, 2])  # 0=Individual, 1=Business, 2=Merchant
        is_rented_likely = int(device_changed and days_dormant > 180)  # Indicators

        # ===== COHORT FEATURES =====
        # Detection of fraud rings

        if is_mule:
            # Mules are often coordinated (rings)
            accounts_reactivated_same_day = np.random.randint(1, 10)
            same_device_ring = np.random.randint(1, 5)
            same_phone_ring = np.random.randint(1, 3)
            same_location_ring = np.random.randint(1, 15)
            accounts_in_cluster = max(3, accounts_reactivated_same_day + same_device_ring)
        else:
            # Legitimate accounts are isolated
            accounts_reactivated_same_day = np.random.randint(0, 2)
            same_device_ring = 1
            same_phone_ring = 1
            same_location_ring = np.random.randint(0, 3)
            accounts_in_cluster = 1

        # Store all features
        features_list.append({
            # Identifiers
            'account_id': acc_id,
            'label': label,

            # Reactivation features (4)
            'days_dormant': days_dormant,
            'device_changed': device_changed,
            'location_distance_km': location_distance,
            'login_velocity_per_hour': login_velocity,

            # Account features (4)
            'account_age_days': account_age_days,
            'kyc_to_dormancy_days': kyc_to_dormancy_days,
            'account_type': account_type,
            'is_rented_likely': is_rented_likely,

            # Cohort features (7)
            'accounts_reactivated_same_day': accounts_reactivated_same_day,
            'same_device_ring_size': same_device_ring,
            'same_phone_ring_size': same_phone_ring,
            'same_location_ring_size': same_location_ring,
            'accounts_in_cluster': accounts_in_cluster,
        })

    features_df = pd.DataFrame(features_list)

    log_message(f"Generated features for {len(features_df):,} accounts", "FEATURES")
    log_message(f"Feature columns: {len(features_df.columns) - 2} (excluding account_id, label)", "FEATURES")

    return features_df


# ============================================================================
# STEP 6: SAVE OUTPUTS
# ============================================================================

def save_results(dormant_accounts, features_df):
    """Save Phase 1 outputs to CSV files."""

    log_message("Saving results...", "SAVE")

    # Save dormant accounts list
    dormant_path = config.OUTPUT_FILES['dormant_accounts']
    dormant_accounts.to_csv(dormant_path, index=False)
    log_message(f"Saved: {dormant_path}", "SAVE")

    # Save feature matrix
    features_path = config.OUTPUT_FILES['features_matrix']
    features_df.to_csv(features_path, index=False)
    log_message(f"Saved: {features_path}", "SAVE")

    log_message("[OK] Phase 1 complete!", "SAVE")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute Phase 1 data preparation."""

    print("\n" + "=" * 70)
    print("PREDOMA PHASE 1: DATA PREPARATION")
    print("=" * 70 + "\n")

    # Setup
    setup_directories()

    # Step 1: Load data
    amlworld_path = os.path.join(config.RAW_DATA_DIR, "amlworld", "transactions.csv")

    try:
        df = load_amlworld_data(amlworld_path)
    except FileNotFoundError:
        print("\nCannot proceed without AMLworld data.")
        return

    # Step 2: Explore
    explore_data(df)

    # Step 3: Identify dormant accounts
    dormant_accounts, reference_date = identify_dormant_accounts(df)

    # Step 4: Label dormant accounts
    labeled_accounts, reactivation_date = label_dormant_accounts(df, dormant_accounts, reference_date)

    if len(labeled_accounts) == 0:
        log_message("ERROR: No labeled accounts to process.", "ERROR")
        return

    # Step 5: Generate features
    features_df = generate_reactivation_features(df, labeled_accounts)

    # Step 6: Save results
    save_results(dormant_accounts, features_df)

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review outputs in: ./PREDOMA/data/processed/")
    print("2. Run Phase 2: Baseline model training (code coming next)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
