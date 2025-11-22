#!/usr/bin/env python3
"""
Script para generar datasets sintéticos basados en los datasets de Kaggle
Para el proyecto de Machine Learning
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Crear directorio de datos
data_dir = Path('/app/data')
data_dir.mkdir(exist_ok=True)

print("Generando datasets sintéticos...")

# ============================================
# 1. TELCO CUSTOMER CHURN DATASET
# ============================================
print("\n1. Generando Telco Customer Churn dataset...")

n_customers = 7043  # Tamaño similar al dataset original

# Generar datos categóricos
gender = np.random.choice(['Male', 'Female'], n_customers)
senior_citizen = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48])
dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])
phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.90, 0.10])
multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.42, 0.48, 0.10])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])
online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.28, 0.50, 0.22])
online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.34, 0.44, 0.22])
device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.34, 0.44, 0.22])
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.29, 0.49, 0.22])
streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.38, 0.40, 0.22])
streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.39, 0.39, 0.22])
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])
payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    n_customers,
    p=[0.33, 0.23, 0.22, 0.22]
)

# Generar datos numéricos
tenure = np.random.randint(0, 73, n_customers)
monthly_charges = np.random.uniform(18.25, 118.75, n_customers)
total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_customers)
total_charges = np.maximum(total_charges, 0)  # No negativos

# Generar target Churn con lógica realista
# Mayor probabilidad de churn si: tenure bajo, monthly charges alto, contract month-to-month
churn_prob = np.zeros(n_customers)
for i in range(n_customers):
    prob = 0.2  # Base probability
    
    # Factores que aumentan churn
    if tenure[i] < 12:
        prob += 0.3
    if monthly_charges[i] > 80:
        prob += 0.2
    if contract[i] == 'Month-to-month':
        prob += 0.25
    if senior_citizen[i] == 1:
        prob += 0.1
    if internet_service[i] == 'Fiber optic':
        prob += 0.15
    
    # Factores que disminuyen churn
    if contract[i] == 'Two year':
        prob -= 0.3
    if online_security[i] == 'Yes':
        prob -= 0.1
    if tech_support[i] == 'Yes':
        prob -= 0.1
    
    churn_prob[i] = np.clip(prob, 0.05, 0.8)

churn = np.array(['Yes' if np.random.random() < p else 'No' for p in churn_prob])

# Crear DataFrame
telco_df = pd.DataFrame({
    'customerID': [f'ID{i:04d}' for i in range(n_customers)],
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges.round(2),
    'TotalCharges': total_charges.round(2),
    'Churn': churn
})

telco_df.to_csv(data_dir / 'telco_churn.csv', index=False)
print(f"✓ Telco Churn dataset generado: {len(telco_df)} registros")
print(f"  - Churn Yes: {(telco_df['Churn'] == 'Yes').sum()} ({(telco_df['Churn'] == 'Yes').sum()/len(telco_df)*100:.1f}%)")
print(f"  - Churn No: {(telco_df['Churn'] == 'No').sum()} ({(telco_df['Churn'] == 'No').sum()/len(telco_df)*100:.1f}%)")

# ============================================
# 2. CREDIT CARD DATASET FOR CLUSTERING
# ============================================
print("\n2. Generando Credit Card dataset...")

n_cards = 8950  # Tamaño similar al dataset original

# Generar features con correlaciones realistas
balance = np.random.lognormal(7.5, 1.5, n_cards)
credit_limit = balance * np.random.uniform(1.5, 4, n_cards)

# Purchases
purchases = np.random.lognormal(6, 2, n_cards)
oneoff_purchases = purchases * np.random.uniform(0.2, 0.7, n_cards)
installments_purchases = purchases - oneoff_purchases + np.random.normal(0, 200, n_cards)
installments_purchases = np.maximum(installments_purchases, 0)

# Cash advance
cash_advance = np.random.lognormal(5, 2.5, n_cards)

# Frequencies
purchases_frequency = np.random.beta(2, 5, n_cards)
oneoff_purchases_frequency = purchases_frequency * np.random.uniform(0.2, 0.8, n_cards)
purchases_installments_frequency = purchases_frequency * np.random.uniform(0.3, 0.9, n_cards)
cash_advance_frequency = np.random.beta(1.5, 8, n_cards)

# Transactions
cash_advance_trx = np.random.poisson(3, n_cards)
purchases_trx = np.random.poisson(15, n_cards)

# Payments
payments = balance * np.random.uniform(0.5, 1.5, n_cards) + purchases * np.random.uniform(0.3, 0.8, n_cards)
minimum_payments = payments * np.random.uniform(0.05, 0.15, n_cards)
prc_full_payment = np.random.beta(2, 5, n_cards)

# Balance frequency (always 1 for active accounts)
balance_frequency = np.ones(n_cards)

# Crear DataFrame
credit_df = pd.DataFrame({
    'CUST_ID': [f'C{i:04d}' for i in range(n_cards)],
    'BALANCE': balance.round(2),
    'BALANCE_FREQUENCY': balance_frequency,
    'PURCHASES': purchases.round(2),
    'ONEOFF_PURCHASES': oneoff_purchases.round(2),
    'INSTALLMENTS_PURCHASES': installments_purchases.round(2),
    'CASH_ADVANCE': cash_advance.round(2),
    'PURCHASES_FREQUENCY': purchases_frequency.round(6),
    'ONEOFF_PURCHASES_FREQUENCY': oneoff_purchases_frequency.round(6),
    'PURCHASES_INSTALLMENTS_FREQUENCY': purchases_installments_frequency.round(6),
    'CASH_ADVANCE_FREQUENCY': cash_advance_frequency.round(6),
    'CASH_ADVANCE_TRX': cash_advance_trx,
    'PURCHASES_TRX': purchases_trx,
    'CREDIT_LIMIT': credit_limit.round(2),
    'PAYMENTS': payments.round(2),
    'MINIMUM_PAYMENTS': minimum_payments.round(2),
    'PRC_FULL_PAYMENT': prc_full_payment.round(6)
})

credit_df.to_csv(data_dir / 'credit_card.csv', index=False)
print(f"✓ Credit Card dataset generado: {len(credit_df)} registros")
print(f"  - Balance promedio: ${credit_df['BALANCE'].mean():.2f}")
print(f"  - Compras promedio: ${credit_df['PURCHASES'].mean():.2f}")

print("\n✓ Todos los datasets generados exitosamente en /app/data/")
