import pandas as pd
import numpy as np
import os

np.random.seed(42)
NUM_ORDERS = 5000

# ─────────────────────────────────────────────
# PINCODE TIER SYSTEM
# Real Indian geographic distribution
# Tier 1: Metro cities (~10% of orders)
# Tier 2: Mid-size cities (~30% of orders)
# Tier 3: Towns and villages (~60% of orders)
# ─────────────────────────────────────────────

TIER_CONFIG = {
    1: {
        'weight': 0.10,
        'pincode_prefixes': ['110', '400', '560', '600', '700', '500'],  # Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad
        'cod_rate': 0.30,         # metro — more prepaid
        'return_rate_boost': 0.85, # lower return tendency vs national avg
        'value_boost': 1.4,        # higher order values
        'account_age_avg': 600,    # older accounts, more digital savvy
    },
    2: {
        'weight': 0.30,
        'pincode_prefixes': ['411', '302', '226', '395', '380', '160'],  # Pune, Jaipur, Lucknow, Surat, Ahmedabad, Chandigarh
        'cod_rate': 0.50,
        'return_rate_boost': 1.0,  # national average
        'value_boost': 1.0,
        'account_age_avg': 400,
    },
    3: {
        'weight': 0.60,
        'pincode_prefixes': ['243', '847', '636', '515', '754', '491',   # UP towns, Bihar, TN interior, AP, Odisha, Chhattisgarh
                             '273', '431', '585', '678', '795', '788'],  # more tier-3 / rural prefixes
        'cod_rate': 0.72,          # heavy COD dependency in rural India
        'return_rate_boost': 1.25, # higher return rate
        'value_boost': 0.75,       # lower order values
        'account_age_avg': 180,    # newer to e-commerce
    }
}

# ─────────────────────────────────────────────
# CATEGORY SYSTEM
# Real Indian e-commerce return rates
# ─────────────────────────────────────────────

CATEGORY_CONFIG = {
    'Clothing': {
        'subcategories': ['Men', 'Women', 'Kids'],
        'subcat_weights': [0.40, 0.45, 0.15],
        'base_return_rate': 0.32,   # 25-40% — size issues, color mismatch
        'avg_value': 1200,
        'value_std': 700,
        'weight': 0.45,             # largest category in Indian e-comm
    },
    'Electronics': {
        'subcategories': ['Phones', 'Laptops', 'Accessories'],
        'subcat_weights': [0.50, 0.20, 0.30],
        'base_return_rate': 0.10,   # 8-12% — defects only
        'avg_value': 8000,
        'value_std': 5000,
        'weight': 0.30,
    },
    'Jewelry': {
        'subcategories': ['Gold', 'Silver', 'Fashion'],
        'subcat_weights': [0.20, 0.20, 0.60],
        'base_return_rate': 0.04,   # under 5% — low impulse return
        'avg_value': 3000,
        'value_std': 2500,
        'weight': 0.10,
    },
    'Home & Kitchen': {
        'subcategories': ['Appliances', 'Decor', 'Utensils'],
        'subcat_weights': [0.35, 0.35, 0.30],
        'base_return_rate': 0.08,
        'avg_value': 2500,
        'value_std': 1800,
        'weight': 0.15,
    }
}

# ─────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────

def pick_tier():
    return np.random.choice([1, 2, 3], p=[
        TIER_CONFIG[1]['weight'],
        TIER_CONFIG[2]['weight'],
        TIER_CONFIG[3]['weight']
    ])

def pick_category():
    cats = list(CATEGORY_CONFIG.keys())
    weights = [CATEGORY_CONFIG[c]['weight'] for c in cats]
    return np.random.choice(cats, p=weights)

def generate_pincode(tier):
    prefix = np.random.choice(TIER_CONFIG[tier]['pincode_prefixes'])
    suffix = str(np.random.randint(100, 999))
    return int(prefix + suffix)

def generate_order(tier, cat_name):
    tier_cfg = TIER_CONFIG[tier]
    cat_cfg = CATEGORY_CONFIG[cat_name]

    # Subcategory
    subcat = np.random.choice(
        cat_cfg['subcategories'],
        p=cat_cfg['subcat_weights']
    )

    # Payment type — COD rate varies by tier
    payment_type = np.random.choice(
        ['COD', 'Prepaid'],
        p=[tier_cfg['cod_rate'], 1 - tier_cfg['cod_rate']]
    )

    # Order value — tier and category adjusted
    raw_value = np.random.normal(cat_cfg['avg_value'], cat_cfg['value_std'])
    order_value = max(99, int(raw_value * tier_cfg['value_boost']))

    # Customer return history — 0 to 10 past returns
    # Higher for clothing + COD + tier 3
    base_returns = np.random.poisson(1.5)
    if payment_type == 'COD':
        base_returns = np.random.poisson(2.5)
    if tier == 3:
        base_returns = int(base_returns * 1.3)
    past_returns = min(base_returns, 10)

    # Days since account creation
    avg_age = tier_cfg['account_age_avg']
    days_since_account_creation = max(1, int(np.random.exponential(avg_age)))

    # Device type — mobile heavy in India (75%+ nationally, more in tier 3)
    mobile_prob = 0.70 if tier == 1 else (0.78 if tier == 2 else 0.88)
    device_type = np.random.choice(
        ['mobile', 'desktop'],
        p=[mobile_prob, 1 - mobile_prob]
    )

    # Order quantity — mostly 1-2 items, occasionally more
    quantity_weights = [0.55, 0.28, 0.10, 0.05, 0.02]
    order_quantity = np.random.choice([1, 2, 3, 4, 5], p=quantity_weights)

    # Pincode
    pincode = generate_pincode(tier)

    # ─── RETURN PROBABILITY ───────────────────
    # Base from category
    return_prob = cat_cfg['base_return_rate']

    # Tier adjustment
    return_prob *= tier_cfg['return_rate_boost']

    # COD massively increases return risk — India's #1 return driver
    if payment_type == 'COD':
        return_prob *= 2.8

    # High past returns = strong signal
    if past_returns >= 4:
        return_prob *= 1.8
    elif past_returns >= 2:
        return_prob *= 1.3

    # New account = slightly higher risk
    if days_since_account_creation < 30:
        return_prob *= 1.4

    # High quantity — more likely to return partial
    if order_quantity >= 3:
        return_prob *= 1.2

    # High value clothing = size gamble
    if cat_name == 'Clothing' and order_value > 3000:
        return_prob *= 1.15

    # Cap at 95%
    return_prob = min(return_prob, 0.95)

    # Actual return outcome
    will_return = int(np.random.random() < return_prob)

    return {
        'order_value': order_value,
        'category': cat_name,
        'subcategory': subcat,
        'payment_type': payment_type,
        'past_returns': past_returns,
        'pincode': pincode,
        'pincode_tier': tier,
        'days_since_account_creation': days_since_account_creation,
        'device_type': device_type,
        'order_quantity': order_quantity,
        'return_probability': round(return_prob, 4),
        'returned': will_return
    }

def generate_dataset(n):
    records = []
    for _ in range(n):
        tier = pick_tier()
        cat = pick_category()
        records.append(generate_order(tier, cat))
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating ShipSure-AI synthetic dataset...")
    df = generate_dataset(NUM_ORDERS)

    os.makedirs('backend/data', exist_ok=True)
    df.to_csv('backend/data/orders.csv', index=False)

    print(f"\n✅ Dataset saved: backend/data/orders.csv")
    print(f"   Total orders  : {len(df)}")
    print(f"   Return rate   : {df['returned'].mean()*100:.1f}%")
    print(f"\n── By Payment Type ──")
    print(df.groupby('payment_type')['returned'].mean().apply(lambda x: f"{x*100:.1f}%"))
    print(f"\n── By Category ──")
    print(df.groupby('category')['returned'].mean().apply(lambda x: f"{x*100:.1f}%"))
    print(f"\n── By Pincode Tier ──")
    print(df.groupby('pincode_tier')['returned'].mean().apply(lambda x: f"{x*100:.1f}%"))
    print(f"\n── Sample rows ──")
    print(df.head(5).to_string(index=False))
