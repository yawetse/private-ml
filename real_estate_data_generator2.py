# real_estate_data_generator.py
import marimo as mo

__marimo_app__ = mo.App()
app = __marimo_app__

# Cell 1: Imports and Setup
import pandas as pd
import numpy as np
from faker import Faker
import random
import uuid
import datetime
import os
import time
from openai import OpenAI # Only if generating LLM transcripts

# Initialize Faker
fake = Faker('en_US')

# Configuration
NUM_BUYERS = 5000
NUM_HOUSES_FOR_SALE = 1000
NUM_PAST_SALES = 500
NUM_TRANSCRIPTS = 10000 # For both basic and LLM versions

HOUSE_PRICE_MIN = 500_000
HOUSE_PRICE_MAX = 4_000_000

OUTPUT_DIR = "real_estate_synthetic_data"

# Central NJ Towns (Example - can be expanded)
CENTRAL_NJ_TOWNS = [
    "Princeton", "West Windsor", "Plainsboro", "Montgomery", "Hillsborough",
    "Bridgewater", "Edison", "Woodbridge", "East Brunswick", "South Brunswick",
    "Franklin Township", "Piscataway", "New Brunswick", "Hopewell", "Lawrenceville"
]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenAI Client Setup (Conditional)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = None
llm_enabled = False
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        llm_enabled = True
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}. LLM transcript generation will be skipped.")
        llm_enabled = False
else:
    print("OPENAI_API_KEY environment variable not set. LLM transcript generation will be skipped.")

@app.cell # Use the modern cell definition
def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

mo.md(f"""
## Real Estate Synthetic Data Generator

This notebook generates synthetic data for a real estate application, including:
- **Buyers:** {NUM_BUYERS} prospective buyers with financial profiles.
- **Houses for Sale:** {NUM_HOUSES_FOR_SALE} listings in Central NJ.
- **Past Sales:** {NUM_PAST_SALES} historical sales records.
- **Basic Transcripts:** {NUM_TRANSCRIPTS} simulated call transcripts (rule-based).
- **LLM Transcripts:** {NUM_TRANSCRIPTS} simulated call transcripts (using OpenAI, if API key is configured).

Data will be saved to the `{OUTPUT_DIR}` directory.
""")

# Cell 2: Helper Functions
@app.cell # Use the modern cell definition
def generate_credit_score(income_bracket):
    """Generates a credit score, loosely correlated with income bracket."""
    if income_bracket == 'Very High':
        base, std = 780, 40
    elif income_bracket == 'High':
        base, std = 740, 60
    elif income_bracket == 'Medium':
        base, std = 680, 80
    else: # Low
        base, std = 620, 100

    # Add possibility of outliers
    if random.random() < 0.05: # 5% chance of a score outside the norm
        score = np.random.randint(300, 850)
    else:
        score = int(np.random.normal(base, std))

    return np.clip(score, 300, 850) # Clip to valid range

def generate_financials(income_bracket):
    """Generates correlated income, net worth, and debt based on bracket."""
    if income_bracket == 'Very High':
        income = np.random.randint(300_000, 1_500_000)
        net_worth_multiplier = random.uniform(2, 10)
        debt_ratio = random.uniform(0.1, 0.5)
    elif income_bracket == 'High':
        income = np.random.randint(150_000, 300_000)
        net_worth_multiplier = random.uniform(1.5, 7)
        debt_ratio = random.uniform(0.2, 0.6)
    elif income_bracket == 'Medium':
        income = np.random.randint(80_000, 150_000)
        net_worth_multiplier = random.uniform(0.5, 4)
        debt_ratio = random.uniform(0.3, 0.8)
    else: # Low
        income = np.random.randint(40_000, 80_000)
        net_worth_multiplier = random.uniform(0.1, 2)
        debt_ratio = random.uniform(0.4, 1.0) # Can exceed income

    net_worth = int(income * net_worth_multiplier + np.random.normal(0, income * 0.2)) # Add noise
    net_worth = max(0, net_worth) # Ensure non-negative net worth

    total_debt = int(income * debt_ratio + np.random.normal(0, income * 0.1))
    total_debt = max(0, total_debt)

    # Outlier check: high debt regardless of income
    if random.random() < 0.03:
        total_debt = int(income * random.uniform(1.0, 2.5))

    # Outlier check: Low net worth despite high income
    if income_bracket in ['High', 'Very High'] and random.random() < 0.05:
         net_worth = int(income * random.uniform(0.1, 0.5))
         net_worth = max(0, net_worth)


    return income, net_worth, total_debt

def generate_fake_address():
    """Generates a somewhat more realistic Central NJ address."""
    street = fake.street_address()
    town = random.choice(CENTRAL_NJ_TOWNS)
    # Faker might produce zips outside NJ, this is a simplification
    zipcode = fake.zipcode_in_state(state_abbr='NJ')
    return f"{street}, {town}, NJ {zipcode}"

def generate_ssn():
    """Generates a valid-looking SSN."""
    return fake.ssn()


# Cell 3: Generate Buyers Dataset
@app.cell # Use the modern cell definition
def generate_buyers(num_buyers):
    buyers_data = []
    income_brackets = ['Low', 'Medium', 'High', 'Very High']
    bracket_probabilities = [0.15, 0.45, 0.30, 0.10] # Example distribution

    for _ in range(num_buyers):
        buyer_id = str(uuid.uuid4())
        full_name = fake.name()
        address = generate_fake_address()
        ssn = generate_ssn()
        phone_number = fake.phone_number()
        email = fake.email()

        income_bracket = np.random.choice(income_brackets, p=bracket_probabilities)
        annual_income, net_worth, total_debt = generate_financials(income_bracket)
        credit_score = generate_credit_score(income_bracket)

        # Desired price range based on financials, with variation
        base_desired = annual_income * random.uniform(3, 7) # Simple heuristic
        desired_min = int(max(HOUSE_PRICE_MIN * 0.8, base_desired * 0.7))
        desired_max = int(min(HOUSE_PRICE_MAX * 1.2, base_desired * 1.3))
        # Ensure min < max and within bounds
        desired_min = min(desired_min, HOUSE_PRICE_MAX)
        desired_max = max(desired_min + 50000, desired_max) # Ensure some range
        desired_max = min(desired_max, HOUSE_PRICE_MAX*1.1) # Allow slight overshoot
        desired_min = max(desired_min, HOUSE_PRICE_MIN*0.9)

        # Simple Pre-Approval Model (can be much more complex)
        # Based on Debt-to-Income ratio (DTI) and credit score factor
        max_monthly_payment = (annual_income / 12) * random.uniform(0.35, 0.50) # Simplified max DTI
        credit_factor = 1.0 + (credit_score - 700) / 1000 # Small adjustment for credit
        estimated_loan = max(0, max_monthly_payment * 200 * credit_factor - total_debt * 0.5) # Very rough estimate
        pre_approved_amount = int(np.clip(estimated_loan, 0, HOUSE_PRICE_MAX * 1.5)) # Allow some high approvals

        # Introduce outliers in pre-approval
        if random.random() < 0.05: # 5% chance of unusually high/low approval
            pre_approved_amount = int(pre_approved_amount * random.uniform(0.5, 2.0))
            pre_approved_amount = max(0, pre_approved_amount)


        buyers_data.append({
            "BuyerID": buyer_id,
            "FullName": full_name,
            "Address": address,
            "SSN": ssn,
            "PhoneNumber": phone_number,
            "Email": email,
            "IncomeBracket": income_bracket,
            "AnnualIncome": annual_income,
            "NetWorth": net_worth,
            "TotalDebt": total_debt,
            "CreditScore": credit_score,
            "DesiredPriceRange_Min": desired_min,
            "DesiredPriceRange_Max": desired_max,
            "PreApprovedAmount": pre_approved_amount
        })

    df = pd.DataFrame(buyers_data)
    return df

def create_buyers_df():
    df_buyers = generate_buyers(NUM_BUYERS)
    # Save to CSV
    buyers_csv_path = os.path.join(create_output_dir(), "buyers.csv")
    df_buyers.to_csv(buyers_csv_path, index=False)
    return df_buyers, buyers_csv_path

df_buyers, buyers_csv_path = create_buyers_df()

mo.md(f"### Buyers Dataset ({NUM_BUYERS} records)")
mo.md(f"Saved to: `{buyers_csv_path}`")
mo.ui.dataframe(df_buyers.head())


# Cell 4: Generate Houses for Sale Dataset
@app.cell # Use the modern cell definition
def generate_houses(num_houses):
    houses_data = []
    property_types = ['Single Family', 'Townhouse', 'Condo', 'Multi-Family']
    type_probabilities = [0.65, 0.15, 0.10, 0.10]

    for _ in range(num_houses):
        house_id = str(uuid.uuid4())
        address = generate_fake_address()
        listing_price = np.random.randint(HOUSE_PRICE_MIN, HOUSE_PRICE_MAX + 1)

        # Correlate features with price (loosely)
        price_factor = (listing_price - HOUSE_PRICE_MIN) / (HOUSE_PRICE_MAX - HOUSE_PRICE_MIN) # Normalize price 0-1

        bedrooms = np.random.randint(2, 7) + int(price_factor * 2) # More bedrooms for higher price
        bedrooms = max(2, bedrooms) # Min 2 bed

        bathrooms = round(np.random.uniform(1.5, 5.0) + price_factor * 2, 1)
        bathrooms = max(1.5, min(bathrooms, 6.0)) # Bounds 1.5 - 6
        # Ensure .0 or .5
        bathrooms = round(bathrooms * 2) / 2

        # Sqft correlated with beds, baths, price
        base_sqft = 1000
        sqft = int(base_sqft + (bedrooms * 250) + (bathrooms * 150) + (price_factor * 3000) + np.random.normal(0, 300))
        sqft = max(800, sqft) # Min sqft

        lot_size = round(np.random.uniform(0.05, 3.0) + price_factor * 2.0, 2) # Larger lots for higher price generally
        lot_size = max(0.05, lot_size) # Min lot size

        year_built = np.random.randint(1940, datetime.datetime.now().year + 1)
        # Tendency for newer homes to be more expensive (slight bias)
        if price_factor > 0.7 and random.random() < 0.6:
            year_built = np.random.randint(1990, datetime.datetime.now().year + 1)
        elif price_factor < 0.3 and random.random() < 0.6:
            year_built = np.random.randint(1940, 1985)


        property_type = np.random.choice(property_types, p=type_probabilities)
        # Adjust price slightly based on type (e.g., condos cheaper)
        if property_type == 'Condo' and listing_price > 1_000_000:
             listing_price = int(listing_price * random.uniform(0.6, 0.9))
        elif property_type == 'Single Family' and listing_price < 700_000:
             listing_price = int(listing_price * random.uniform(1.0, 1.3))
        listing_price = np.clip(listing_price, HOUSE_PRICE_MIN, HOUSE_PRICE_MAX)


        houses_data.append({
            "HouseID": house_id,
            "Address": address,
            "ListingPrice": listing_price,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "SquareFootage": sqft,
            "LotSize_Acres": lot_size,
            "YearBuilt": year_built,
            "PropertyType": property_type,
            "Status": "For Sale"
        })

    df = pd.DataFrame(houses_data)
    return df

def create_houses_df():
    df_houses = generate_houses(NUM_HOUSES_FOR_SALE)
    # Save to CSV
    houses_csv_path = os.path.join(create_output_dir(), "houses_for_sale.csv")
    df_houses.to_csv(houses_csv_path, index=False)
    return df_houses, houses_csv_path

df_houses, houses_csv_path = create_houses_df()

mo.md(f"### Houses for Sale Dataset ({NUM_HOUSES_FOR_SALE} records)")
mo.md(f"Saved to: `{houses_csv_path}`")
mo.ui.dataframe(df_houses.head())


# Cell 5: Generate Past Sales Dataset
@app.cell # Use the modern cell definition
def generate_past_sales(num_sales, buyers_df, houses_df_structure):
    """ Generates past sales, linking to buyers and simulating house details """
    sales_data = []
    sale_notes_categories = [
        "Normal", "Job Relocation", "Downsizing", "Upsizing",
        "Divorce", "Estate Sale", "Job Loss", "Bankruptcy/Foreclosure"
    ]
    # Higher probability for normal sales, lower for outliers
    sale_notes_probabilities = [0.65, 0.10, 0.05, 0.05, 0.04, 0.04, 0.035, 0.035]

    # Ensure we have enough buyers to sample from
    if len(buyers_df) < num_sales:
        print("Warning: Not enough unique buyers generated for the number of sales. Buyers will be reused.")
        buyer_indices = np.random.choice(buyers_df.index, num_sales, replace=True)
    else:
        buyer_indices = np.random.choice(buyers_df.index, num_sales, replace=False) # Unique buyers per sale if possible

    available_buyers = buyers_df.loc[buyer_indices].copy()

    for i in range(num_sales):
        sale_id = str(uuid.uuid4())

        # Generate house details similar to 'for sale' but assign as sold
        # We generate *new* house details for past sales, not using the 'for sale' list directly
        temp_house_df = generate_houses(1) # Use the same generation logic
        house_details = temp_house_df.iloc[0]

        # Select a buyer for this sale
        buyer_info = available_buyers.iloc[i]
        buyer_id = buyer_info["BuyerID"]

        # Try to make the house price somewhat plausible for the buyer
        # Adjust generated house price towards buyer's capability, but allow mismatch
        listing_price = house_details["ListingPrice"]
        target_price = (buyer_info['DesiredPriceRange_Min'] + buyer_info['DesiredPriceRange_Max']) / 2
        target_price = min(max(target_price, HOUSE_PRICE_MIN), HOUSE_PRICE_MAX) # Clamp target

        # Nudge listing price towards target, but keep variability
        adjusted_listing_price = int(listing_price * 0.5 + target_price * 0.5 + np.random.normal(0, listing_price * 0.1))
        adjusted_listing_price = np.clip(adjusted_listing_price, HOUSE_PRICE_MIN, HOUSE_PRICE_MAX)

        # Determine Sale Price (usually close to listing)
        sale_price_ratio = random.uniform(0.93, 1.07) # +/- 7% range
        sale_price = int(adjusted_listing_price * sale_price_ratio)

        # Outlier: Sale significantly different from listing
        if random.random() < 0.08: # 8% chance of bigger deviation
            sale_price = int(adjusted_listing_price * random.uniform(0.85, 1.15))
        sale_price = np.clip(sale_price, int(HOUSE_PRICE_MIN*0.8), int(HOUSE_PRICE_MAX*1.1)) # Clamp sale price


        # Sale Date
        sale_date = fake.date_between(start_date="-5y", end_date="today")

        # Sale Notes
        sale_category = np.random.choice(sale_notes_categories, p=sale_notes_probabilities)
        sale_details = f"Standard transaction."
        if sale_category == "Job Relocation":
            sale_details = f"Seller relocated for a new job opportunity in {fake.city()}."
            # Maybe bias sale price slightly lower for quicker sale
            if random.random() < 0.3: sale_price = int(sale_price * random.uniform(0.92, 0.98))
        elif sale_category == "Downsizing":
            sale_details = "Seller downsizing after retirement/children moved out."
        elif sale_category == "Upsizing":
            sale_details = "Seller buying a larger home for growing family."
            # Maybe bias sale price slightly higher if market is hot
            if random.random() < 0.2: sale_price = int(sale_price * random.uniform(1.01, 1.05))
        elif sale_category == "Divorce":
            sale_details = "Sale resulting from divorce proceedings."
            # Often pushes for faster sale, potentially lower price
            if random.random() < 0.4: sale_price = int(sale_price * random.uniform(0.90, 0.97))
        elif sale_category == "Estate Sale":
            sale_details = "Property sold as part of an estate settlement."
            # Price can vary wildly depending on condition and heir motivation
            if random.random() < 0.5: sale_price = int(sale_price * random.uniform(0.88, 1.02))
        elif sale_category == "Job Loss":
            sale_details = "Forced sale due to unexpected job loss and financial hardship."
            # Strong bias for lower price
            sale_price = int(adjusted_listing_price * random.uniform(0.85, 0.95))
        elif sale_category == "Bankruptcy/Foreclosure":
            sale_details = "Sale managed through bankruptcy court or bank foreclosure process."
             # Strong bias for lower price
            sale_price = int(adjusted_listing_price * random.uniform(0.80, 0.93))

        sale_price = max(int(HOUSE_PRICE_MIN*0.75), sale_price) # Final price floor

        sales_data.append({
            "SaleID": sale_id,
            "HouseID": house_details["HouseID"], # Link to the generated house's ID
            "BuyerID": buyer_id,
            "SellerFullName": fake.name(), # Generate fake seller
            "SellerAddress": generate_fake_address(), # Fake seller address
            "ListingPrice": adjusted_listing_price, # Use adjusted price
            "SalePrice": sale_price,
            "SaleDate": sale_date,
            "SaleCategory": sale_category,
            "SaleDetails": sale_details,
            # Include buyer financial snapshot at time of sale (copied from buyer record)
            "Buyer_AnnualIncome": buyer_info["AnnualIncome"],
            "Buyer_NetWorth": buyer_info["NetWorth"],
            "Buyer_TotalDebt": buyer_info["TotalDebt"],
            "Buyer_CreditScore": buyer_info["CreditScore"],
             # House details at time of sale
            "House_Address": house_details["Address"],
            "House_Bedrooms": house_details["Bedrooms"],
            "House_Bathrooms": house_details["Bathrooms"],
            "House_SquareFootage": house_details["SquareFootage"],
            "House_YearBuilt": house_details["YearBuilt"],
            "House_PropertyType": house_details["PropertyType"],
        })

    df = pd.DataFrame(sales_data)
    return df

def create_sales_df():
    # Pass buyers df and the structure (columns) of houses df
    df_sales = generate_past_sales(NUM_PAST_SALES, df_buyers, df_houses)
    # Save to CSV
    sales_csv_path = os.path.join(create_output_dir(), "past_sales.csv")
    df_sales.to_csv(sales_csv_path, index=False)
    return df_sales, sales_csv_path

df_sales, sales_csv_path = create_sales_df()

mo.md(f"### Past Sales Dataset ({NUM_PAST_SALES} records)")
mo.md(f"Saved to: `{sales_csv_path}`")
mo.ui.dataframe(df_sales.head())


# Cell 6: Generate Basic Call Transcripts (Rule-Based)
@app.cell # Use the modern cell definition
def generate_basic_transcripts(num_transcripts, buyers_df):
    transcripts_data = []
    broker_names = [fake.name() for _ in range(25)] # Pool of brokers
    banker_names = [fake.name() for _ in range(25)] # Pool of bankers

    # Ensure we have buyers to sample from
    if len(buyers_df) == 0:
        print("Warning: Buyers DataFrame is empty. Cannot generate transcripts.")
        return pd.DataFrame()

    buyer_ids = buyers_df['BuyerID'].tolist()

    for _ in range(num_transcripts):
        transcript_id = str(uuid.uuid4())
        call_datetime = fake.date_time_between(start_date="-2y", end_date="now")
        buyer_id = random.choice(buyer_ids)

        # Fetch buyer details (handle potential KeyError if ID somehow missing)
        try:
            buyer_info = buyers_df[buyers_df['BuyerID'] == buyer_id].iloc[0]
        except IndexError:
            continue # Skip if buyer_id not found

        broker_name = random.choice(broker_names)
        banker_name = random.choice(banker_names)

        # Extract PII
        buyer_name = buyer_info['FullName']
        buyer_address = buyer_info['Address']
        buyer_ssn = buyer_info['SSN']
        buyer_ssn_last4 = buyer_ssn.split('-')[-1]
        buyer_phone = buyer_info['PhoneNumber']
        buyer_income = buyer_info['AnnualIncome']
        buyer_desired_max = buyer_info['DesiredPriceRange_Max']


        # Simple transcript templates including PII
        templates = [
            f"MB: Hi {buyer_name}, this is {banker_name}. Just confirming your application details for the mortgage - is your SSN still {buyer_ssn}? \nBuyer: Yes, that's correct. \nMB: Great, and the address {buyer_address} is current?",
            f"Broker: Hello {buyer_name}, {broker_name} calling. Regarding houses around ${buyer_desired_max:,.0f}, I have a new listing you might like. \nBuyer: Oh really? Tell me more. \nBroker: It's on Maple St, let's connect later. Your number is {buyer_phone}, right?",
            f"MB: {banker_name} here for {buyer_name}. We need to verify income for the pre-approval. \nBuyer: Okay, what do you need? \nMB: Can you confirm your full SSN {buyer_ssn} and current residence at {buyer_address} for security?",
            f"Broker: {broker_name} checking in with {buyer_name}. Any thoughts on the properties we saw last week? \nBuyer: Still considering. The one near {random.choice(CENTRAL_NJ_TOWNS)} park was nice. \nBroker: Got it. Just confirming your details for updates: Name: {buyer_name}, Address: {buyer_address}, SSN: {buyer_ssn}.",
            f"MB: {buyer_name}, it's {banker_name}. The underwriter needs clarification on your debt-to-income ratio, given your stated income of ${buyer_income:,.0f}. \nBuyer: Okay, what specifically? \nMB: Let's review your file. Confirming SSN ending in {buyer_ssn_last4} and address {buyer_address}."
        ]

        transcript_text = random.choice(templates)

        transcripts_data.append({
            "TranscriptID": transcript_id,
            "CallDateTime": call_datetime,
            "BuyerID": buyer_id,
            "BrokerName": broker_name,
            "MortgageBankerName": banker_name,
            "TranscriptText": transcript_text
        })

    df = pd.DataFrame(transcripts_data)
    return df

def create_basic_transcripts_df():
    df_basic_transcripts = generate_basic_transcripts(NUM_TRANSCRIPTS, df_buyers)
    # Save to CSV
    basic_transcripts_csv_path = os.path.join(create_output_dir(), "basic_call_transcripts.csv")
    if not df_basic_transcripts.empty:
        df_basic_transcripts.to_csv(basic_transcripts_csv_path, index=False)
    else:
        print("Skipping save for empty basic transcripts DataFrame.")
    return df_basic_transcripts, basic_transcripts_csv_path

df_basic_transcripts, basic_transcripts_csv_path = create_basic_transcripts_df()

mo.md(f"### Basic Call Transcripts Dataset ({NUM_TRANSCRIPTS} records)")
mo.md(f"Saved to: `{basic_transcripts_csv_path}`")
if not df_basic_transcripts.empty:
    mo.ui.dataframe(df_basic_transcripts.head())
else:
    mo.md("*(Basic transcripts generation skipped as no buyer data was available)*")


# Cell 7: Generate LLM Call Transcripts (Optional - Requires OpenAI API Key)
@app.cell # Use the modern cell definition
def generate_llm_transcript_entry(client, buyer_info, broker_name, banker_name):
    """Generates a single transcript using OpenAI API."""
    if not client:
        return None # Skip if client not initialized

    buyer_name = buyer_info['FullName']
    buyer_address = buyer_info['Address']
    buyer_ssn = buyer_info['SSN']
    buyer_income = buyer_info['AnnualIncome']
    buyer_desired_max = buyer_info['DesiredPriceRange_Max']
    pre_approved = buyer_info['PreApprovedAmount']

    # Choose scenario randomly
    participants = random.choice([f"Mortgage Banker '{banker_name}'", f"Real Estate Broker '{broker_name}'"])
    scenario = random.choice([
        f"Discussing pre-approval status. Pre-approved amount is ${pre_approved:,.0f}.",
        f"Scheduling a property viewing for a house priced around ${buyer_desired_max:,.0f}.",
        f"Verifying personal information (SSN, Address) for loan application.",
        f"Following up after a property showing.",
        f"Discussing required documents for mortgage underwriting (e.g., pay stubs, bank statements).",
        f"Answering questions about current mortgage rates based on buyer's profile (Income: ${buyer_income:,.0f})."
    ])

    prompt = f"""
    Generate a brief, realistic, 3-sentence call transcript excerpt between a {participants} and prospective home buyer '{buyer_name}'.
    The conversation context is: {scenario}.

    **Crucially, the transcript MUST include the following PII for the buyer within the dialogue:**
    - Full Name: {buyer_name}
    - Full SSN: {buyer_ssn}
    - Full Address: {buyer_address}

    Keep the dialogue natural and concise (around 3 sentences total). Structure it like 'Speaker: Dialogue text'.
    """

    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo", # Or use "gpt-4" for potentially better quality but higher cost/latency
            model="gpt-4o-mini", # Or use "gpt-4" for potentially better quality but higher cost/latency
            messages=[
                {"role": "system", "content": "You are an AI assistant creating synthetic call transcript data for a real estate application. Include specific PII as requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Add some variability
            max_tokens=150 # Limit output length
        )
        transcript_text = response.choices[0].message.content.strip()
        # Basic validation: Check if required PII seems present (can be improved)
        if buyer_name in transcript_text and buyer_ssn in transcript_text and buyer_address in transcript_text:
             return transcript_text
        else:
             print(f"Warning: LLM response might be missing required PII. Prompt: {prompt} \nResponse: {transcript_text}")
             # Fallback or retry logic could be added here
             return f"LLM_FAILED_PII_CHECK: {transcript_text}" # Mark failed check

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Exponential backoff could be implemented here for rate limits
        time.sleep(2) # Simple delay
        return None


def generate_llm_transcripts(num_transcripts, buyers_df, client):
    """Generates multiple transcripts using the LLM helper function."""
    if not client or not llm_enabled:
        print("LLM client not available. Skipping LLM transcript generation.")
        return pd.DataFrame()

    transcripts_data = []
    broker_names = [fake.name() for _ in range(25)]
    banker_names = [fake.name() for _ in range(25)]

    if len(buyers_df) == 0:
        print("Warning: Buyers DataFrame is empty. Cannot generate LLM transcripts.")
        return pd.DataFrame()

    buyer_ids = buyers_df['BuyerID'].tolist()
    print(f"Starting LLM transcript generation for {num_transcripts} entries...")

    # Consider generating in batches if NUM_TRANSCRIPTS is very large
    # For simplicity, generating one by one with progress indication
    generated_count = 0
    while generated_count < num_transcripts:
        transcript_id = str(uuid.uuid4())
        call_datetime = fake.date_time_between(start_date="-2y", end_date="now")
        buyer_id = random.choice(buyer_ids)

        try:
            buyer_info = buyers_df[buyers_df['BuyerID'] == buyer_id].iloc[0]
        except IndexError:
            continue # Skip if buyer_id somehow missing

        broker_name = random.choice(broker_names)
        banker_name = random.choice(banker_names)

        transcript_text = generate_llm_transcript_entry(client, buyer_info, broker_name, banker_name)

        if transcript_text:
            transcripts_data.append({
                "TranscriptID": transcript_id,
                "CallDateTime": call_datetime,
                "BuyerID": buyer_id,
                "BrokerName": broker_name, # Assigned randomly, LLM could potentially name them
                "MortgageBankerName": banker_name, # Assigned randomly
                "TranscriptText": transcript_text
            })
            generated_count += 1
            if generated_count % 100 == 0: # Print progress
                 print(f"Generated {generated_count}/{num_transcripts} LLM transcripts...")
        else:
            # Handle cases where API call failed or returned None
            print(f"Skipping transcript entry due to generation error for buyer {buyer_id}.")
            # Optional: Implement a maximum retry limit if needed

        # Add a small delay to avoid hitting rate limits aggressively
        time.sleep(0.1) # Adjust as needed based on your OpenAI plan tier

    print(f"Finished LLM transcript generation. Total generated: {generated_count}")
    df = pd.DataFrame(transcripts_data)
    return df


def create_llm_transcripts_df():
    if not llm_enabled or client is None:
         return pd.DataFrame(), "LLM Transcripts Skipped (OpenAI API key not configured or client init failed)"

    # **WARNING:** Setting NUM_TRANSCRIPTS high (e.g., 10000) will take significant time and API cost.
    # Reduce NUM_TRANSCRIPTS_LLM for testing.
    NUM_TRANSCRIPTS_LLM = 100 # <<< START WITH A SMALL NUMBER FOR TESTING (e.g., 100)
    # NUM_TRANSCRIPTS_LLM = NUM_TRANSCRIPTS # Uncomment for full generation

    print(f"Attempting to generate {NUM_TRANSCRIPTS_LLM} LLM transcripts...")
    df_llm_transcripts = generate_llm_transcripts(NUM_TRANSCRIPTS_LLM, df_buyers, client)

    llm_transcripts_csv_path = os.path.join(create_output_dir(), "llm_call_transcripts.csv")
    if not df_llm_transcripts.empty:
        df_llm_transcripts.to_csv(llm_transcripts_csv_path, index=False)
        msg = f"Saved to: `{llm_transcripts_csv_path}`"
    else:
        msg = "LLM transcripts DataFrame is empty. No file saved."
        llm_transcripts_csv_path = None # Ensure path is None if not saved

    return df_llm_transcripts, msg, llm_transcripts_csv_path # Return path for display

# Check if LLM generation should run
if llm_enabled and client:
    df_llm_transcripts, llm_save_msg, llm_transcripts_csv_path = create_llm_transcripts_df()
    mo.md(f"### LLM Call Transcripts Dataset ({len(df_llm_transcripts)} records generated)")
    mo.md(llm_save_msg)
    if not df_llm_transcripts.empty and llm_transcripts_csv_path:
        mo.ui.dataframe(df_llm_transcripts.head())
    elif not llm_enabled:
        mo.md("*(LLM transcript generation skipped as OpenAI API key was not provided or client failed)*")
    else:
        mo.md("*(LLM transcript generation resulted in an empty dataset)*")
else:
    mo.md("### LLM Call Transcripts Dataset")
    mo.md("*(LLM transcript generation skipped as OpenAI API key was not provided or client failed)*")
    # Define placeholders if skipped
    df_llm_transcripts = pd.DataFrame()
    llm_transcripts_csv_path = None

# Cell 8: Summary
mo.md(f"""
## Data Generation Complete

The following datasets have been generated and saved in the `{OUTPUT_DIR}` directory:

1.  **Buyers:** `{os.path.basename(buyers_csv_path)}` ({len(df_buyers)} records)
2.  **Houses for Sale:** `{os.path.basename(houses_csv_path)}` ({len(df_houses)} records)
3.  **Past Sales:** `{os.path.basename(sales_csv_path)}` ({len(df_sales)} records)
4.  **Basic Transcripts:** `{os.path.basename(basic_transcripts_csv_path)}` ({len(df_basic_transcripts)} records)
5.  **LLM Transcripts:** {f'`{os.path.basename(llm_transcripts_csv_path)}` ({len(df_llm_transcripts)} records)' if llm_transcripts_csv_path else '*Skipped*'}
""")

# This is required for Marimo to know it's an app
if __name__ == "__main__":
    pass