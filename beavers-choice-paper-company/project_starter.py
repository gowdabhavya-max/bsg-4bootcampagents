import pandas as pd
import numpy as np
import os
import re
import json
import time
import difflib
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# ---------------------------------------------------------------------------
# Environment and model
# ---------------------------------------------------------------------------
# Look for a .env in the current directory tree and next to this script, so the
# key is found no matter where python is launched from.
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
dotenv.load_dotenv()


def build_model() -> OpenAIServerModel:
    """Create the shared LLM client. The key is read from the environment, never hardcoded."""
    api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set UDACITY_OPENAI_API_KEY (or OPENAI_API_KEY) in the "
            "environment or in a .env file next to this script."
        )
    return OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_base="https://openai.vocareum.com/v1",
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Business rules (tune these in one place)
# ---------------------------------------------------------------------------
# Historical quotes price items at the catalog unit_price (for example "A4 paper at $0.05 each")
# and then apply bulk discounts of about 10%, so customers are quoted at the catalog price.
BASE_MARKUP = 1.0
# Bulk discount tiers per order line: (minimum units, discount). History shows about 10% for 500+ units.
DISCOUNT_TIERS = [(5000, 0.15), (1000, 0.12), (500, 0.10), (200, 0.05)]
# Upper bound on the total discount (tier + any extra discretionary discount)
MAX_TOTAL_DISCOUNT = 0.20
# The extra discretionary discount an agent may add on top of the automatic tier
MAX_EXTRA_DISCOUNT_PERCENT = 5.0
# The starter seeds opening stock at cost = unit_price, which would leave no margin once a discount
# is given. Reorders are therefore assumed to cost this fraction of the catalog price (a wholesale
# assumption, tune it if the real supplier terms are known).
SUPPLIER_COST_RATIO = 0.70
# Never let a purchase push cash below this amount
CASH_RESERVE = 1000.0
# Used when a catalog item is stocked for the first time
DEFAULT_MIN_STOCK = 100
# Replenishment orders bring projected stock up to min_stock_level * this factor
RESTOCK_TARGET_MULTIPLIER = 3
FAR_FUTURE = "2099-12-31"

_CATALOG = {item["item_name"]: item for item in paper_supplies}

# The evaluation loop tells the system what "today" is for each request. Tools
# read the date from here instead of trusting a date typed by the LLM.
_ctx = {"date": None}
_DATE_RE = re.compile(r"Date of request:\s*(\d{4}-\d{2}-\d{2})")


def _today() -> str:
    return _ctx["date"] or datetime.now().strftime("%Y-%m-%d")


def _asof(date_str: str = None) -> str:
    """End-of-day timestamp so that every transaction dated on `date_str` is included."""
    return f"{date_str or _today()}T23:59:59"


def _j(obj) -> str:
    """Tools return JSON text: easy for the LLM to read and free of numpy types."""
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=None)


def _stock(item_name: str, as_of: str) -> int:
    return int(get_stock_level(item_name, as_of)["current_stock"].iloc[0])


def _normalize(text_in: str) -> str:
    """Lower-case and simplify customer wording so that variants of the same words compare equal."""
    lowered = text_in.lower()
    lowered = re.sub(r"(\d+)(?:\s*(?:\"|inch(?:es)?))?\s*x\s*(\d+)(?:\s*(?:\"|inch(?:es)?))?", r"\1x\2 ", lowered)  # 24" x 36" -> 24x36
    lowered = re.sub(r"colou?r(?:ful|ed)", "colored", lowered)                                             # colorful -> colored
    lowered = lowered.replace("card stock", "cardstock")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _tokens(text_in: str) -> set:
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in _normalize(text_in).split()}


# Customer wording that a fuzzy comparison cannot resolve ('washi tape', 'napkins', 'printer paper').
# Each entry is (words that must all appear, catalog item). The first entry that matches wins, so
# specific items come before general ones ('glossy' before 'a4', 'wrapping' before 'decorative').
_ALIASES = [
    ({"washi"}, "Decorative adhesive tape (washi tape)"),
    ({"adhesive"}, "Decorative adhesive tape (washi tape)"),
    ({"masking"}, "Decorative adhesive tape (washi tape)"),
    ({"disposable", "cup"}, "Disposable cups"),
    ({"cup"}, "Paper cups"),
    ({"plate"}, "Paper plates"),
    ({"napkin"}, "Paper napkins"),
    ({"streamer"}, "Party streamers"),
    ({"party", "bag"}, "Paper party bags"),
    ({"name", "tag"}, "Name tags with lanyards"),
    ({"lanyard"}, "Name tags with lanyards"),
    ({"table", "cover"}, "Table covers"),
    ({"sticky"}, "Sticky notes"),
    ({"notepad"}, "Notepads"),
    ({"folder"}, "Presentation folders"),
    ({"envelope"}, "Envelopes"),
    ({"flyer"}, "Flyers"),
    ({"large", "poster"}, "Large poster paper (24x36 inches)"),
    ({"24x36"}, "Large poster paper (24x36 inches)"),
    ({"poster"}, "Poster paper"),
    ({"roll", "banner"}, "Rolls of banner paper (36-inch width)"),
    ({"banner"}, "Banner paper"),
    ({"cover", "stock"}, "100 lb cover stock"),
    ({"text", "paper"}, "80 lb text paper"),
    ({"glossy"}, "Glossy paper"),
    ({"photo"}, "Photo paper"),
    ({"matte"}, "Matte paper"),
    ({"recycled"}, "Recycled paper"),
    ({"eco"}, "Eco-friendly paper"),
    ({"kraft"}, "Kraft paper"),
    ({"butcher"}, "Butcher paper"),
    ({"crepe"}, "Crepe paper"),
    ({"glitter"}, "Glitter paper"),
    ({"wrapping"}, "Wrapping paper"),
    ({"construction"}, "Construction paper"),
    ({"patterned"}, "Patterned paper"),
    ({"decorative", "paper"}, "Decorative paper"),
    ({"invitation"}, "Invitation cards"),
    ({"letterhead"}, "Letterhead paper"),
    ({"legal"}, "Legal-size paper"),
    ({"cardstock"}, "Cardstock"),
    ({"heavyweight"}, "Heavyweight paper"),
    ({"bright"}, "Bright-colored paper"),
    ({"colored"}, "Colored paper"),
    ({"letter"}, "Letter-sized paper"),
    ({"a4"}, "A4 paper"),
    ({"printer"}, "Standard copy paper"),
    ({"printing"}, "Standard copy paper"),
    ({"copier"}, "Standard copy paper"),
    ({"copy"}, "Standard copy paper"),
    ({"plain"}, "Standard copy paper"),
    ({"standard"}, "Standard copy paper"),
    ({"white", "paper"}, "Standard copy paper"),
]


def _alias_match(query: str):
    """Catalog item for common customer wording, or None."""
    words = _tokens(query)
    for required_words, item in _ALIASES:
        if required_words <= words:
            return item
    return None


def _match_catalog(query: str, top: int = 3) -> List[tuple]:
    """Rank catalog items against free text: synonym hit first, then token overlap and string similarity."""
    q_norm, q_tok = _normalize(query), _tokens(query)
    scored = []
    for name in _CATALOG:
        n_norm, n_tok = _normalize(name), _tokens(name)
        if q_norm == n_norm:
            score = 1.0
        else:
            jaccard = len(q_tok & n_tok) / max(len(q_tok | n_tok), 1)
            ratio = difflib.SequenceMatcher(None, q_norm, n_norm).ratio()
            score = 0.5 * jaccard + 0.5 * ratio
        scored.append((round(score, 3), name))
    ranked = sorted(scored, reverse=True)
    alias = _alias_match(query)
    if alias and ranked[0][0] < 1.0:  # an exact catalog name always wins over a synonym
        ranked = [(0.99, alias)] + [entry for entry in ranked if entry[1] != alias]
    return ranked[:top]


def _resolve(query: str):
    """Return (catalog_name, None) for a confident match, otherwise (None, customer-safe message)."""
    matches = _match_catalog(query)
    best_score, best = matches[0]
    second = matches[1][0] if len(matches) > 1 else 0.0
    if best_score >= 0.99 or (best_score >= 0.7 and best_score - second >= 0.05):
        return best, None
    options = ", ".join(name for _, name in matches)
    return None, f"We do not sell '{query}'. Our closest products are: {options}."


def _parse_lines(order_lines: str):
    """Parse 'Item name:quantity;Item name:quantity' into ([(catalog_name, qty)], [problems])."""
    merged, problems = {}, []
    for part in re.split(r"[;\n]+", order_lines or ""):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            problems.append(f"Could not read order line '{part}'. Use 'Item name:quantity'.")
            continue
        name, qty_text = part.rsplit(":", 1)
        try:
            qty = int(float(qty_text.strip().replace(",", "")))
        except ValueError:
            problems.append(f"Quantity '{qty_text.strip()}' for '{name.strip()}' is not a number.")
            continue
        if qty <= 0:
            problems.append(f"Quantity for '{name.strip()}' must be positive.")
            continue
        item, error = _resolve(name.strip())
        if error:
            problems.append(error)
            continue
        merged[item] = merged.get(item, 0) + qty
    return list(merged.items()), problems


def _parse_deadline(deadline: str):
    """Return (YYYY-MM-DD or None, problem or None). 'none' means the customer gave no date."""
    if deadline is None or deadline.strip().lower() in {"", "none", "n/a", "na", "null", "no deadline"}:
        return None, None
    parsed = pd.to_datetime(deadline, errors="coerce")
    if pd.isna(parsed):
        return None, f"Could not understand the required delivery date '{deadline}'."
    return parsed.strftime("%Y-%m-%d"), None


def _price_lines(lines: List[tuple], extra_discount_percent: float) -> List[Dict]:
    """Apply markup and bulk discounts to each (item, qty) line."""
    # Clamped here, in the one pricing function, so a quote and the sale that follows it always agree
    extra = min(max(float(extra_discount_percent or 0.0), 0.0), MAX_EXTRA_DISCOUNT_PERCENT) / 100.0
    priced = []
    for item, qty in lines:
        list_price = _CATALOG[item]["unit_price"] * BASE_MARKUP
        tier = next((d for minimum, d in DISCOUNT_TIERS if qty >= minimum), 0.0)
        discount = min(tier + extra, MAX_TOTAL_DISCOUNT)
        unit_price = list_price * (1 - discount)
        priced.append({
            "item": item,
            "quantity": qty,
            "list_unit_price": round(list_price, 4),
            "discount_percent": round(discount * 100, 1),
            "unit_price": round(unit_price, 4),
            "line_total": round(unit_price * qty, 2),
        })
    return priced


def _ensure_inventory_row(item_name: str) -> None:
    """Register a catalog item in the inventory table the first time we stock it."""
    with db_engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM inventory WHERE item_name = :n"), {"n": item_name}
        ).fetchone()
        if not exists:
            conn.execute(
                text(
                    "INSERT INTO inventory (item_name, category, unit_price, current_stock, min_stock_level) "
                    "VALUES (:n, :c, :p, 0, :m)"
                ),
                {
                    "n": item_name,
                    "c": _CATALOG[item_name]["category"],
                    "p": _CATALOG[item_name]["unit_price"],
                    "m": DEFAULT_MIN_STOCK,
                },
            )


def _min_stock_level(item_name: str) -> int:
    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT min_stock_level FROM inventory WHERE item_name = :n"), {"n": item_name}
        ).fetchone()
    return int(row[0]) if row else DEFAULT_MIN_STOCK


def _incoming_arrival_dates(item_name: str, after_date: str) -> List[str]:
    """Dates (after `after_date`) on which supplier deliveries for this item are already booked."""
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT substr(transaction_date, 1, 10) FROM transactions "
                "WHERE item_name = :n AND transaction_type = 'stock_orders' "
                "AND substr(transaction_date, 1, 10) > :d ORDER BY 1"
            ),
            {"n": item_name, "d": after_date},
        ).fetchall()
    return [row[0] for row in rows]


def _ready_date(item_name: str, quantity: int, today: str, extra_units: int = 0, extra_arrival: str = None):
    """
    Earliest date on which `quantity` units are on hand, counting deliveries already
    booked and, optionally, a hypothetical extra delivery. Returns None if never.
    """
    days = [today] + _incoming_arrival_dates(item_name, today)
    if extra_arrival:
        days.append(extra_arrival)
    for day in sorted(set(days)):
        supply = _stock(item_name, _asof(day))
        if extra_arrival and day >= extra_arrival:
            supply += extra_units
        if supply >= quantity:
            return day
    return None


def _supplier_cost(item_name: str, quantity: int) -> float:
    """What a supplier order costs the company."""
    return round(quantity * _CATALOG[item_name]["unit_price"] * SUPPLIER_COST_RATIO, 2)


def _place_stock_order(item_name: str, quantity: int):
    """Order from the supplier. Stock is booked on the arrival date, so it cannot be sold earlier."""
    cost = _supplier_cost(item_name, quantity)
    arrival = get_supplier_delivery_date(_today(), quantity)
    _ensure_inventory_row(item_name)
    transaction_id = create_transaction(item_name, "stock_orders", int(quantity), cost, arrival)
    return transaction_id, cost, arrival


# ---------------------------------------------------------------------------
# Tools for the inventory agent (stock knowledge and supplier purchasing)
# ---------------------------------------------------------------------------
@tool
def find_catalog_item(description: str) -> str:
    """Matches a customer's wording (for example 'heavy cardstock') to the closest catalog item names.

    Args:
        description: The item as the customer described it.

    Returns:
        JSON with up to three candidate catalog items, each with a match score, list price and units in stock today.
    """
    today = _today()
    candidates = []
    for score, name in _match_catalog(description):
        candidates.append({
            "item_name": name,
            "match_score": score,
            "category": _CATALOG[name]["category"],
            "in_stock_today": _stock(name, _asof(today)),
        })
    return _j({"query": description, "candidates": candidates,
               "note": "Use the exact item_name in later tool calls. Score 1.0 is an exact match."})


@tool
def check_supplier_delivery(quantity: int) -> str:
    """Estimates when the supplier could deliver an order of a given size if placed today.

    Args:
        quantity: Number of units that would be ordered from the supplier.

    Returns:
        JSON with the order date and the estimated supplier delivery date.
    """
    delivery = get_supplier_delivery_date(_today(), int(quantity))
    return _j({"order_date": _today(), "quantity": int(quantity), "estimated_delivery_date": delivery})


@tool
def check_stock(item_name: str) -> str:
    """Checks stock for one item: units on hand today, units already on order, and its reorder threshold.

    Args:
        item_name: Exact catalog item name (use find_catalog_item first if unsure).

    Returns:
        JSON with on-hand stock, pending supplier deliveries, minimum stock level and a reorder flag.
    """
    item, error = _resolve(item_name)
    if error:
        return _j({"error": error})
    on_hand = _stock(item, _asof())
    projected = _stock(item, _asof(FAR_FUTURE))
    minimum = _min_stock_level(item)
    return _j({
        "item_name": item,
        "as_of": _today(),
        "on_hand": on_hand,
        "on_order_incoming": projected - on_hand,
        "min_stock_level": minimum,
        "below_minimum": projected < minimum,
    })


@tool
def list_inventory(category: str) -> str:
    """Lists every item currently in stock with its quantity, using the full inventory snapshot.

    Args:
        category: 'all' or a category such as 'paper', 'product', 'large_format' or 'specialty'.

    Returns:
        JSON mapping item names to units on hand, plus catalog items that are currently out of stock.
    """
    snapshot = get_all_inventory(_asof())
    wanted = category.strip().lower()
    in_stock = {
        name: int(units) for name, units in snapshot.items()
        if wanted in {"", "all"} or _CATALOG.get(name, {}).get("category") == wanted
    }
    out_of_stock = sorted(
        name for name, info in _CATALOG.items()
        if name not in snapshot and (wanted in {"", "all"} or info["category"] == wanted)
    )
    return _j({"as_of": _today(), "in_stock": in_stock, "not_in_stock": out_of_stock})


@tool
def list_low_stock_items(category: str) -> str:
    """Finds stocked items whose projected stock (on hand plus incoming) is below the minimum level.

    Args:
        category: 'all' or a category such as 'paper', 'product', 'large_format' or 'specialty'.

    Returns:
        JSON list of items to reorder with on-hand, incoming and minimum quantities.
    """
    wanted = category.strip().lower()
    inventory_rows = pd.read_sql("SELECT item_name, category, min_stock_level FROM inventory", db_engine)
    low = []
    for _, row in inventory_rows.iterrows():
        if wanted not in {"", "all"} and row["category"] != wanted:
            continue
        on_hand = _stock(row["item_name"], _asof())
        projected = _stock(row["item_name"], _asof(FAR_FUTURE))
        if projected < int(row["min_stock_level"]):
            low.append({"item_name": row["item_name"], "on_hand": on_hand,
                        "incoming": projected - on_hand, "min_stock_level": int(row["min_stock_level"])})
    return _j({"as_of": _today(), "low_stock_items": low})


@tool
def reorder_stock(item_name: str, quantity: int) -> str:
    """Places a supplier order for one item. Stock becomes available on the supplier delivery date.

    Args:
        item_name: Exact catalog item name.
        quantity: Number of units to order from the supplier.

    Returns:
        JSON confirming the order, its cost and arrival date, or the reason it was refused.
    """
    item, error = _resolve(item_name)
    if error:
        return _j({"status": "refused", "reason": error})
    if int(quantity) <= 0:
        return _j({"status": "refused", "reason": "Quantity must be positive."})
    cost = _supplier_cost(item, int(quantity))
    cash = get_cash_balance(_asof())
    if cash - cost < CASH_RESERVE:
        return _j({"status": "refused", "reason": "Not enough available cash for this purchase."})
    transaction_id, cost, arrival = _place_stock_order(item, int(quantity))
    return _j({"status": "ordered", "transaction_id": transaction_id, "item_name": item,
               "quantity": int(quantity), "cost": cost, "arrival_date": arrival})


@tool
def secure_stock_for_order(order_lines: str, delivery_deadline: str) -> str:
    """Makes sure stock will be available for a customer order, ordering any shortfall from the supplier.
    Nothing is ordered unless every line can be ready by the customer's required date.

    Args:
        order_lines: Order as 'Item name:quantity' entries separated by semicolons, for example 'A4 paper:500;Cardstock:200'.
        delivery_deadline: Date the customer needs the goods by as YYYY-MM-DD, or 'none'.

    Returns:
        JSON with status 'secured' or 'cannot_secure', the stock position of each line, units ordered from the supplier,
        the date the whole order can be ready, and any problems found.
    """
    today = _today()
    lines, problems = _parse_lines(order_lines)
    deadline, deadline_problem = _parse_deadline(delivery_deadline)
    if deadline_problem:
        problems.append(deadline_problem)
    if deadline and deadline < today:
        problems.append(f"The required date {deadline} is before the request date {today}.")
    if not lines and not problems:
        problems.append("No order lines found.")

    stock_plan, supplier_orders, order_cost = [], [], 0.0
    for item, quantity in lines:
        on_hand = _stock(item, _asof(today))
        entry = {"item": item, "quantity": quantity, "on_hand": on_hand, "ordered_from_supplier": 0}
        ready = _ready_date(item, quantity, today)
        if ready and (not deadline or ready <= deadline):
            entry["ready_date"] = ready  # current stock plus deliveries already booked is enough
        else:
            # Only deliveries that arrive by the deadline count towards the customer's order
            shortfall = quantity - _stock(item, _asof(deadline or FAR_FUTURE))
            if shortfall <= 0:
                shortfall = quantity - on_hand
            arrival = get_supplier_delivery_date(today, shortfall)
            new_ready = _ready_date(item, quantity, today, extra_units=shortfall, extra_arrival=arrival)
            if new_ready is None or (deadline and new_ready > deadline):
                problems.append(
                    f"{item}: the earliest we can have {quantity} units ready is {new_ready or arrival}, "
                    f"which is after the required date {deadline}."
                )
            else:
                entry.update(ordered_from_supplier=shortfall, ready_date=new_ready)
                supplier_orders.append((item, shortfall))
                order_cost += _supplier_cost(item, shortfall)
        stock_plan.append(entry)

    if supplier_orders and get_cash_balance(_asof(today)) - order_cost < CASH_RESERVE:
        problems.append("We cannot source the missing stock for this order at the moment.")
    if problems:
        return _j({"status": "cannot_secure", "problems": problems, "lines": stock_plan,
                   "note": "Nothing was ordered from the supplier."})

    for item, shortfall in supplier_orders:
        _place_stock_order(item, shortfall)
    return _j({"status": "secured", "request_date": today,
               "ready_date": max(entry["ready_date"] for entry in stock_plan), "lines": stock_plan})


@tool
def restock_low_items(category: str) -> str:
    """Replenishes every stocked item whose projected stock is below its minimum level, up to a target level.

    Args:
        category: 'all' or a category such as 'paper', 'product', 'large_format' or 'specialty'.

    Returns:
        JSON listing each replenishment order placed and any that were skipped.
    """
    wanted = category.strip().lower()
    inventory_rows = pd.read_sql("SELECT item_name, category, min_stock_level FROM inventory", db_engine)
    placed, skipped = [], []
    cash = get_cash_balance(_asof())
    for _, row in inventory_rows.iterrows():
        if wanted not in {"", "all"} and row["category"] != wanted:
            continue
        item, minimum = row["item_name"], int(row["min_stock_level"])
        projected = _stock(item, _asof(FAR_FUTURE))
        if projected >= minimum:
            continue
        quantity = minimum * RESTOCK_TARGET_MULTIPLIER - projected
        cost = _supplier_cost(item, quantity)
        if cash - cost < CASH_RESERVE:
            skipped.append({"item_name": item, "reason": "insufficient cash"})
            continue
        _, cost, arrival = _place_stock_order(item, quantity)
        cash -= cost
        # Supplier cost is deliberately left out: this result reaches the orchestrator, which writes to customers
        placed.append({"item_name": item, "quantity": quantity, "arrival_date": arrival})
    return _j({"as_of": _today(), "orders_placed": placed, "skipped": skipped})


@tool
def get_financial_report(detail: str) -> str:
    """Builds the internal financial and inventory report for today. Internal use only.

    Args:
        detail: 'summary' for headline numbers or 'full' to include the itemised inventory valuation.

    Returns:
        JSON with cash balance, inventory value, total assets and top-selling products.
    """
    report = generate_financial_report(_asof())
    result = {
        "as_of": _today(),
        "cash_balance": round(report["cash_balance"], 2),
        "inventory_value": round(report["inventory_value"], 2),
        "total_assets": round(report["total_assets"], 2),
        # The starting-cash entry has no item name, so leave it out of the best-seller list
        "top_selling_products": [p for p in report["top_selling_products"] if isinstance(p["item_name"], str)],
    }
    if detail.strip().lower() == "full":
        result["inventory_summary"] = [
            {"item_name": i["item_name"], "stock": int(i["stock"]), "value": round(float(i["value"]), 2)}
            for i in report["inventory_summary"]
        ]
    return _j(result)


# ---------------------------------------------------------------------------
# Tools for the quoting agent
# ---------------------------------------------------------------------------
_DISCOUNT_RE = re.compile(r"(\d{1,2})\s*%\s*(?:bulk\s+)?(?:discount|off)", re.IGNORECASE)


def _is_usable_quote(quote: Dict) -> bool:
    """History contains failed rows (total -1, 'Error parsing response'), which must not guide pricing."""
    explanation = str(quote["quote_explanation"]).lower()
    return float(quote["total_amount"]) > 0 and "error parsing" not in explanation


@tool
def search_quotes(search_terms: str, limit: int) -> str:
    """Searches historical quotes for similar past requests to guide pricing.

    Args:
        search_terms: Comma-separated keywords such as an item, job or event type, for example 'cardstock, festival'.
        limit: Maximum number of past quotes to return.

    Returns:
        JSON with similar past quotes (failed history rows removed), the bulk discounts those quotes mention
        and their typical discount percent.
    """
    limit = max(1, min(int(limit), 10))
    seen, results = set(), []
    # search_quote_history requires ALL terms to match, so query term by term
    # and merge, which behaves like "match any" and returns more useful history.
    for term in [t.strip() for t in search_terms.split(",") if t.strip()]:
        for quote in search_quote_history([term], limit=limit * 2):
            key = (quote["original_request"], quote["total_amount"])
            if key in seen or not _is_usable_quote(quote):
                continue
            seen.add(key)
            results.append({
                "original_request": str(quote["original_request"])[:300],
                "total_amount": quote["total_amount"],
                "quote_explanation": str(quote["quote_explanation"])[:300],
                "job_type": quote["job_type"],
                "order_size": quote["order_size"],
                "event_type": quote["event_type"],
            })
    results = results[:limit]
    discounts = sorted(int(d) for r in results for d in _DISCOUNT_RE.findall(r["quote_explanation"]))
    return _j({
        "matches": results,
        "discounts_mentioned_percent": discounts,
        "typical_discount_percent": discounts[len(discounts) // 2] if discounts else None,
        "note": "Past totals are often rounded and inconsistent. Use the explanations and typical discount, "
                "not the totals, as guidance.",
    })


@tool
def calculate_quote(order_lines: str, extra_discount_percent: float) -> str:
    """Prices an order at catalog prices with bulk discounts, without changing any data.

    Args:
        order_lines: Order as 'Item name:quantity' entries separated by semicolons, for example 'A4 paper:500;Cardstock:200'.
        extra_discount_percent: Additional discretionary discount (0 to 5) for repeat or event customers. Use 0 if unsure.

    Returns:
        JSON with per-line prices, discounts and the order total, or the problems found in the order.
    """
    lines, problems = _parse_lines(order_lines)
    if problems or not lines:
        return _j({"status": "cannot_quote", "problems": problems or ["No order lines found."]})
    priced = _price_lines(lines, extra_discount_percent)
    return _j({
        "status": "quoted",
        "lines": priced,
        "order_total": round(sum(p["line_total"] for p in priced), 2),
        "bulk_discount_tiers": ", ".join(
            f"{int(d * 100)}% from {minimum} units" for minimum, d in sorted(DISCOUNT_TIERS)) + " per item",
    })


# ---------------------------------------------------------------------------
# Tools for the ordering agent (sales finalisation)
# ---------------------------------------------------------------------------
def _plan_sale(order_lines: str, delivery_deadline: str, extra_discount_percent: float) -> Dict:
    """Check that stock (on hand or already delivered by then) covers the order in time, and price it."""
    today = _today()
    lines, problems = _parse_lines(order_lines)
    deadline, deadline_problem = _parse_deadline(delivery_deadline)
    if deadline_problem:
        problems.append(deadline_problem)
    if deadline and deadline < today:
        problems.append(f"The required date {deadline} is before the request date {today}.")
    if not lines and not problems:
        problems.append("No order lines found.")

    sale_lines = []
    for item, quantity in lines:
        ready = _ready_date(item, quantity, today)
        if ready is None:
            problems.append(f"{item}: we cannot supply {quantity} units.")
        elif deadline and ready > deadline:
            problems.append(
                f"{item}: {quantity} units will only be ready on {ready}, after the required date {deadline}."
            )
        sale_lines.append({"item": item, "quantity": quantity, "ready_date": ready})

    priced = _price_lines(lines, extra_discount_percent) if lines else []
    ready_dates = [line["ready_date"] for line in sale_lines if line["ready_date"]]
    return {
        "feasible": not problems,
        "problems": problems,
        "request_date": today,
        "required_by": deadline,
        "ready_date": max(ready_dates) if ready_dates else None,
        "lines": sale_lines,
        "pricing": priced,
        "order_total": round(sum(p["line_total"] for p in priced), 2),
    }


def _allocation_plan(item_name: str, quantity: int, today: str, ready_date: str) -> List[tuple]:
    """
    Split a sale into (day, units) parts, taking stock on the earliest day it exists.

    Units already on hand are reserved today and units still to be delivered are reserved on
    their arrival date, so the ledger never shows the same unit sold twice.
    """
    arrivals = [day for day in _incoming_arrival_dates(item_name, today) if day <= ready_date]
    parts, remaining, allocated = [], quantity, 0
    for day in sorted({today, ready_date, *arrivals}):
        available = _stock(item_name, _asof(day)) - allocated
        take = min(remaining, max(available, 0))
        if take > 0:
            parts.append((day, take))
            allocated += take
            remaining -= take
        if remaining == 0:
            break
    if remaining > 0:  # not expected, because the plan already proved stock is ready by ready_date
        parts.append((ready_date, remaining))
    return parts


@tool
def check_order_feasibility(order_lines: str, delivery_deadline: str, extra_discount_percent: float) -> str:
    """Dry run: checks that stock covers the order by the customer's deadline, without recording anything.

    Args:
        order_lines: Order as 'Item name:quantity' entries separated by semicolons.
        delivery_deadline: Date the customer needs the goods by as YYYY-MM-DD, or 'none'.
        extra_discount_percent: The extra discount used in the quote (0 to 5).

    Returns:
        JSON saying whether the order can be fulfilled, the ready date, the price, and every problem found.
    """
    return _j(_plan_sale(order_lines, delivery_deadline, extra_discount_percent))


@tool
def fulfill_order(order_lines: str, delivery_deadline: str, extra_discount_percent: float) -> str:
    """Completes a sale: re-checks stock and the deadline, then records the sale in the database. All lines or nothing.

    Args:
        order_lines: Order as 'Item name:quantity' entries separated by semicolons.
        delivery_deadline: Date the customer needs the goods by as YYYY-MM-DD, or 'none'.
        extra_discount_percent: The extra discount used in the quote (0 to 5).

    Returns:
        JSON with status 'fulfilled' and the confirmed total and ready date, or status 'rejected' with the reasons.
    """
    plan = _plan_sale(order_lines, delivery_deadline, extra_discount_percent)
    if not plan["feasible"]:
        return _j({"status": "rejected", "reasons": plan["problems"], "request_date": plan["request_date"]})

    line_ready = {line["item"]: line["ready_date"] for line in plan["lines"]}
    sold = []
    for price in plan["pricing"]:
        # Book each unit on the day it becomes available, so stock is reserved at once and
        # a later order cannot sell the same units again.
        parts = _allocation_plan(price["item"], int(price["quantity"]), plan["request_date"], line_ready[price["item"]])
        booked = 0.0
        for position, (day, units) in enumerate(parts):
            is_last = position == len(parts) - 1
            amount = round(price["line_total"] - booked, 2) if is_last else round(price["line_total"] * units / price["quantity"], 2)
            create_transaction(price["item"], "sales", int(units), amount, day)
            booked += amount
        sold.append({"item": price["item"], "quantity": price["quantity"],
                     "unit_price": price["unit_price"], "discount_percent": price["discount_percent"],
                     "line_total": price["line_total"]})
    return _j({"status": "fulfilled", "order_total": plan["order_total"],
               "ready_date": plan["ready_date"], "required_by": plan["required_by"], "items": sold})


@tool
def get_cash_position(purpose: str) -> str:
    """Returns the company's current cash balance. Internal use only, never share it with customers.

    Args:
        purpose: Short reason for the check, for example 'large order health check'.

    Returns:
        JSON with the cash balance as of today.
    """
    return _j({"as_of": _today(), "purpose": purpose, "cash_balance": round(get_cash_balance(_asof()), 2)})


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def _make_agent(cls, instructions: str, **kwargs):
    """Create a ToolCallingAgent and give it role instructions in a way that works across smolagents versions."""
    try:
        return cls(instructions=instructions, **kwargs)
    except TypeError:
        # Older smolagents has no `instructions` argument: append to the system prompt template instead
        agent = cls(**kwargs)
        agent.prompt_templates["system_prompt"] += "\n\n" + instructions
        try:
            # Some versions render the system prompt once at construction, so render it again
            agent.system_prompt = agent.initialize_system_prompt()
        except (AttributeError, TypeError):
            pass
        return agent


class InventoryAgent(ToolCallingAgent):
    """Owns stock knowledge and supplier purchasing: stock questions, reorders and securing stock for orders."""


class QuotingAgent(ToolCallingAgent):
    """Builds competitive quotes using historical quotes and bulk-discount strategy."""


class OrderingAgent(ToolCallingAgent):
    """Finalises sales: verifies stock and delivery timelines, then records the transaction."""


class OrchestratorAgent(ToolCallingAgent):
    """Receives customer requests and coordinates the three worker agents."""


INVENTORY_INSTRUCTIONS = """
You are the inventory manager of the Beaver's Choice Paper Company. You own stock levels and all supplier purchasing.
- Never guess stock levels. Always use your tools. Call find_catalog_item for every item the customer names and use the
  exact item_name it returns (its top candidate is right when match_score is 0.9 or higher) in every later tool call.
- Use check_stock for single items, list_inventory for overviews and check_supplier_delivery for lead times. Consider stock already on order.
- For a customer purchase, call secure_stock_for_order with 'Item name:quantity;Item name:quantity' and the required date (YYYY-MM-DD or 'none').
  It orders any shortfall from the supplier only when the whole order can be ready in time. Report its status, ready date and every problem.
- Reorder a single item with reorder_stock, or replenish everything below its minimum level with restock_low_items.
- Use get_financial_report for internal reporting.
- Reply with concise facts: exact catalog item names, quantities and dates. Never quote supplier prices in customer-facing wording.
- Customers use loose wording. Match by paper TYPE, not size: A3, A4, A5 and 8.5x11 do not create separate items
  ('A4 glossy paper' is 'Glossy paper', 'A3 matte paper' is 'Matte paper', 'heavy cardstock' is 'Cardstock', 'poster board' is 'Poster paper').
  Plain 'printer', 'copy' or 'white' paper is 'Standard copy paper' unless A4, letter or legal is named.
- Use quantities exactly as written. Reams, packs, boxes and rolls each count as one unit.
- If something is not sold at all (for example balloons or tickets) never substitute it: report it as not in the catalog.
  When you matched the customer's wording to a differently named catalog item, say so in your reply.
"""
INVENTORY_INSTRUCTIONS += "Catalog item names (use them exactly): " + "; ".join(_CATALOG) + "\n"

QUOTING_INSTRUCTIONS = """
You are the pricing specialist of the Beaver's Choice Paper Company. You own prices and discounts, nothing else.
- Call search_quotes with 2-4 keywords (item, job, event type) to see how similar past requests were priced.
  History is noisy: rely on typical_discount_percent and the explanations, never copy a past total.
- Build the price with calculate_quote using 'Item name:quantity;Item name:quantity' (use the exact catalog names you are given). Bulk discounts are automatic.
- You may add up to 5 extra_discount_percent only when similar past quotes gave a clearly larger discount than the automatic tier, otherwise use 0.
- Reply with a per-line breakdown, the discount applied, the total, the extra discount percent you used and a one-sentence explanation a customer can read.
- Never mention internal costs, margins or the markup.
"""

ORDERING_INSTRUCTIONS = """
You are the order fulfilment specialist of the Beaver's Choice Paper Company. You own the sale: final checks and the database record.
- You are given the agreed order lines, the customer's required date (YYYY-MM-DD or 'none'), the extra discount used in the quote and the quoted total.
- For orders quoted above 2000 dollars, call get_cash_position first as an internal health check. Never share that number.
- Call check_order_feasibility, then fulfill_order with exactly the same arguments if it is feasible.
- Report the tool result exactly: status, total, ready date, or every rejection reason. Never claim an order is fulfilled unless fulfill_order returned status 'fulfilled'.
"""

ORCHESTRATOR_INSTRUCTIONS = """
You handle customer requests for the Beaver's Choice Paper Company. You do not use tools yourself, you delegate to your team. Follow this workflow:
1. Read the request and the customer context (job, event, order size). List each item and quantity exactly as written by the customer,
   and the date the customer needs the goods by (YYYY-MM-DD, assume the request year if missing, or 'none').
2. If the customer only asks a question (stock, availability), ask inventory_agent and answer.
3. Otherwise ask inventory_agent to secure stock for the order (items as 'Item name:quantity;Item name:quantity', plus the required date).
   It also translates the customer's wording into exact catalog names: use those names from now on.
   If it reports the order cannot be secured, stop here and explain the reasons to the customer. Nothing was ordered in that case.
   When an item is not in our catalog, say which one and that the order was not placed, and offer to go ahead without it.
4. Ask quoting_agent for a quote, giving it the exact catalog names, quantities, job and event.
5. Ask ordering_agent to fulfil the order with the same lines, required date, the extra discount percent from the quote and the quoted total.
6. After a fulfilled order, ask inventory_agent to run restock_low_items with category 'all'.
Write the final answer to the customer: whether the order was fulfilled, the items, prices, discounts and why they apply, the total and the ready date.
If it could not be fulfilled, give the plain reason and an alternative (a smaller quantity or a later date).
Never reveal internal information: cash balances, supplier costs, margins, reorder logic, tool names or error messages.
Never invent numbers, use only what your team reports.
Check the request line by line: every item the customer names must appear in the lines you send, with the quantity they wrote.
Never add an item or a quantity the customer did not ask for, and never drop one silently. If an item cannot be supplied, say which one.
Write for a customer: never say 'not recognized', 'not in the catalog', 'stock', 'inventory' or 'incoming'. Say 'we do not carry X'
or 'we cannot deliver X by that date, the earliest is D', offer the closest product we do sell, and never give unit counts we hold.
"""


def build_multi_agent_system(model) -> OrchestratorAgent:
    """Wire up the four agents: an orchestrator plus inventory, quoting and ordering workers."""
    inventory_agent = _make_agent(
        InventoryAgent, INVENTORY_INSTRUCTIONS,
        tools=[find_catalog_item, check_stock, list_inventory, list_low_stock_items,
               check_supplier_delivery, secure_stock_for_order, reorder_stock,
               restock_low_items, get_financial_report],
        model=model, name="inventory_agent", max_steps=10,
        description="Owns stock levels and supplier purchasing: answers stock questions, checks lead times, "
                    "secures stock for a customer order and replenishes low stock. Give it item names and quantities.",
    )
    quoting_agent = _make_agent(
        QuotingAgent, QUOTING_INSTRUCTIONS,
        tools=[search_quotes, calculate_quote],
        model=model, name="quoting_agent", max_steps=10,
        description="Prepares a customer quote using historical quotes and bulk discounts. "
                    "Give it order lines as 'Item name:quantity;Item name:quantity' plus the job and event.",
    )
    ordering_agent = _make_agent(
        OrderingAgent, ORDERING_INSTRUCTIONS,
        tools=[check_order_feasibility, fulfill_order, get_cash_position],
        model=model, name="ordering_agent", max_steps=10,
        description="Finalises a sale: verifies stock and the delivery deadline, then records the sale in the database. "
                    "Give it order lines, the required date (YYYY-MM-DD or 'none'), the extra discount percent and the quoted total.",
    )
    return _make_agent(
        OrchestratorAgent, ORCHESTRATOR_INSTRUCTIONS,
        tools=[], managed_agents=[inventory_agent, quoting_agent, ordering_agent],
        model=model, name="orchestrator_agent", max_steps=15,
        description="Coordinates inventory, quoting and ordering for customer requests.",
    )


def handle_request(orchestrator: OrchestratorAgent, request_with_date: str, customer_context: str = "") -> str:
    """
    Run one customer request through the system.

    The request text ends with '(Date of request: YYYY-MM-DD)'. `customer_context` carries the
    job, event and order size recorded for the request, which helps the quoting agent search history.
    """
    match = _DATE_RE.search(request_with_date)
    _ctx["date"] = match.group(1) if match else None
    try:
        return str(orchestrator.run(
            f"Customer request:\n{request_with_date}\n\n"
            f"Customer context: {customer_context or 'not provided'}\n"
            f"The current date is {_today()}."
        ))
    except Exception as exc:  # keep the evaluation loop alive if one request fails
        return f"We are sorry, we could not process this request right now ({type(exc).__name__})."


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios(model=None):

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    orchestrator = build_multi_agent_system(model or build_model())

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        customer_context = f"job: {row['job']}; event: {row['event']}; order size: {row['need_size']}"
        response = handle_request(orchestrator, request_with_date, customer_context)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
