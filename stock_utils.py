import pandas as pd
from rapidfuzz import process, fuzz

# Predefined universe of supported stocks (e.g. NIFTY 50)
SUPPORTED_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "LT.NS": "Larsen & Toubro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "HCLTECH.NS": "HCL Technologies",
    "TATAMOTORS.NS": "Tata Motors",
    "SUNPHARMA.NS": "Sun Pharmaceuticals",
    "MARUTI.NS": "Maruti Suzuki",
    "NTPC.NS": "NTPC",
    "TATASTEEL.NS": "Tata Steel",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid Corp",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "M&M.NS": "Mahindra & Mahindra",
    "WIPRO.NS": "Wipro",
    "ASIANPAINT.NS": "Asian Paints",
    "ADANIENT.NS": "Adani Enterprises",
    "TITAN.NS": "Titan Company",
    "NESTLEIND.NS": "Nestle India",
    "JSWSTEEL.NS": "JSW Steel",
    "TECHM.NS": "Tech Mahindra",
    "HINDALCO.NS": "Hindalco Industries",
    "HAL.NS": "Hindustan Aeronautics",
    "TRENT.NS": "Trent",
    "BEL.NS": "Bharat Electronics",
    "TATACONSUM.NS": "Tata Consumer",
    "CIPLA.NS": "Cipla",
    "GRASIM.NS": "Grasim Industries",
    "COALINDIA.NS": "Coal India",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "DRREDDY.NS": "Dr Reddy's Laboratories",
    "BRITANNIA.NS": "Britannia Industries",
    "EICHERMOT.NS": "Eicher Motors",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "INDUSINDBK.NS": "IndusInd Bank",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "DIVISLAB.NS": "Divi's Laboratories",
    "SBILIFE.NS": "SBI Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "UPL.NS": "UPL",
    # ---------------------------------------------
    # High-Volatility / Penny Stocks for Demo
    # ---------------------------------------------
    "SUZLON.NS": "Suzlon Energy",
    "YESBANK.NS": "Yes Bank",
    "IDEA.NS": "Vodafone Idea",
    "RPOWER.NS": "Reliance Power",
    "JPASSOCIAT.NS": "Jaiprakash Associates",
    "GTLINFRA.NS": "GTL Infrastructure",
    "SOUTHBANK.NS": "South Indian Bank"
}

def get_stock_suggestions(query: str, limit: int = 5) -> list[dict]:
    """
    Returns the top fuzzy matches for a user query against the stock universe.
    It matches against both the ticker symbol and the company name.
    """
    if not query:
        return []
    
    query = query.upper().strip()
    
    # We will build a lookup list formatted as "TICKER | Name"
    # This allows rapidfuzz to match against either part natively.
    choices = [f"{ticker} | {name}" for ticker, name in SUPPORTED_STOCKS.items()]
    
    # process.extract returns a list of tuples: (match_string, score, index)
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
    
    suggestions = []
    for match_str, score, _ in results:
        # Only include reasonable matches to filter out complete noise
        if score > 40: 
            ticker, name = match_str.split(" | ")
            suggestions.append({
                "ticker": ticker,
                "name": name,
                "score": round(score, 1)
            })
            
    return suggestions
