from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Enable CORS for all origins (for public web usage; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set this to your site URL for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample fund data (add real data as needed)
FUNDS = [
    {"name": "Large Cap Alpha Fund", "type": "Large Cap", "xirr": 19, "bear": 14, "bull": 21, "style": "Growth", "risk": "Conservative", "min_years": 5},
    {"name": "Flexi Cap Pro Fund", "type": "Flexi Cap", "xirr": 21, "bear": 15, "bull": 23, "style": "Blend", "risk": "Moderate", "min_years": 7},
    {"name": "Mid Cap Growth Fund", "type": "Mid Cap", "xirr": 22, "bear": 13, "bull": 24, "style": "Aggressive", "risk": "Aggressive", "min_years": 10},
    {"name": "Hybrid Secure Fund", "type": "Hybrid", "xirr": 16, "bear": 13, "bull": 18, "style": "Balanced", "risk": "Conservative", "min_years": 3},
    {"name": "Global Equity Fund", "type": "International", "xirr": 18, "bear": 13, "bull": 20, "style": "Diversifier", "risk": "Moderate", "min_years": 5},
    # Add more funds as required
]

class RecommendRequest(BaseModel):
    goal: str
    tenor: int
    amount: int
    approach: str
    risk: str

class FundRecommendation(BaseModel):
    name: str
    type: str
    xirr: float
    bear: float
    bull: float
    style: str
    explanation: str

class RecommendResponse(BaseModel):
    recommendations: List[FundRecommendation]

def veteran_explanation(fund, risk):
    return (
        f"Selected {fund['name']} as a {fund['type']} fund with veteran XIRR of {fund['xirr']}% "
        f"(bear market: {fund['bear']}%, bull market: {fund['bull']}%). "
        f"Suitable for a {risk} investor due to its {fund['style'].lower()} style and proven performance."
    )

def select_funds(goal, tenor, amount, approach, risk):
    # Determine fund limit
    fund_limit = 3
    if approach.lower().startswith("sip") and 5000 < amount <= 10000:
        fund_limit = 5
    elif approach.lower().startswith("sip") and amount > 10000:
        fund_limit = 7

    # Filter funds matching the veteran criteria
    shortlist = [
        f for f in FUNDS
        if f["min_years"] <= tenor
        and f["xirr"] >= 18 and f["bear"] >= 13 and f["bull"] >= 20
        and (
            (risk == "Conservative" and (f["type"] in ["Hybrid", "Large Cap"])) or
            (risk == "Moderate" and f["type"] in ["Large Cap", "Flexi Cap", "Hybrid", "International"]) or
            (risk == "Aggressive")
        )
    ]
    seen_types = set()
    selected = []
    for fund in shortlist:
        if fund["type"] not in seen_types:
            fund_copy = fund.copy()
            fund_copy["explanation"] = veteran_explanation(fund, risk)
            selected.append(fund_copy)
            seen_types.add(fund["type"])
        if len(selected) == fund_limit:
            break
    return selected

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    funds = select_funds(
        request.goal, request.tenor, request.amount,
        request.approach, request.risk
    )
    return {"recommendations": funds}
