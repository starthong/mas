from __future__ import annotations

from datetime import datetime, timedelta

from src.tools.registry import Tool, ToolRegistry


def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    rates = {
        ("USD", "CNY"): 7.1,
        ("CNY", "USD"): 1 / 7.1,
        ("EUR", "JPY"): 160.0,
        ("JPY", "CNY"): 0.048,
        ("GBP", "USD"): 1.25,
        ("USD", "EUR"): 0.92,
        ("HKD", "CNY"): 0.91,
        ("AUD", "USD"): 0.66,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return f"unsupported pair: {from_currency}->{to_currency}"
    return f"{amount * rate:.2f} {to_currency.upper()}"


def calendar_add(date: str, days: int) -> str:
    value = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=int(days))
    return value.strftime("%Y-%m-%d")


def translate(text: str, target_language: str) -> str:
    return f"[mock {target_language}] {text}"


def get_stock_price(symbol: str) -> str:
    prices = {
        "AAPL": "227.50 USD",
        "MSFT": "415.20 USD",
        "TSLA": "248.10 USD",
        "BABA": "82.40 USD",
        "GOOGL": "171.30 USD",
        "AMZN": "186.90 USD",
        "NVDA": "121.70 USD",
        "0700.HK": "420.00 HKD",
    }
    return prices.get(symbol.upper(), f"unknown symbol: {symbol}")


def final_answer(answer: str) -> str:
    return answer


def build_ood_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def add(name: str, description: str, properties: dict, required: list[str], func):
        registry.register(
            Tool(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                func=func,
            )
        )

    add(
        "currency_convert",
        "Convert an amount between currencies.",
        {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
        ["amount", "from_currency", "to_currency"],
        currency_convert,
    )
    add(
        "calendar_add",
        "Add or subtract days from a date in YYYY-MM-DD format.",
        {"date": {"type": "string"}, "days": {"type": "integer"}},
        ["date", "days"],
        calendar_add,
    )
    add(
        "translate",
        "Translate text into a target language.",
        {"text": {"type": "string"}, "target_language": {"type": "string"}},
        ["text", "target_language"],
        translate,
    )
    add(
        "get_stock_price",
        "Get a mock stock price for a ticker symbol.",
        {"symbol": {"type": "string"}},
        ["symbol"],
        get_stock_price,
    )
    add(
        "final_answer",
        "Return a final answer when no external tool is needed.",
        {"answer": {"type": "string"}},
        ["answer"],
        final_answer,
    )
    return registry
