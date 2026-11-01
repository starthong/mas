"""真实工具后端：Open-Meteo 天气、Serper/Tavily/Sogou/Wikipedia 检索。"""
from __future__ import annotations

import html
import os
import re
import urllib.parse
from functools import lru_cache

import requests

TIMEOUT = 8
USER_AGENT = "mas-toolcall-research/1.0"

_WMO = {
    0: "晴",
    1: "晴间多云",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


@lru_cache(maxsize=256)
def get_weather(city: str) -> str:
    """调用 Open-Meteo 获取城市当前天气。"""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"城市未找到: {city}")
    place = results[0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,"
            "relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    weather.raise_for_status()
    current = weather.json().get("current") or {}
    code = int(current.get("weather_code", -1))
    desc = _WMO.get(code, f"天气代码 {code}")
    observed = current.get("time", "")
    return (
        f"{city}：{desc}，{current.get('temperature_2m')}°C，"
        f"体感 {current.get('apparent_temperature')}°C，"
        f"湿度 {current.get('relative_humidity_2m')}%，"
        f"风速 {current.get('wind_speed_10m')} km/h"
        f"（Open-Meteo，{observed}）"
    )


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(text or "")).strip()


def _serper(query: str) -> list[str]:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return []
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": 3},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    items = response.json().get("organic") or []
    return [
        f"{item.get('title', '')}: {item.get('snippet', '')} ({item.get('link', '')})"
        for item in items[:3]
    ]


def _tavily(query: str) -> list[str]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": 3},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    items = response.json().get("results") or []
    return [
        f"{item.get('title', '')}: {item.get('content', '')} ({item.get('url', '')})"
        for item in items[:3]
    ]


def _sogou(query: str) -> list[str]:
    response = requests.get(
        "https://www.sogou.com/web",
        params={"query": query},
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    blocks = re.findall(
        r"<h3[^>]*>(.*?)</h3>(.*?)(?=<h3|\Z)",
        response.text,
        flags=re.S,
    )
    rows = []
    for heading, tail in blocks[:3]:
        title = _strip_html(heading)
        snippet = re.sub(r"\s+", " ", _strip_html(tail))[:240]
        if title:
            rows.append(f"{title}: {snippet}")
    return rows


def _wikipedia(query: str, lang: str) -> list[str]:
    response = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "srprop": "snippet",
            "format": "json",
            "utf8": 1,
        },
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    items = response.json().get("query", {}).get("search", [])
    return [
        f"{item.get('title', '')}: {_strip_html(item.get('snippet', ''))}" for item in items
    ]


@lru_cache(maxsize=256)
def search(query: str) -> str:
    """按 Serper、Tavily、Sogou、Wikipedia 顺序调用真实检索。"""
    errors: list[str] = []
    for provider, func in (
        ("Serper", _serper),
        ("Tavily", _tavily),
        ("Sogou", _sogou),
    ):
        try:
            rows = func(query)
            if rows:
                return f"[{provider}] " + " | ".join(rows)
        except Exception as exc:  # noqa: BLE001 - try next provider
            errors.append(f"{provider}: {exc}")

    for lang in ("zh", "en"):
        try:
            rows = _wikipedia(query, lang)
            if rows:
                return f"[Wikipedia/{lang}] " + " | ".join(rows)
        except Exception as exc:  # noqa: BLE001 - try next language
            errors.append(f"Wikipedia/{lang}: {exc}")

    raise RuntimeError("检索后端不可用: " + "; ".join(errors[-3:]))
