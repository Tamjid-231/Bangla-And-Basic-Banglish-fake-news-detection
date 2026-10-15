import html
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

from banglish_normalizer import BANGLISH_WORDS, has_bangla, normalize_banglish
from preprocess import clean_text


SEARCH_URL = "https://news.google.com/rss/search"
SEARCH_TIMEOUT_SECONDS = 6
MAX_FEED_BYTES = 2_000_000
SEARCH_LOCALES = {
    "bn": {"hl": "bn", "gl": "BD", "ceid": "BD:bn"},
    "en": {"hl": "en", "gl": "BD", "ceid": "BD:en"},
}
SEARCH_AMBIGUOUS_ROMAN_WORDS = {"bangla", "bangladesh", "govt", "news"}

# Fact-check sources are checked before regular news sources.
FACT_CHECK_SOURCES = {
    "afp fact check",
    "boom bangladesh",
    "bdfactcheck",
    "fact crescendo",
    "rumor scanner",
    "rumorscanner",
    "যাচাই",
}

TRUSTED_NEWS_SOURCES = {
    "ajker patrika",
    "bangladesh protidin",
    "bbc bangla",
    "bbc news বাংলা",
    "bbc",
    "bdnews24",
    "bangla tribune",
    "bss",
    "channel 24",
    "daily ittefaq",
    "desh rupantor",
    "dhaka tribune",
    "ekushey tv",
    "kalbela",
    "prothom alo",
    "rtv bangladesh",
    "প্রথম আলো",
    "the daily star",
    "unb",
    "আজকের পত্রিকা",
    "দেশ রূপান্তর",
    "ইত্তেফাক",
    "কালবেলা",
    "কালের কণ্ঠ",
    "যুগান্তর",
    "সমকাল",
}

FALSE_WORDS = {
    "false",
    "fake",
    "hoax",
    "incorrect",
    "misleading",
    "বানোয়াট",
    "বিভ্রান্তিকর",
    "ভুয়া",
    "ভুয়া",
    "মিথ্যা",
    "গুজব",
}

TRUE_PHRASES = {
    "not fake",
    "not false",
    "গুজব নয়",
    "গুজব নয়",
    "ভুয়া নয়",
    "ভুয়া নয়",
}

MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "এই",
    "একটি",
    "এবং",
    "এর",
    "ও",
    "কি",
    "কে",
    "খবর",
    "যে",
    "সংবাদ",
}


def _plain_text(value: str) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def format_source_label(title: str, source: str) -> str:
    """Return one readable label without repeating the source name."""
    clean_title = _plain_text(title) or "Untitled report"
    clean_source = _plain_text(source)
    if not clean_source:
        return clean_title

    title_key = clean_title.casefold()
    source_key = clean_source.casefold()
    if title_key == source_key:
        return clean_title
    if any(
        title_key.endswith(f"{separator}{source_key}")
        for separator in (" - ", " – ", " — ", " | ")
    ):
        return clean_title
    return f"{clean_title} - {clean_source}"


def _source_kind(source_name: str, source_url: str) -> str:
    source_text = f"{source_name} {source_url}".lower()
    if any(name in source_text for name in FACT_CHECK_SOURCES):
        return "fact-check"
    if any(name in source_text for name in TRUSTED_NEWS_SOURCES):
        return "trusted-news"
    return "other"


def _important_tokens(text: str) -> set[str]:
    normalized_text = unicodedata.normalize("NFC", clean_text(text))
    tokens = re.findall(r"[\u0980-\u09ffa-z0-9]+", normalized_text)
    return {
        token
        for token in tokens
        if len(token) > 1 and token not in MATCH_STOPWORDS
    }


def matching_score(claim: str, evidence_text: str) -> int:
    """Return a simple text match score, not a truth probability."""
    claim_tokens = _important_tokens(claim)
    evidence_tokens = _important_tokens(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return 0

    overlap = len(claim_tokens & evidence_tokens)
    token_score = overlap / max(1, min(len(claim_tokens), 10))
    claim_part = " ".join(clean_text(claim).split()[:18])
    evidence_part = " ".join(clean_text(evidence_text).split()[:24])
    sequence_score = SequenceMatcher(None, claim_part, evidence_part).ratio()
    return round(min(1.0, token_score * 0.8 + sequence_score * 0.2) * 100)


def _fact_check_verdict(text: str) -> str | None:
    lowered = clean_text(text)
    if any(phrase in lowered for phrase in TRUE_PHRASES):
        return "true"
    if any(word in lowered.split() for word in FALSE_WORDS):
        return "false"
    if "সত্য প্রমাণিত" in lowered or "দাবিটি সত্য" in lowered:
        return "true"
    return None


def _clean_search_text(text: str) -> str:
    """Clean a search query without changing English words into Bangla."""
    text = unicodedata.normalize("NFC", str(text).lower())
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\u0980-\u09ffa-z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_search_query(text: str) -> str:
    cleaned = _clean_search_text(text)
    return " ".join(cleaned.split()[:18])


def build_search_queries(text: str) -> list[str]:
    """Return queries from the full text, including distinctive later sentences."""
    raw_query = build_search_query(text)
    if not raw_query:
        return []

    queries = [raw_query]
    full_cleaned = _clean_search_text(text)
    full_tokens = full_cleaned.split()

    if len(full_tokens) > 18:
        tail_query = " ".join(full_tokens[-10:])
        if tail_query not in queries:
            queries.append(tail_query)

    for sentence in re.split(r"[\u0964.!?\n]+", str(text)):
        sentence_query = build_search_query(sentence)
        if sentence_query and sentence_query not in queries:
            queries.append(sentence_query)
        if len(queries) >= 4:
            break

    roman_tokens = re.findall(r"[a-z]+", raw_query)
    banglish_hints = sum(
        1
        for token in roman_tokens
        if token in BANGLISH_WORDS and token not in SEARCH_AMBIGUOUS_ROMAN_WORDS
    )
    if not has_bangla(raw_query) and banglish_hints >= 2:
        for query in list(queries):
            normalized = build_search_query(normalize_banglish(query))
            if normalized and normalized not in queries:
                queries.append(normalized)
    return queries[:6]


def build_compact_query_variants(query: str) -> list[str]:
    """Create shorter fallback queries when a full headline is too restrictive."""
    tokens = query.split()
    if len(tokens) <= 4:
        return []

    middle_start = max(0, (len(tokens) - 3) // 2)
    candidates = [
        " ".join(tokens[middle_start : middle_start + 3]),
        " ".join(tokens[:5]),
        " ".join(tokens[-5:]),
    ]
    return [
        candidate
        for index, candidate in enumerate(candidates)
        if candidate
        and candidate != query
        and candidate not in candidates[:index]
    ]


def _read_google_news(
    query: str,
    max_results: int = 10,
    locale: str = "bn",
) -> list[dict]:
    # Google News RSS gives a small structured result without an extra package.
    parameters = urllib.parse.urlencode({"q": query, **SEARCH_LOCALES[locale]})
    request = urllib.request.Request(
        f"{SEARCH_URL}?{parameters}",
        headers={
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            "Accept-Language": "bn-BD,bn;q=0.9,en;q=0.7",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FakeNewsCapstone/1.1",
        },
    )
    with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
        raw_xml = response.read(MAX_FEED_BYTES + 1)
    if len(raw_xml) > MAX_FEED_BYTES:
        raise ValueError("News feed response is too large.")

    root = ET.fromstring(raw_xml)
    results = []
    for item in root.findall("./channel/item")[:max_results]:
        source_element = item.find("source")
        source_name = (
            _plain_text(source_element.text) if source_element is not None else "Unknown"
        )
        source_url = (
            source_element.attrib.get("url", "") if source_element is not None else ""
        )
        results.append(
            {
                "title": _plain_text(item.findtext("title", "")),
                "url": _plain_text(item.findtext("link", "")),
                "snippet": _plain_text(item.findtext("description", "")),
                "source": source_name,
                "source_url": source_url,
            }
        )
    return results


def _search_failure_reason(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"http_{error.code}"
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            return "timeout"
        return "connection_failed"
    if isinstance(error, ET.ParseError):
        return "invalid_feed"
    return "search_failed"


def _search_google_news(queries: list[str], max_results: int = 10) -> tuple[list[dict], str]:
    """Try useful query/locale combinations and merge duplicate-free results."""
    query_locales = [
        (query, "bn" if has_bangla(query) else "en")
        for query in queries
    ]
    plans = list(query_locales)

    for variant_index in range(3):
        for query, primary_locale in query_locales:
            compact_queries = build_compact_query_variants(query)
            if variant_index < len(compact_queries):
                plans.append((compact_queries[variant_index], primary_locale))

    plans.extend(
        (query, "en" if primary_locale == "bn" else "bn")
        for query, primary_locale in query_locales
    )

    unique_plans = []
    for plan in plans:
        if plan not in unique_plans:
            unique_plans.append(plan)
    plans = unique_plans[:6]

    collected = []
    seen = set()
    successful_query = queries[0]
    errors = []

    # Short fallback queries help when Google News treats a full headline as too strict.
    for attempt, (query, locale) in enumerate(plans, start=1):
        try:
            batch = _read_google_news(query, max_results=max_results, locale=locale)
            successful_query = query
            for item in batch:
                identity = item.get("url") or item.get("title")
                if identity and identity not in seen:
                    seen.add(identity)
                    collected.append({**item, "search_query": query})
            if len(collected) >= max_results:
                break
        except Exception as error:
            errors.append(error)
            if attempt < len(plans):
                time.sleep(0.25)

    if not collected and errors and len(errors) == len(plans):
        raise errors[-1]
    return collected[:max_results], successful_query


def analyze_evidence(claim: str, results: list[dict], model_prediction: str) -> dict:
    claim_tokens = _important_tokens(claim)
    minimum_overlap = min(2, len(claim_tokens))
    prepared = []
    for result in results:
        title = str(result.get("title", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        source = str(result.get("source", "Unknown")).strip()
        source_url = str(result.get("source_url", "")).strip()
        score = matching_score(claim, f"{title} {snippet}")
        token_overlap = len(
            claim_tokens & _important_tokens(f"{title} {snippet}")
        )
        source_kind = _source_kind(source, source_url)
        search_query_tokens = _important_tokens(result.get("search_query", ""))
        specific_query_match = (
            source_kind in {"trusted-news", "fact-check"}
            and len(search_query_tokens) >= 4
            and max((len(token) for token in search_query_tokens), default=0) >= 8
            and len(claim_tokens & search_query_tokens) >= 4
        )
        verdict = (
            _fact_check_verdict(f"{title} {snippet}")
            if source_kind == "fact-check"
            else None
        )
        prepared.append(
            {
                **result,
                "display_label": format_source_label(title, source),
                "match_score": score,
                "token_overlap": token_overlap,
                "specific_query_match": specific_query_match,
                "source_kind": source_kind,
                "fact_check_verdict": verdict,
            }
        )

    prepared.sort(
        key=lambda item: (
            item["source_kind"] == "fact-check",
            item["source_kind"] == "trusted-news",
            item["specific_query_match"],
            item["match_score"],
        ),
        reverse=True,
    )
    relevant = [
        item
        for item in prepared
        if (
            item["match_score"] >= 18
            and item["token_overlap"] >= minimum_overlap
        )
        or item["specific_query_match"]
    ]
    fact_false = [
        item
        for item in relevant
        if item["source_kind"] == "fact-check"
        and item["fact_check_verdict"] == "false"
    ]
    fact_true = [
        item
        for item in relevant
        if item["source_kind"] == "fact-check"
        and item["fact_check_verdict"] == "true"
    ]
    trusted_sources = {
        item["source"].lower()
        for item in relevant
        if item["source_kind"] in {"trusted-news", "fact-check"}
    }

    # A direct fact-check verdict has priority over normal article matches.
    if fact_false and fact_true:
        status = "conflicting_evidence"
        category = "conflicting"
        overall = "manual_review"
        note = "Trusted fact-check results do not agree. Manual review is needed."
    elif fact_false:
        status = "fact_checked_false"
        category = "conflicting"
        overall = "fake"
        note = "A relevant fact-check source rated this claim as false or misleading."
    elif fact_true:
        status = "fact_checked_true"
        category = "supporting"
        overall = "real"
        note = "A relevant fact-check source supports this claim."
    elif len(trusted_sources) >= 2:
        status = "supported_by_sources"
        category = "supporting"
        overall = "likely_real" if model_prediction == "real" else "manual_review"
        note = "At least two independent trusted sources contain a matching report."
    elif len(trusted_sources) == 1:
        status = "limited_evidence"
        category = "limited"
        overall = "unverified"
        note = "Only one relevant trusted source was found. More evidence is needed."
    else:
        status = "not_enough_evidence"
        category = "insufficient"
        overall = "unverified"
        note = "No sufficiently matching trusted source was found."

    return {
        "status": status,
        "category": category,
        "overall": overall,
        "note": note,
        "trusted_source_count": len(trusted_sources),
        "results": relevant[:6],
    }


def search_web_evidence(text: str, model_prediction: str) -> dict:
    queries = build_search_queries(text)
    if not queries:
        return {
            "available": False,
            "query": "",
            "query_variants": [],
            "provider": "Google News RSS",
            "status": "not_enough_evidence",
            "category": "insufficient",
            "overall": "unverified",
            "note": "The text did not contain enough searchable words.",
            "trusted_source_count": 0,
            "results": [],
        }

    try:
        results, used_query = _search_google_news(queries)
        evidence = analyze_evidence(text, results, model_prediction)
        return {
            "available": True,
            "query": used_query,
            "query_variants": queries,
            "provider": "Google News RSS",
            **evidence,
        }
    except Exception as error:
        # Prediction should still work if the internet or search service is down.
        failure_reason = _search_failure_reason(error)
        failure_notes = {
            "timeout": (
                "Web evidence is unavailable. Check your internet connection "
                "and try again."
            ),
            "connection_failed": (
                "Web evidence is unavailable. Check your internet connection "
                "and try again."
            ),
            "invalid_feed": "The news service returned an unreadable response. Please try again later.",
            "search_failed": "Live news search failed unexpectedly. Please try again later.",
        }
        if failure_reason.startswith("http_"):
            note = "Google News RSS temporarily rejected the request. Please try again later."
        else:
            note = failure_notes.get(failure_reason, failure_notes["search_failed"])
        return {
            "available": False,
            "query": queries[0],
            "query_variants": queries,
            "provider": "Google News RSS",
            "failure_reason": failure_reason,
            "status": "web_search_unavailable",
            "category": "unavailable",
            "overall": "unverified",
            "note": f"{note} The model prediction is still available.",
            "trusted_source_count": 0,
            "results": [],
        }
