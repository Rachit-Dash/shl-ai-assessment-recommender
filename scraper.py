import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from utils.helpers import TEST_TYPE_LABELS, clean_text, describe_test_type, parse_test_type_codes


BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"
OUTPUT_PATH = Path("catalog/shl_catalog.json")


def fetch(session: requests.Session, url: str, params: Optional[dict] = None, timeout: int = 20) -> Optional[str]:
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        print(f"Warning: failed to fetch {url}: {exc}")
        return None


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; SHLInternAssignmentBot/1.0; +https://www.shl.com/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def extract_catalog_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "/products/product-catalog/view/" not in href and "/solutions/products/product-catalog/view/" not in href:
            continue
        absolute = urljoin(BASE_URL, href.split("#")[0])
        links.append(absolute)
    return sorted(set(links))


def scrape_catalog_links(session: requests.Session, page_size: int = 12, max_pages: int = 80) -> List[str]:
    all_links = []
    seen = set()
    for page in range(max_pages):
        start = page * page_size
        html = fetch(session, CATALOG_URL, params={"type": 2, "start": start})
        if not html:
            break
        links = extract_catalog_links(html)
        new_links = [link for link in links if link not in seen]
        if not new_links:
            break
        for link in new_links:
            seen.add(link)
            all_links.append(link)
        time.sleep(0.3)
    return all_links


def page_lines(soup: BeautifulSoup) -> List[str]:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return [clean_text(line) for line in soup.get_text("\n").splitlines() if clean_text(line)]


def find_after_label(lines: List[str], labels: Iterable[str], max_lookahead: int = 4) -> str:
    lowered_labels = [label.lower() for label in labels]
    for index, line in enumerate(lines):
        lower = line.lower().rstrip(":")
        for label in lowered_labels:
            if lower == label.rstrip(":"):
                for offset in range(1, max_lookahead + 1):
                    if index + offset < len(lines) and lines[index + offset]:
                        return clean_text(lines[index + offset])
            if lower.startswith(label.rstrip(":")):
                remainder = clean_text(line[len(label) :].strip(" :-"))
                if remainder:
                    return remainder
    return ""


def extract_between(lines: List[str], start_labels: Iterable[str], stop_labels: Iterable[str]) -> str:
    start_index = None
    start_labels_lower = [label.lower() for label in start_labels]
    stop_labels_lower = [label.lower() for label in stop_labels]
    for index, line in enumerate(lines):
        if line.lower().rstrip(":") in [label.rstrip(":") for label in start_labels_lower]:
            start_index = index + 1
            break
    if start_index is None:
        return ""
    collected = []
    for line in lines[start_index:]:
        lowered = line.lower().rstrip(":")
        if lowered in [label.rstrip(":") for label in stop_labels_lower]:
            break
        if line.lower() in {"back to product catalog", "download fact sheet"}:
            continue
        collected.append(line)
    return clean_text(" ".join(collected))


def infer_category(test_type: str, description: str) -> str:
    codes = parse_test_type_codes(test_type)
    if codes:
        return ", ".join(TEST_TYPE_LABELS[code] for code in codes)
    lowered = description.lower()
    if any(word in lowered for word in ["java", "python", "sql", "coding", "programming", "technical"]):
        return "Knowledge & Skills"
    if any(word in lowered for word in ["personality", "behavior", "behaviour"]):
        return "Personality & Behavior"
    if any(word in lowered for word in ["reasoning", "cognitive", "ability", "aptitude"]):
        return "Ability & Aptitude"
    return "Individual Test Solution"


def is_likely_job_solution(name: str, url: str) -> bool:
    normalized = f"{name} {url}".lower()
    if "individual" in normalized or "verify" in normalized:
        return False
    job_solution_markers = [
        "solution/",
        " solution",
        "7.0 solution",
        "job focused",
        "job-focused",
        "job family",
    ]
    return any(marker in normalized for marker in job_solution_markers)


def parse_product_page(session: requests.Session, url: str) -> Optional[Dict[str, str]]:
    html = fetch(session, url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    lines = page_lines(soup)
    heading = soup.find(["h1", "h2"])
    name = clean_text(heading.get_text(" ")) if heading else ""
    if not name:
        for line in lines:
            if line.lower() not in {"products", "product catalog", "back to product catalog"}:
                name = line
                break
    name = re.sub(r"^#+\s*", "", name).strip()
    if not name or is_likely_job_solution(name, url):
        return None

    description = extract_between(
        lines,
        ["Description"],
        [
            "Job levels",
            "Languages",
            "Assessment length",
            "Approximate Completion Time in minutes",
            "Test Type",
            "Remote Testing",
            "Downloads",
        ],
    )
    if not description:
        meta_description = soup.find("meta", attrs={"name": "description"})
        description = clean_text(meta_description.get("content", "")) if meta_description else ""

    duration = find_after_label(lines, ["Approximate Completion Time in minutes", "Assessment length", "Duration"])
    test_type = find_after_label(lines, ["Test Type"])
    remote_testing = find_after_label(lines, ["Remote Testing", "Remote Testing Support"])
    adaptive = find_after_label(lines, ["Adaptive/IRT", "Adaptive Support", "Adaptive"])

    if not test_type:
        test_type = infer_category("", description)

    return {
        "name": name,
        "url": url,
        "description": description,
        "category": infer_category(test_type, description),
        "duration": duration or "Not specified",
        "remote_testing_support": remote_testing or "Not specified",
        "adaptive_support": adaptive or "Not specified",
        "test_type": describe_test_type(test_type),
    }


def dedupe_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    output = []
    for item in items:
        key = item["url"].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    output.sort(key=lambda row: row["name"].lower())
    return output


def scrape(output_path: Path = OUTPUT_PATH) -> List[Dict[str, str]]:
    session = make_session()
    links = scrape_catalog_links(session)
    print(f"Found {len(links)} candidate SHL catalog links")
    items = []
    for index, link in enumerate(links, start=1):
        item = parse_product_page(session, link)
        if item:
            items.append(item)
        print(f"[{index}/{len(links)}] {link}")
        time.sleep(0.2)

    items = dedupe_items(items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(items, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(items)} catalog items to {output_path}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL Individual Test Solutions catalog.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()
    scrape(Path(args.output))


if __name__ == "__main__":
    main()
