"""
OmniCompiler - Multi-Source, License-Aware Safe Image Retrieval Pipeline.

Enforces a 4-Layer Data-as-Code Contract:
  Layer 1: Strict Domain Whitelisting, HTTPS Enforcement, Redirect & SSRF Guard
  Layer 2: Fail-Closed robots.txt Check (RFC 9309), License Allowlist & Attribution Check
  Layer 3: Raw Byte Download, Size Guard, PIL Structure Verification & Raw SHA-256 Digest
  Layer 4: Dynamic User-Query CLIP Zero-Shot Quality & Relevance Gate

Default Target Split: 10 Total Images (8 Training + 2 Validation)

Usage:
  python tinyFish.py -q "sneaker in low light"
  python tinyFish.py
  python tinyFish.py --verify
"""

import argparse
import csv
import datetime
import hashlib
import html
import io
import ipaddress
import os
import re
import socket
import time
import uuid
from collections import Counter
from typing import Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import torch
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

# ==============================================================================
# CONFIG & POLICY CONSTANTS
# ==============================================================================

USER_AGENT = "OmniCompilerBot/1.0 (IIITD Hackathon Safe Retrieval; contact: researcher@example.com)"
POLICY_VERSION = "2026.10-v2"

# Default dataset target counts: 10 images total (8 train + 2 val)
DEFAULT_TRAIN_COUNT = 8
DEFAULT_VAL_COUNT = 2

TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY")
TINYFISH_ENDPOINT = os.getenv("TINYFISH_ENDPOINT", "https://api.tinyfish.io/v1/agent/run")
FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENVERSE_TOKEN = os.getenv("OPENVERSE_TOKEN")

# Canonical license keys produced by normalize_license(). Anything else is rejected.
ALLOWED_LICENSES = {"cc0", "pd", "cc-by", "cc-by-sa", "mit"}
ATTRIBUTION_REQUIRED_LICENSES = {"cc-by", "cc-by-sa"}
HF_ALLOWED_LICENSES = {"cc0-1.0", "mit"}

DATASET_DIR = "omni_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val", "images")
MANIFEST_DIR = os.path.join(DATASET_DIR, "manifests")
CACHE_DIR = os.path.join(DATASET_DIR, ".cache")

MAX_PAGES = 10
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MIN_SIDE = 128
DOWNLOAD_DELAY = 0.2
Image.MAX_IMAGE_PIXELS = 50_000_000

OI_CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
OI_LABELS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv"
OI_IMAGES_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

ROBOTS_CACHE: Dict[str, Optional[RobotFileParser]] = {}


# ==============================================================================
# SCHEMAS (DATA-AS-CODE CONTRACT)
# ==============================================================================


class ImageCandidate(BaseModel):
    source_name: str
    source_page_url: str
    direct_image_url: str
    license_label: str
    license_url: str = ""
    license_evidence: str = ""
    normalized_license: Optional[str]
    creator: str
    title: str = ""


class ProvenanceManifest(BaseModel):
    asset_id: str
    policy_version: str
    source_name: str
    retrieval_method: str
    query: str
    clip_target_prompt: str
    source_page_url: str
    direct_image_url: str
    license_label: str
    license_type: str
    license_url: str
    license_evidence: str
    creator: str
    attribution_text: str
    retrieved_at: str
    file_path: str
    raw_sha256: str
    sha256: str
    verification_status: str


# ==============================================================================
# DYNAMIC PROMPT BUILDERS (DRIVEN BY USER INPUT)
# ==============================================================================


def build_clip_gate_prompts(user_query: str, custom_positive: Optional[str] = None) -> List[str]:
    """Dynamically builds CLIP zero-shot prompts from the user's runtime query."""
    clean_query = user_query.strip()
    positive_prompt = custom_positive.strip() if custom_positive else f"a clean, real photograph of {clean_query}"
    return [
        positive_prompt,                                           # Index 0: Must win to be ACCEPTED
        f"an image of {clean_query} with a watermark or logo",     # Index 1: Reject watermarked images
        f"a cartoon, 3D render, or AI generated {clean_query}",    # Index 2: Reject synthetic/art
        "unrelated background, blurry noise, or text document",    # Index 3: Reject irrelevant/corrupt
        "graphic violence or nsfw content",                        # Index 4: Safety gate
    ]


def build_tinyfish_agent_prompt(
    user_query: str,
    target_domain: str,
    allowed_licenses: Set[str],
    max_results: int = 10,
) -> str:
    """Dynamically constructs the TinyFish web-scraping prompt from user input."""
    lic_list = ", ".join(sorted(allowed_licenses)).upper()
    return f"""
ROLE & OBJECTIVE:
You are the Safe Image Retrieval Agent for OmniCompiler.
Search for and extract up to {max_results} candidate images matching the user query: "{user_query}"
STRICT DOMAIN BOUNDARY: You may ONLY search and navigate within: https://{target_domain}

DOM INSPECTION & LICENSE CONTRACT:
1. Open or inspect the detail elements for each candidate image matching "{user_query}".
2. Inspect the page DOM to locate the license tag, license deed URL, and creator attribution associated with the specific image asset (not general page footers).
3. Accept ONLY images explicitly licensed under: {lic_list}.
4. Reject any image with missing license tags, "All Rights Reserved", "NC" (Non-Commercial), or "ND" (No Derivatives).
5. If the license is CC-BY or CC-BY-SA, verify that creator/author attribution is present in the DOM.

RETURN ONLY VALID JSON MATCHING THIS SCHEMA:
{{
  "query": "{user_query}",
  "domain": "{target_domain}",
  "candidates": [
    {{
      "source_page_url": "https://{target_domain}/...",
      "direct_image_url": "https://...",
      "license_label": "exact license text from DOM",
      "license_url": "https://creativecommons.org/...",
      "license_evidence": "DOM selector or metadata field used",
      "creator": "author name from DOM",
      "title": "image title or alt text matching {user_query}"
    }}
  ]
}}
""".strip()


# ==============================================================================
# HELPERS: LICENSES & TEXT CLEANUP
# ==============================================================================


def normalize_license(raw: str) -> Optional[str]:
    """Map messy license strings ('CC BY-SA 4.0', 'Public domain', ...) to a canonical key.
    Returns None for anything non-commercial / no-derivatives / unknown."""
    tokens = re.findall(r"[a-z0-9]+", (raw or "").lower())
    if not tokens or "nc" in tokens or "nd" in tokens:
        return None
    if tokens[0] == "cc0" or "zero" in tokens:
        return "cc0"
    if tokens[0] in ("pd", "pdm") or tokens[:2] == ["public", "domain"]:
        return "pd"
    if tokens[0] == "mit":
        return "mit"
    if tokens[0] == "cc" and "by" in tokens:
        return "cc-by-sa" if "sa" in tokens else "cc-by"
    return None


def cc_url_to_label(url: str) -> str:
    """'https://creativecommons.org/licenses/by-sa/2.0/' -> 'CC BY-SA 2.0'"""
    m = re.search(r"/licenses/([a-z\-]+)/([0-9.]+)", url or "")
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else (url or "Unknown")


def clean_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip() or "Unknown"


def singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word


def get_json(url: str, params=None, headers=None) -> dict:
    r = SESSION.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


# ==============================================================================
# SAFETY LAYER 1: URL VALIDATION (HTTPS ONLY, STRICT ALLOWLIST, SSRF GUARD)
# ==============================================================================


def host_matches(hostname: str, allowed: Set[str]) -> bool:
    return any(hostname == h or hostname.endswith("." + h) for h in allowed)


def is_public_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def validate_url(url: str, allowed_hosts: Set[str]) -> bool:
    """Strictly validates HTTPS scheme, explicit domain allowlist, and global IP resolution."""
    if not allowed_hosts:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host:
        return False
    if not host_matches(host, allowed_hosts):
        return False
    return is_public_host(host)


def safe_get(url: str, allowed_hosts: Set[str], max_redirects: int = 3) -> bytes:
    """Download with every redirect hop re-validated against the allowlist and a hard byte cap."""
    for _ in range(max_redirects + 1):
        if not validate_url(url, allowed_hosts):
            raise ValueError(f"blocked URL or redirect hop: {url}")
        with SESSION.get(url, stream=True, allow_redirects=False, timeout=20) as r:
            if r.status_code in (301, 302, 303, 307, 308):
                location = r.headers.get("Location")
                if not location:
                    raise ValueError("redirect without Location header")
                url = urljoin(url, location)
                continue
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            if ctype and not ctype.startswith("image/") and "octet-stream" not in ctype:
                raise ValueError(f"non-image Content-Type: {ctype}")
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf.extend(chunk)
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("file exceeds MAX_DOWNLOAD_BYTES")
            return bytes(buf)
    raise ValueError("too many redirects")


# ==============================================================================
# SAFETY LAYER 2A: FAIL-CLOSED ROBOTS.TXT CHECKER (RFC 9309)
# ==============================================================================


def check_robots_txt(url: str, allowed_hosts: Set[str], user_agent: str = USER_AGENT) -> bool:
    """Fetches and caches robots.txt per domain. Applies a fail-closed policy if unreachable or disallowed."""
    if not validate_url(url, allowed_hosts):
        return False
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if robots_url not in ROBOTS_CACHE:
        rp = RobotFileParser()
        try:
            r = SESSION.get(robots_url, timeout=10)
            if r.status_code in (401, 403) or r.status_code >= 500:
                ROBOTS_CACHE[robots_url] = None  # Fail-closed
            elif r.status_code == 404:
                # Standard RFC 9309: 404 means no robots.txt restrictions published
                rp.parse([])
                ROBOTS_CACHE[robots_url] = rp
            else:
                rp.parse(r.text.splitlines())
                ROBOTS_CACHE[robots_url] = rp
        except Exception:
            ROBOTS_CACHE[robots_url] = None  # Fail-closed on network error

    rp = ROBOTS_CACHE[robots_url]
    if rp is None:
        return False
    bot_token = user_agent.split("/")[0]
    return rp.can_fetch(user_agent, url) and rp.can_fetch(bot_token, url)


# ==============================================================================
# SOURCES
# ==============================================================================


class Source:
    name = "base"
    retrieval_method = "api"
    page_hosts: Set[str] = set()
    image_hosts: Set[str] = set()

    def search(self, query: str) -> Iterator[ImageCandidate]:
        raise NotImplementedError

    def candidate(
        self,
        page_url: str,
        image_url: str,
        label: str,
        creator: Optional[str],
        title: str = "",
        license_url: str = "",
        license_evidence: str = "",
    ) -> ImageCandidate:
        return ImageCandidate(
            source_name=self.name,
            source_page_url=page_url,
            direct_image_url=image_url,
            license_label=label,
            license_url=license_url,
            license_evidence=license_evidence or f"{self.name}_metadata",
            normalized_license=normalize_license(label),
            creator=(creator or "").strip() or "Unknown",
            title=title or "",
        )


class TinyFishSource(Source):
    """Dynamic DOM-scraping source powered by TinyFish Agent API and user's query."""
    name = "tinyfish"
    retrieval_method = "tinyfish_dom_agent"
    page_hosts = {"commons.wikimedia.org", "openverse.org", "flickr.com", "www.flickr.com"}
    image_hosts = {"upload.wikimedia.org", "live.staticflickr.com", "staticflickr.com"}

    def search(self, query: str) -> Iterator[ImageCandidate]:
        if not TINYFISH_API_KEY:
            print("   [tinyfish] TINYFISH_API_KEY not set - skipping TinyFish DOM agent source.")
            return

        target_domains = ["commons.wikimedia.org", "openverse.org"]
        for domain in target_domains:
            dynamic_prompt = build_tinyfish_agent_prompt(
                user_query=query,
                target_domain=domain,
                allowed_licenses=ALLOWED_LICENSES,
                max_results=DEFAULT_TRAIN_COUNT + DEFAULT_VAL_COUNT,
            )
            try:
                r = SESSION.post(
                    TINYFISH_ENDPOINT,
                    headers={"Authorization": f"Bearer {TINYFISH_API_KEY}"},
                    json={
                        "goal": dynamic_prompt,
                        "start_url": f"https://{domain}",
                        "query": query,
                    },
                    timeout=60,
                )
                r.raise_for_status()
                payload = r.json()
                for item in payload.get("candidates", []):
                    page_url = item.get("source_page_url")
                    img_url = item.get("direct_image_url")
                    if page_url and img_url:
                        yield self.candidate(
                            page_url=page_url,
                            image_url=img_url,
                            label=item.get("license_label", "Unknown"),
                            creator=item.get("creator", "Unknown"),
                            title=item.get("title", query),
                            license_url=item.get("license_url", ""),
                            license_evidence=item.get("license_evidence", "DOM license element"),
                        )
            except Exception as e:
                print(f"   [tinyfish] warning on {domain}: {e}")


class WikimediaSource(Source):
    name = "wikimedia"
    retrieval_method = "mediawiki_api"
    page_hosts = {"commons.wikimedia.org", "en.wikipedia.org"}
    image_hosts = {"upload.wikimedia.org"}
    API = "https://commons.wikimedia.org/w/api.php"

    def search(self, query: str) -> Iterator[ImageCandidate]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": 1024,
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|ObjectName",
        }
        for _ in range(MAX_PAGES):
            data = get_json(self.API, params)
            pages = data.get("query", {}).get("pages", {})
            for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                img_url = info.get("thumburl") or info.get("url")
                desc_url = info.get("descriptionurl")
                if not img_url or not desc_url:
                    continue
                meta = info.get("extmetadata", {})
                yield self.candidate(
                    page_url=desc_url,
                    image_url=img_url,
                    label=meta.get("LicenseShortName", {}).get("value", "Unknown"),
                    creator=clean_html(meta.get("Artist", {}).get("value", "")),
                    title=clean_html(meta.get("ObjectName", {}).get("value", "")),
                    license_url=meta.get("LicenseUrl", {}).get("value", ""),
                    license_evidence="extmetadata.LicenseShortName",
                )
            if "continue" not in data:
                return
            params.update(data["continue"])
            time.sleep(0.5)


class OpenverseSource(Source):
    name = "openverse"
    retrieval_method = "openverse_api"
    API = "https://api.openverse.org/v1/images/"
    page_hosts = {
        "flickr.com", "www.flickr.com", "commons.wikimedia.org", "wordpress.org",
        "rawpixel.com", "www.rawpixel.com", "si.edu", "www.si.edu", "europeana.eu",
    }
    image_hosts = {
        "live.staticflickr.com", "staticflickr.com", "upload.wikimedia.org",
        "pd.w.org", "ids.si.edu", "images.rawpixel.com",
    }

    def search(self, query: str) -> Iterator[ImageCandidate]:
        headers = {"Authorization": f"Bearer {OPENVERSE_TOKEN}"} if OPENVERSE_TOKEN else {}
        for page in range(1, MAX_PAGES + 1):
            data = get_json(
                self.API,
                {
                    "q": query,
                    "page": page,
                    "page_size": 20,
                    "license": "cc0,pdm,by,by-sa",
                },
                headers,
            )
            results = data.get("results", [])
            if not results:
                return
            for item in results:
                lic = (item.get("license") or "").lower()
                label = {"cc0": "CC0 1.0", "pdm": "Public Domain Mark 1.0"}.get(
                    lic, f"CC {lic.upper()} {item.get('license_version') or ''}".strip()
                )
                page_url = item.get("foreign_landing_url")
                img_url = item.get("url")
                if page_url and img_url:
                    yield self.candidate(
                        page_url=page_url,
                        image_url=img_url,
                        label=label,
                        creator=item.get("creator"),
                        title=item.get("title") or "",
                        license_url=item.get("license_url") or "",
                        license_evidence="openverse.license_metadata",
                    )
            time.sleep(0.3)


class FlickrSource(Source):
    name = "flickr"
    retrieval_method = "flickr_rest_api"
    image_hosts = {"staticflickr.com", "live.staticflickr.com"}
    page_hosts = {"flickr.com", "www.flickr.com"}
    API = "https://www.flickr.com/services/rest/"
    LICENSES = {
        "4": "CC BY 2.0",
        "5": "CC BY-SA 2.0",
        "9": "CC0 1.0",
        "10": "Public Domain Mark 1.0",
    }

    def search(self, query: str) -> Iterator[ImageCandidate]:
        if not FLICKR_API_KEY:
            print("   [flickr] FLICKR_API_KEY not set - skipping this source.")
            return
        for page in range(1, MAX_PAGES + 1):
            data = get_json(
                self.API,
                {
                    "method": "flickr.photos.search",
                    "api_key": FLICKR_API_KEY,
                    "text": query,
                    "license": ",".join(self.LICENSES),
                    "media": "photos",
                    "content_types": 0,
                    "safe_search": 1,
                    "sort": "relevance",
                    "per_page": 50,
                    "page": page,
                    "extras": "license,owner_name,url_l,url_c",
                    "format": "json",
                    "nojsoncallback": 1,
                },
            )
            if data.get("stat") != "ok":
                raise RuntimeError(f"Flickr API error: {data.get('message')}")
            photos = data["photos"]["photo"]
            if not photos:
                return
            for p in photos:
                img_url = p.get("url_l") or p.get("url_c")
                if not img_url:
                    continue
                yield self.candidate(
                    page_url=f"https://www.flickr.com/photos/{p['owner']}/{p['id']}",
                    image_url=img_url,
                    label=self.LICENSES.get(str(p.get("license")), "Unknown"),
                    creator=p.get("ownername"),
                    title=p.get("title") or "",
                    license_evidence=f"flickr.license_id={p.get('license')}",
                )
            if page >= int(data["photos"].get("pages", 1)):
                return
            time.sleep(0.3)


class OpenImagesSource(Source):
    name = "openimages"
    retrieval_method = "openimages_v7_index"
    page_hosts = {"flickr.com", "www.flickr.com"}
    image_hosts = {"staticflickr.com", "live.staticflickr.com", "flickr.com"}

    @staticmethod
    def _cached(url: str) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, url.rsplit("/", 1)[-1])
        if not os.path.exists(path):
            print(f"   [openimages] downloading index {os.path.basename(path)} (one-time)...")
            with SESSION.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".part", path)
        return path

    def _find_class(self, query: str) -> Optional[str]:
        wanted = {query.lower(), singular(query.lower())}
        with open(self._cached(OI_CLASSES_URL), newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].startswith("/") and row[1].lower() in wanted:
                    return row[0]
        return None

    def search(self, query: str) -> Iterator[ImageCandidate]:
        mid = self._find_class(query)
        if not mid:
            print(f"   [openimages] no Open Images class named '{query}' - skipping.")
            return
        ids = set()
        with open(self._cached(OI_LABELS_URL), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["LabelName"] == mid and row["Confidence"] == "1":
                    ids.add(row["ImageID"])
        with open(self._cached(OI_IMAGES_URL), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["ImageID"] in ids and row.get("OriginalURL"):
                    lic_url = row.get("License", "")
                    yield self.candidate(
                        page_url=row.get("OriginalLandingURL") or row["OriginalURL"],
                        image_url=row["OriginalURL"],
                        label=cc_url_to_label(lic_url),
                        creator=row.get("Author"),
                        title=row.get("Title") or "",
                        license_url=lic_url,
                        license_evidence="openimages_csv.License",
                    )


class HuggingFaceSource(Source):
    name = "huggingface"
    retrieval_method = "hf_datasets_server"
    page_hosts = {"huggingface.co"}
    image_hosts = {"huggingface.co", "hf.co", "datasets-server.huggingface.co"}
    HUB = "https://huggingface.co/api/datasets"
    DS_SERVER = "https://datasets-server.huggingface.co"
    MAX_ROWS_PER_DATASET = 2000

    @staticmethod
    def _license(ds: dict) -> str:
        lic = (ds.get("cardData") or {}).get("license")
        if isinstance(lic, list):
            lic = lic[0] if lic else None
        if not lic:
            lic = next((t.split(":", 1)[1] for t in ds.get("tags", []) if t.startswith("license:")), "")
        return str(lic or "").lower()

    def _datasets(self, query: str, headers: dict):
        seen = set()
        listings = [{"search": query, "filter": ["task_categories:image-classification"]}]
        for lic in HF_ALLOWED_LICENSES:
            listings.append(
                {
                    "filter": ["task_categories:image-classification", f"license:{lic}"],
                    "sort": "downloads",
                    "direction": -1,
                }
            )
        for extra in listings:
            for ds in get_json(self.HUB, {**extra, "limit": 20, "full": "true"}, headers):
                if ds["id"] not in seen and self._license(ds) in HF_ALLOWED_LICENSES:
                    seen.add(ds["id"])
                    yield ds

    def search(self, query: str) -> Iterator[ImageCandidate]:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        targets = [re.compile(rf"\b{re.escape(t)}\b") for t in {query.lower(), singular(query.lower())}]
        for ds in self._datasets(query, headers):
            ds_id, lic = ds["id"], self._license(ds)
            try:
                splits = get_json(f"{self.DS_SERVER}/splits", {"dataset": ds_id}, headers).get("splits", [])
                chosen = next((s for s in splits if s.get("split") == "train"), splits[0] if splits else None)
                if not chosen:
                    continue
                for offset in range(0, self.MAX_ROWS_PER_DATASET, 100):
                    data = get_json(
                        f"{self.DS_SERVER}/rows",
                        {
                            "dataset": ds_id,
                            "config": chosen["config"],
                            "split": chosen["split"],
                            "offset": offset,
                            "length": 100,
                        },
                        headers,
                    )
                    feats = data.get("features", [])
                    img_col = next((f["name"] for f in feats if f["type"].get("_type") == "Image"), None)
                    lab = next((f for f in feats if f["type"].get("_type") == "ClassLabel"), None)
                    if not img_col or not lab:
                        break
                    names = lab["type"]["names"]
                    rows = data.get("rows", [])
                    for r in rows:
                        idx = r["row"].get(lab["name"])
                        if not isinstance(idx, int) or not 0 <= idx < len(names):
                            continue
                        cls = names[idx].lower().replace("_", " ")
                        src = (r["row"].get(img_col) or {}).get("src")
                        if src and any(t.search(cls) for t in targets):
                            yield self.candidate(
                                page_url=f"https://huggingface.co/datasets/{ds_id}",
                                image_url=src,
                                label=lic,
                                creator=ds_id.split("/")[0],
                                title=f"{ds_id} / {cls}",
                                license_evidence=f"hf_card.license={lic}",
                            )
                    if len(rows) < 100:
                        break
            except Exception as e:
                print(f"   [huggingface] skipping {ds_id}: {e}")


SOURCES: Dict[str, Source] = {
    s.name: s
    for s in (
        TinyFishSource(),
        OpenverseSource(),
        WikimediaSource(),
        FlickrSource(),
        OpenImagesSource(),
        HuggingFaceSource(),
    )
}


def interleave(gens: Dict[str, Iterator[ImageCandidate]]) -> Iterator[ImageCandidate]:
    """Round-robin across active sources; drops exhausted or failing sources gracefully."""
    active = dict(gens)
    while active:
        for name in list(active):
            try:
                cand = next(active[name])
            except StopIteration:
                print(f"   [{name}] exhausted")
                del active[name]
                continue
            except Exception as e:
                print(f"   [{name}] source error, disabling it: {e}")
                del active[name]
                continue
            yield cand


# ==============================================================================
# SAFETY LAYERS 3 & 4: IMAGE DECODING, CLIP QUALITY GATE & INTEGRITY VERIFIER
# ==============================================================================


def decode_image(data: bytes) -> Image.Image:
    """LAYER 3: Verify structure on the RAW downloaded bytes before decoding to RGB."""
    Image.open(io.BytesIO(data)).verify()
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def clip_gate(model, processor, device: str, img: Image.Image, gate_prompts: List[str]) -> Tuple[bool, List[float]]:
    """LAYER 4: Accept only if the user's dynamic positive prompt (index 0) wins."""
    inputs = processor(text=gate_prompts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    return torch.argmax(probs).item() == 0, [round(p, 2) for p in probs.tolist()]


def verify_dataset_integrity(manifest_dir: str) -> None:
    """Recalculates SHA-256 of every stored image file and compares against its JSON manifest."""
    print(f"\n🔍 [OmniCompiler] Running Cryptographic Provenance Audit in '{manifest_dir}'...")
    if not os.path.exists(manifest_dir):
        print("   ⚠️ No manifests directory found yet. Run a retrieval query first.")
        return

    passed = failed = 0
    for fname in sorted(os.listdir(manifest_dir)):
        if not fname.endswith(".json"):
            continue
        mpath = os.path.join(manifest_dir, fname)
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                manifest = ProvenanceManifest.model_validate_json(f.read())
        except Exception as e:
            print(f"   ❌ INVALID MANIFEST SCHEMA: {fname} ({e})")
            failed += 1
            continue

        if not os.path.exists(manifest.file_path):
            print(f"   ❌ MISSING FILE: {manifest.asset_id} expected at {manifest.file_path}")
            failed += 1
            continue

        digest = hashlib.sha256()
        with open(manifest.file_path, "rb") as img_f:
            for chunk in iter(lambda: img_f.read(8192), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()

        if actual_hash == manifest.sha256:
            print(f"   ✅ VERIFIED: {manifest.asset_id} | {manifest.license_type.upper()} | sha256={actual_hash[:12]}...")
            passed += 1
        else:
            print(
                f"   ❌ TAMPER DETECTED: {manifest.asset_id} | Expected {manifest.sha256[:12]}..., got {actual_hash[:12]}..."
            )
            failed += 1

    print(f"\n==========================================")
    print(f"PROVENANCE AUDIT SUMMARY: {passed} Verified | {failed} Failed")
    print(f"==========================================\n")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def run_curation() -> None:
    parser = argparse.ArgumentParser(description="OmniCompiler Safe Image Retrieval Pipeline")
    parser.add_argument("-q", "--query", default=None, help="Target search query (if omitted, prompts interactively)")
    parser.add_argument("--clip-prompt", default=None, help="Optional custom CLIP positive prompt override")
    parser.add_argument("--sources", default=",".join(SOURCES), help=f"Comma list of: {', '.join(SOURCES)}")
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN_COUNT, help="Number of training images (default: 8)")
    parser.add_argument("--val", type=int, default=DEFAULT_VAL_COUNT, help="Number of validation images (default: 2)")
    parser.add_argument("--verify", action="store_true", help="Run SHA-256 cryptographic integrity check on existing dataset")
    args = parser.parse_args()

    if args.verify:
        verify_dataset_integrity(MANIFEST_DIR)
        return

    # Take input dynamically from CLI or prompt user interactively at runtime
    query = (args.query or "").strip()
    if not query:
        query = input("Enter image search query for OmniCompiler (e.g., 'sneaker in low light'): ").strip()
    if not query:
        raise SystemExit("Error: A search query is required.")

    chosen = [s.strip() for s in args.sources.split(",") if s.strip() in SOURCES]
    if not chosen:
        raise SystemExit(f"No valid sources selected. Choose from: {', '.join(SOURCES)}")

    gate_prompts = build_clip_gate_prompts(query, args.clip_prompt)

    label_dir = os.path.join(TRAIN_DIR, re.sub(r"[^a-zA-Z0-9_\-]+", "_", query.lower()))
    for d in (label_dir, VAL_DIR, MANIFEST_DIR):
        os.makedirs(d, exist_ok=True)

    total_target = args.train + args.val
    print("\n🚀 [OmniCompiler] Loading CLIP zero-shot quality gate...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"\n🎯 [OmniCompiler] Dynamic User Query: '{query}'")
    print(f"🎯 [OmniCompiler] Target Split: {total_target} Total Images ({args.train} Train + {args.val} Validation)")
    print(f"🎯 [OmniCompiler] Dynamic CLIP Positive Gate: '{gate_prompts[0]}'")
    print(f"🌐 [OmniCompiler] Active Whitelisted Sources: {', '.join(chosen)}")

    gens = {n: SOURCES[n].search(query) for n in chosen}

    train_count = val_count = 0
    stats, accepted_by = Counter(), Counter()
    seen_urls: Set[str] = set()
    seen_hashes: Set[str] = set()

    def reject(reason: str, cand: ImageCandidate, detail: str = "") -> None:
        stats[reason] += 1
        print(f"   ❌ [{cand.source_name}] {reason} {detail}".rstrip())

    for cand in interleave(gens):
        if train_count >= args.train and val_count >= args.val:
            break
        if cand.direct_image_url in seen_urls:
            stats["duplicate_url"] += 1
            continue
        seen_urls.add(cand.direct_image_url)
        src = SOURCES[cand.source_name]
        print(f"\n--- [{cand.source_name}] {(cand.title or cand.direct_image_url)[:65]} ---")

        # LAYER 1: Domain Whitelist & SSRF Protection
        if not (
            validate_url(cand.source_page_url, src.page_hosts)
            and validate_url(cand.direct_image_url, src.image_hosts)
        ):
            reject("layer1_url_or_domain_blocked", cand)
            continue

        # LAYER 2A: Fail-Closed robots.txt Check
        if not check_robots_txt(cand.direct_image_url, src.image_hosts):
            reject("layer2_robots_txt_disallowed", cand)
            continue

        # LAYER 2B: License & Attribution Contract
        if cand.normalized_license not in ALLOWED_LICENSES:
            reject("layer2_unapproved_license", cand, f"('{cand.license_label}')")
            continue
        if cand.normalized_license in ATTRIBUTION_REQUIRED_LICENSES and cand.creator in ("", "Unknown"):
            reject("layer2_missing_attribution", cand, f"({cand.license_label} requires creator)")
            continue

        # LAYER 3: Safe Download, Raw SHA-256 & Image Structure Verification
        try:
            raw_bytes = safe_get(cand.direct_image_url, src.image_hosts)
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            img = decode_image(raw_bytes)
        except Exception as e:
            reject("layer3_download_or_decode", cand, f"({e})")
            continue
        finally:
            time.sleep(DOWNLOAD_DELAY)

        if min(img.size) < MIN_SIDE:
            reject("layer3_too_small", cand, f"{img.size}")
            continue

        # LAYER 4: Dynamic CLIP Quality & Relevance Gate
        ok, probs = clip_gate(clip_model, clip_processor, device, img, gate_prompts)
        if not ok:
            reject("layer4_clip_gate", cand, f"probs={probs}")
            continue

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        jpeg_bytes = buf.getvalue()
        stored_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

        if stored_sha256 in seen_hashes or raw_sha256 in seen_hashes:
            reject("duplicate_image_hash", cand)
            continue
        seen_hashes.add(stored_sha256)
        seen_hashes.add(raw_sha256)

        asset_id = f"img_{uuid.uuid4().hex[:8]}"
        is_train = train_count < args.train
        file_path = os.path.join(
            label_dir if is_train else VAL_DIR,
            f"{'train' if is_train else 'val'}_{asset_id}.jpg",
        )
        with open(file_path, "wb") as f:
            f.write(jpeg_bytes)

        attribution_text = (
            f"'{cand.title or query}' by {cand.creator}, licensed under {cand.license_label} ({cand.source_page_url})"
        )

        manifest = ProvenanceManifest(
            asset_id=asset_id,
            policy_version=POLICY_VERSION,
            source_name=cand.source_name,
            retrieval_method=src.retrieval_method,
            query=query,
            clip_target_prompt=gate_prompts[0],
            source_page_url=cand.source_page_url,
            direct_image_url=cand.direct_image_url,
            license_label=cand.license_label,
            license_type=cand.normalized_license,
            license_url=cand.license_url,
            license_evidence=cand.license_evidence,
            creator=cand.creator,
            attribution_text=attribution_text,
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            file_path=file_path,
            raw_sha256=raw_sha256,
            sha256=stored_sha256,
            verification_status="ACCEPTED",
        )

        manifest_path = os.path.join(MANIFEST_DIR, f"manifest_{asset_id}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=4))

        accepted_by[cand.source_name] += 1
        if is_train:
            train_count += 1
            print(f"   ✅ [TRAIN] {train_count}/{args.train} | {cand.license_label} | sha256={stored_sha256[:10]}")
        else:
            val_count += 1
            print(f"   ✅ [VAL] {val_count}/{args.val} | {cand.license_label} | sha256={stored_sha256[:10]}")

    complete = train_count >= args.train and val_count >= args.val
    print("\n==========================================")
    print("STAGE 2: CURATION " + ("COMPLETE." if complete else "INCOMPLETE - sources ran dry or errored."))
    print(f"Dataset path: {os.path.abspath(DATASET_DIR)}")
    print(f"Total Collected: {train_count + val_count}/{total_target} (Train: {train_count}/{args.train} | Val: {val_count}/{args.val})")
    print(f"Accepted per source: {dict(accepted_by) or 'none'}")
    print(f"Rejections by layer: {dict(stats) or 'none'}")
    print("==========================================\n")


if __name__ == "__main__":
    run_curation()