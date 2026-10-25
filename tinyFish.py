"""
OmniCompiler - Multi-Source, License-Aware Safe Image Retrieval Pipeline.

Supports both Round 1 (Initial Curation) and Round 2 (BLIP/KNN Targeted Healing Curation)
while preserving seen_urls and seen_hashes across rounds so duplicate images are never re-fetched.
"""

import argparse
import csv
import datetime
import hashlib
import html
import io
import ipaddress
import json
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

DEFAULT_TRAIN_COUNT = 8
DEFAULT_VAL_COUNT = 2

TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY")
TINYFISH_ENDPOINT = os.getenv("TINYFISH_ENDPOINT", "https://agent.tinyfish.ai/v1/automation/run")
FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENVERSE_TOKEN = os.getenv("OPENVERSE_TOKEN")

ALLOWED_LICENSES = {"cc0", "pd", "cc-by", "cc-by-sa", "mit"}
ATTRIBUTION_REQUIRED_LICENSES = {"cc-by", "cc-by-sa"}
HF_ALLOWED_LICENSES = {"cc0-1.0", "mit", "apache-2.0", "cc-by-4.0"}

DATASET_DIR = "omni_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val", "images")
MANIFEST_DIR = os.path.join(DATASET_DIR, "manifests")
CACHE_DIR = os.path.join(DATASET_DIR, ".cache")

MAX_PAGES = 10
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MIN_SIDE = 128
DOWNLOAD_DELAY = 0.15
Image.MAX_IMAGE_PIXELS = 50_000_000

ROBOTS_EXEMPT_API_HOSTS = {
    "datasets-server.huggingface.co",
    "cdn-lfs.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cas-bridge.xethub.hf.co",
    "api.openverse.org",
}

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
# HELPERS: LICENSES, URL NORMALIZATION & TEXT CLEANUP
# ==============================================================================


def ensure_https(url: str) -> str:
    if not url:
        return ""
    u = url.strip()
    if u.startswith("http://"):
        return "https://" + u[len("http://"):]
    return u


def normalize_license(raw: str) -> Optional[str]:
    tokens = re.findall(r"[a-z0-9]+", (raw or "").lower())
    if not tokens or "nc" in tokens or "nd" in tokens:
        return None
    if tokens[0] == "cc0" or "zero" in tokens:
        return "cc0"
    if tokens[0] in ("pd", "pdm") or tokens[:2] == ["public", "domain"]:
        return "pd"
    if tokens[0] in ("mit", "apache"):
        return "mit"
    if tokens[0] == "cc" and "by" in tokens:
        return "cc-by-sa" if "sa" in tokens else "cc-by"
    if tokens[0] == "by":
        return "cc-by-sa" if "sa" in tokens else "cc-by"
    return None


def cc_url_to_label(url: str) -> str:
    if not url:
        return "Unknown"
    if "/publicdomain/zero/" in url:
        return "CC0 1.0"
    if "/publicdomain/mark/" in url:
        return "Public Domain Mark 1.0"
    m = re.search(r"/licenses/([a-z\-]+)/([0-9.]+)", url)
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else url


def clean_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip() or "Unknown"


def singular(word: str) -> str:
    w = word.strip()
    if w.lower().endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.lower().endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if w.lower().endswith("es") and any(w.lower().endswith(s) for s in ("ches", "shes", "xes", "zes", "sses")):
        return w[:-2]
    return w[:-1] if w.lower().endswith("s") and not w.lower().endswith("ss") and len(w) > 3 else w


def get_json(url: str, params=None, headers=None) -> dict:
    r = SESSION.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


# ==============================================================================
# DYNAMIC PROMPT BUILDERS & QUERY PARSERS
# ==============================================================================

SCENE_MODIFIER_PATTERN = re.compile(
    r"\b(?:in\s+[a-z0-9\s]+|at\s+(?:night|dawn|dusk)|underwater|aerial(?:\s+view)?|"
    r"low\s+light|close[\s\-]?up|macro|outdoor|indoor|blurry|top[\s\-]?down|satellite)\b",
    flags=re.IGNORECASE,
)


def split_compound_terms(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(
        r"\s*(?:,|\band\b|\bvs\.?\b|\bversus\b|\bor\b|/|\+|&)\s*",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    seen = set()
    result = []
    for p in parts:
        clean = p.strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            result.append(clean)
    return result


def normalize_labels(labels: Optional[List[str]], dataset_query: Optional[str]) -> List[str]:
    expanded: List[str] = []
    seen = set()
    raw_items = labels if labels else ([dataset_query] if dataset_query else [])
    for item in raw_items:
        for sub in split_compound_terms(item or ""):
            key = sub.lower()
            if key not in seen:
                seen.add(key)
                expanded.append(sub)
    return expanded


def build_class_search_query(label: str, dataset_query: Optional[str]) -> str:
    clean_label = label.strip()
    if not dataset_query:
        return clean_label

    modifiers = SCENE_MODIFIER_PATTERN.findall(dataset_query)
    if modifiers:
        mod_str = " ".join(m.strip() for m in modifiers if m.strip().lower() not in clean_label.lower())
        if mod_str:
            return f"{clean_label} {mod_str}".strip()

    return clean_label


def build_clip_gate_prompts(class_label: str, custom_positive: Optional[str] = None) -> List[str]:
    """Builds CLIP zero-shot gate prompts anchored on the clean class identity."""
    clean_query = class_label.strip()
    sing_query = singular(clean_query)
    if custom_positive:
        positive_prompt = custom_positive.strip()
    elif sing_query.lower() != clean_query.lower():
        positive_prompt = f"a photo of a {sing_query}, {clean_query}"
    else:
        positive_prompt = f"a photo of a {clean_query}"

    return [
        positive_prompt,
        "stock photo overlaid with a large text watermark or logo",
        "a cartoon drawing, vector clip art, diagram, or 3D render",
        "unrelated empty background, blurry noise, or text document",
        "graphic violence, gore, or nsfw adult content",
    ]


def build_tinyfish_agent_prompt(
    user_query: str,
    target_domain: str,
    allowed_licenses: Set[str],
    max_results: int = 10,
) -> str:
    lic_list = ", ".join(sorted(allowed_licenses)).upper()
    return f"""
ROLE & OBJECTIVE:
You are the Safe Image Retrieval Agent for OmniCompiler.
Search for and extract up to {max_results} candidate images matching the user query: "{user_query}"
STRICT DOMAIN BOUNDARY: You may ONLY search and navigate within: https://{target_domain}

DOM INSPECTION & LICENSE CONTRACT:
1. Open or inspect the detail elements for each candidate image matching "{user_query}".
2. Inspect the page DOM to locate the license tag, license deed URL, and creator attribution associated with the specific image asset.
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
# SAFETY LAYER 1 & 2: URL VALIDATION & ROBOTS.TXT
# ==============================================================================


def host_matches(hostname: str, allowed: Set[str]) -> bool:
    return any(hostname == h or hostname.endswith("." + h) for h in allowed)


def is_public_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    ips = [ipaddress.ip_address(i[4][0]) for i in infos if i and i[4]]
    if not ips:
        return False
    for ip in ips:
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return False
    return True


def validate_url(url: str, allowed_hosts: Set[str]) -> bool:
    if not allowed_hosts or not url:
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


def safe_get(url: str, allowed_hosts: Set[str], max_redirects: int = 4) -> bytes:
    url = ensure_https(url)
    for _ in range(max_redirects + 1):
        if not validate_url(url, allowed_hosts):
            raise ValueError(f"blocked URL or redirect hop: {url}")
        with SESSION.get(url, stream=True, allow_redirects=False, timeout=20) as r:
            if r.status_code in (301, 302, 303, 307, 308):
                location = r.headers.get("Location")
                if not location:
                    raise ValueError("redirect without Location header")
                url = ensure_https(urljoin(url, location))
                continue
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            if ctype and not ctype.startswith("image/") and "octet-stream" not in ctype and "binary" not in ctype:
                raise ValueError(f"non-image Content-Type: {ctype}")
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf.extend(chunk)
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("file exceeds MAX_DOWNLOAD_BYTES")
            return bytes(buf)
    raise ValueError("too many redirects")


def check_robots_txt(url: str, allowed_hosts: Set[str], user_agent: str = USER_AGENT) -> bool:
    url = ensure_https(url)
    if not validate_url(url, allowed_hosts):
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host_matches(host, ROBOTS_EXEMPT_API_HOSTS):
        return True

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if robots_url not in ROBOTS_CACHE:
        rp = RobotFileParser()
        try:
            r = SESSION.get(robots_url, timeout=10)
            if r.status_code in (401, 403) or r.status_code >= 500:
                ROBOTS_CACHE[robots_url] = None
            elif r.status_code == 404:
                rp.parse([])
                ROBOTS_CACHE[robots_url] = rp
            else:
                rp.parse(r.text.splitlines())
                ROBOTS_CACHE[robots_url] = rp
        except Exception:
            ROBOTS_CACHE[robots_url] = None

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
            source_page_url=ensure_https(page_url),
            direct_image_url=ensure_https(image_url),
            license_label=label,
            license_url=ensure_https(license_url),
            license_evidence=license_evidence or f"{self.name}_metadata",
            normalized_license=normalize_license(label),
            creator=(creator or "").strip() or "Unknown",
            title=title or "",
        )


class TinyFishSource(Source):
    name = "tinyfish"
    retrieval_method = "tinyfish_dom_agent"
    page_hosts = {"commons.wikimedia.org", "openverse.org", "flickr.com", "www.flickr.com"}
    image_hosts = {"upload.wikimedia.org", "live.staticflickr.com", "staticflickr.com"}

    def search(self, query: str) -> Iterator[ImageCandidate]:
        if not TINYFISH_API_KEY:
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
                    headers={
                        "X-API-Key": TINYFISH_API_KEY,
                        "Authorization": f"Bearer {TINYFISH_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"url": f"https://{domain}", "goal": dynamic_prompt},
                    timeout=45,
                )
                r.raise_for_status()
                payload = r.json()
                result_data = payload.get("result", payload) if isinstance(payload, dict) else {}
                if isinstance(result_data, str):
                    try:
                        result_data = json.loads(result_data)
                    except Exception:
                        result_data = {}
                candidates = result_data.get("candidates", []) if isinstance(result_data, dict) else []
                for item in candidates:
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
    page_hosts = {"commons.wikimedia.org", "en.wikipedia.org", "wikimedia.org"}
    image_hosts = {"upload.wikimedia.org"}
    API = "https://commons.wikimedia.org/w/api.php"

    def search(self, query: str) -> Iterator[ImageCandidate]:
        sing = singular(query)
        search_terms = [f"{query} filetype:bitmap", f"{sing} photo filetype:bitmap"]
        seen_pages: Set[int] = set()

        for gsr in search_terms:
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": gsr,
                "gsrlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": 800,
                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|ObjectName",
            }
            for _ in range(3):
                try:
                    data = get_json(self.API, params)
                except Exception:
                    break
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    break
                for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
                    pid = page.get("pageid")
                    if pid in seen_pages:
                        continue
                    seen_pages.add(pid)
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
                    break
                params.update(data["continue"])
                time.sleep(0.3)


class OpenverseSource(Source):
    name = "openverse"
    retrieval_method = "openverse_api"
    API = "https://api.openverse.org/v1/images/"
    page_hosts = {
        "flickr.com", "www.flickr.com", "commons.wikimedia.org", "wordpress.org",
        "rawpixel.com", "www.rawpixel.com", "si.edu", "www.si.edu", "europeana.eu",
        "openverse.org", "pxhere.com", "stocksnap.io", "inaturalist.org",
        "www.inaturalist.org", "nappy.co", "picryl.com", "burst.shopify.com",
    }
    image_hosts = {
        "live.staticflickr.com", "staticflickr.com", "upload.wikimedia.org",
        "pd.w.org", "ids.si.edu", "images.rawpixel.com", "api.openverse.org",
        "c.pxhere.com", "cdn.stocksnap.io", "static.inaturalist.org",
        "inaturalist-open-data.s3.amazonaws.com", "burst.shopifycdn.com",
    }

    def search(self, query: str) -> Iterator[ImageCandidate]:
        headers = {"Authorization": f"Bearer {OPENVERSE_TOKEN}"} if OPENVERSE_TOKEN else {}
        q_term = singular(query) if len(query.split()) == 1 else query
        for page in range(1, MAX_PAGES + 1):
            try:
                data = get_json(
                    self.API,
                    {
                        "q": q_term,
                        "page": page,
                        "page_size": 20,
                        "license": "cc0,pdm,by,by-sa",
                        "extension": "jpg,png",
                    },
                    headers,
                )
            except Exception:
                return
            results = data.get("results", [])
            if not results:
                return
            for item in results:
                lic = (item.get("license") or "").lower()
                label = {"cc0": "CC0 1.0", "pdm": "Public Domain Mark 1.0"}.get(
                    lic, f"CC {lic.upper()} {item.get('license_version') or ''}".strip()
                )
                page_url = item.get("foreign_landing_url")
                img_url = item.get("url") or item.get("thumbnail")
                if page_url and img_url:
                    yield self.candidate(
                        page_url=page_url,
                        image_url=img_url,
                        label=label,
                        creator=item.get("creator") or item.get("source") or "Openverse Contributor",
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
                return
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
    page_hosts = {"flickr.com", "www.flickr.com", "staticflickr.com", "live.staticflickr.com"}
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
        q_clean = query.strip().lower()
        wanted = {q_clean, singular(q_clean)}
        with open(self._cached(OI_CLASSES_URL), newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].startswith("/") and row[1].strip().lower() in wanted:
                    return row[0]
        return None

    def search(self, query: str) -> Iterator[ImageCandidate]:
        mid = self._find_class(query)
        if not mid:
            return
        ids = set()
        with open(self._cached(OI_LABELS_URL), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["LabelName"] == mid and row["Confidence"] == "1":
                    ids.add(row["ImageID"])
        with open(self._cached(OI_IMAGES_URL), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["ImageID"] in ids:
                    img_url = row.get("Thumbnail300KURL") or row.get("OriginalURL")
                    if not img_url:
                        continue
                    lic_url = row.get("License", "")
                    author = (row.get("Author") or "").strip() or "OpenImages Flickr Contributor"
                    yield self.candidate(
                        page_url=row.get("OriginalLandingURL") or img_url,
                        image_url=img_url,
                        label=cc_url_to_label(lic_url),
                        creator=author,
                        title=row.get("Title") or query,
                        license_url=lic_url,
                        license_evidence="openimages_csv.License",
                    )


class HuggingFaceSource(Source):
    name = "huggingface"
    retrieval_method = "hf_datasets_server"
    page_hosts = {"huggingface.co"}
    image_hosts = {
        "huggingface.co",
        "hf.co",
        "datasets-server.huggingface.co",
        "cdn-lfs.hf.co",
        "cdn-lfs-us-1.hf.co",
        "cas-bridge.xethub.hf.co",
    }
    HUB = "https://huggingface.co/api/datasets"
    DS_SERVER = "https://datasets-server.huggingface.co"
    MAX_ROWS_PER_DATASET = 300

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
        params = {
            "search": singular(query),
            "filter": ["task_categories:image-classification"],
            "sort": "downloads",
            "direction": -1,
            "limit": 15,
            "full": "true",
        }
        try:
            items = get_json(self.HUB, params, headers)
        except Exception:
            return
        for ds in items:
            if ds["id"] not in seen and self._license(ds) in HF_ALLOWED_LICENSES:
                seen.add(ds["id"])
                yield ds

    def search(self, query: str) -> Iterator[ImageCandidate]:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        base_terms = {query.lower(), singular(query.lower())}
        targets = [re.compile(rf"\b{re.escape(t)}\b") for t in base_terms if t]

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
                    img_col = next((f["name"] for f in feats if isinstance(f.get("type"), dict) and f["type"].get("_type") == "Image"), None)
                    lab = next((f for f in feats if isinstance(f.get("type"), dict) and f["type"].get("_type") == "ClassLabel"), None)
                    if not img_col or not lab:
                        break
                    names = lab["type"].get("names", [])
                    if not any(any(t.search(n.lower().replace("_", " ")) for t in targets) for n in names):
                        break

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
            except Exception:
                continue


SOURCES: Dict[str, Source] = {
    s.name: s
    for s in (
        OpenImagesSource(),
        OpenverseSource(),
        WikimediaSource(),
        HuggingFaceSource(),
        FlickrSource(),
        TinyFishSource(),
    )
}


def interleave(gens: Dict[str, Iterator[ImageCandidate]], disabled_sources: Optional[Set[str]] = None) -> Iterator[ImageCandidate]:
    active = dict(gens)
    while active:
        for name in list(active):
            if disabled_sources and name in disabled_sources:
                del active[name]
                continue
            try:
                cand = next(active[name])
            except StopIteration:
                del active[name]
                continue
            except Exception:
                del active[name]
                continue
            yield cand


# ==============================================================================
# SAFETY LAYERS 3 & 4: IMAGE DECODING, CLIP QUALITY GATE & INTEGRITY VERIFIER
# ==============================================================================


def decode_image(data: bytes) -> Image.Image:
    Image.open(io.BytesIO(data)).verify()
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def clip_gate(model, processor, device: str, img: Image.Image, gate_prompts: List[str]) -> Tuple[bool, List[float]]:
    inputs = processor(text=gate_prompts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    prob_list = [round(p, 2) for p in probs.tolist()]
    return torch.argmax(probs).item() == 0, prob_list


def verify_dataset_integrity(manifest_dir: str) -> None:
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
        except Exception:
            failed += 1
            continue

        if not os.path.exists(manifest.file_path):
            failed += 1
            continue

        digest = hashlib.sha256()
        with open(manifest.file_path, "rb") as img_f:
            for chunk in iter(lambda: img_f.read(8192), b""):
                digest.update(chunk)
        if digest.hexdigest() == manifest.sha256:
            passed += 1
        else:
            failed += 1

    print(f"\nPROVENANCE AUDIT SUMMARY: {passed} Verified | {failed} Failed\n")


# ==============================================================================
# OMNICOMPILER ORCHESTRATOR EXPORT FUNCTIONS
# ==============================================================================


def load_clip_gate():
    print("\n🚀 [OmniCompiler] Loading CLIP zero-shot quality gate...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor, device


MAX_CANDIDATES_PER_CLASS = 150
MAX_SOURCE_STREAK_FAILS = 30


def run_curation(
    labels,
    dataset_query,
    target_train,
    target_val,
    sources,
    clip,
    custom_search_queries: Optional[Dict[str, str]] = None,
    per_class_targets: Optional[Dict[str, Tuple[int, int]]] = None,
    seen_urls: Optional[Set[str]] = None,
    seen_hashes: Optional[Set[str]] = None,
):
    """
    Multi-class curation pipeline supporting both Round 1 (initial) and Round 2 (healing).
    - Always records `class_idx` on every sample for `maths_unified.py`.
    - Preserves `seen_urls` and `seen_hashes` across rounds so healing never downloads duplicates.
    """
    clip_model, clip_processor, device = clip

    resolved_labels = normalize_labels(labels, dataset_query)
    if not resolved_labels:
        raise ValueError("No valid class labels or search query provided.")

    if seen_urls is None:
        seen_urls = set()
    if seen_hashes is None:
        seen_hashes = set()

    curation = {
        "labels": resolved_labels,
        "classes": resolved_labels,
        "search_queries": {},
        "counts": {},
        "rejections": {},
        "samples": [],
        "seen_urls": seen_urls,
        "seen_hashes": seen_hashes,
    }

    for d in (TRAIN_DIR, VAL_DIR, MANIFEST_DIR):
        os.makedirs(d, exist_ok=True)

    for class_idx, label in enumerate(resolved_labels):
        t_train, t_val = (
            per_class_targets[label]
            if (per_class_targets and label in per_class_targets)
            else (target_train, target_val)
        )
        if t_train + t_val <= 0:
            curation["counts"][label] = {"train": 0, "val": 0}
            continue

        clean_class_term = label.strip()
        if custom_search_queries and label in custom_search_queries:
            query = custom_search_queries[label].strip()
        else:
            query = build_class_search_query(label, dataset_query)

        curation["search_queries"][label] = query
        curation["counts"][label] = {"train": 0, "val": 0}
        reject_stats: Counter = Counter()

        # CLIP gate always verifies the clean class identity (e.g. "a photo of a cat")
        gate_prompts = build_clip_gate_prompts(clean_class_term)
        label_dir = os.path.join(TRAIN_DIR, re.sub(r"[^a-zA-Z0-9_\-]+", "_", clean_class_term.lower()))
        os.makedirs(label_dir, exist_ok=True)

        gens = {}
        for n in sources:
            if n not in SOURCES:
                continue
            src_query = clean_class_term if n in ("openimages", "huggingface") else query
            gens[n] = SOURCES[n].search(src_query)

        train_count = val_count = 0
        inspected = 0
        source_fail_streak: Dict[str, int] = Counter()
        disabled_sources: Set[str] = set()

        print(f"\n🎯 [OmniCompiler] Curating Class '{label}' (Query: '{query}' | Target: {t_train} Train + {t_val} Val)")

        for cand in interleave(gens, disabled_sources):
            if train_count >= t_train and val_count >= t_val:
                break
            if inspected >= MAX_CANDIDATES_PER_CLASS:
                break

            src_name = cand.source_name
            if source_fail_streak[src_name] >= MAX_SOURCE_STREAK_FAILS:
                disabled_sources.add(src_name)
                continue

            if cand.direct_image_url in seen_urls:
                continue
            seen_urls.add(cand.direct_image_url)
            inspected += 1

            src = SOURCES[src_name]

            if not (validate_url(cand.source_page_url, src.page_hosts) and validate_url(cand.direct_image_url, src.image_hosts)):
                reject_stats["layer1_domain_or_https"] += 1
                source_fail_streak[src_name] += 1
                continue

            if not check_robots_txt(cand.direct_image_url, src.image_hosts):
                reject_stats["layer2_robots_txt"] += 1
                source_fail_streak[src_name] += 1
                continue

            if cand.normalized_license not in ALLOWED_LICENSES:
                reject_stats["layer2_license"] += 1
                source_fail_streak[src_name] += 1
                continue
            if cand.normalized_license in ATTRIBUTION_REQUIRED_LICENSES and cand.creator in ("", "Unknown"):
                reject_stats["layer2_attribution"] += 1
                source_fail_streak[src_name] += 1
                continue

            try:
                raw_bytes = safe_get(cand.direct_image_url, src.image_hosts)
                raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                img = decode_image(raw_bytes)
            except Exception:
                reject_stats["layer3_download_or_decode"] += 1
                source_fail_streak[src_name] += 1
                continue
            finally:
                time.sleep(DOWNLOAD_DELAY)

            if min(img.size) < MIN_SIDE:
                reject_stats["layer3_too_small"] += 1
                source_fail_streak[src_name] += 1
                continue

            ok, probs = clip_gate(clip_model, clip_processor, device, img, gate_prompts)
            if not ok:
                reject_stats["layer4_clip_gate"] += 1
                source_fail_streak[src_name] += 1
                continue

            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=95)
            jpeg_bytes = buf.getvalue()
            stored_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

            if stored_sha256 in seen_hashes or raw_sha256 in seen_hashes:
                reject_stats["duplicate_hash"] += 1
                continue
            seen_hashes.add(stored_sha256)
            seen_hashes.add(raw_sha256)

            source_fail_streak[src_name] = 0

            asset_id = f"img_{uuid.uuid4().hex[:8]}"
            is_train = train_count < t_train
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
            with open(os.path.join(MANIFEST_DIR, f"manifest_{asset_id}.json"), "w", encoding="utf-8") as f:
                f.write(manifest.model_dump_json(indent=4))

            if is_train:
                train_count += 1
            else:
                val_count += 1

            curation["counts"][label]["train"] = train_count
            curation["counts"][label]["val"] = val_count
            curation["samples"].append({
                "path": file_path,
                "label": label,
                "class_idx": class_idx,
                "split": "train" if is_train else "val",
            })
            print(f"  ✅ [ACCEPTED] ({label} | {cand.source_name}) {file_path}")

        curation["rejections"][label] = dict(reject_stats)
        if train_count == 0 and val_count == 0 and reject_stats:
            print(f"  ⚠️ [Rejections for '{label}']: {dict(reject_stats)}")

    return curation


def print_curation_summary(curation, title="STAGE 2: CURATION SUMMARY"):
    print("\n==========================================")
    print(title)
    for label, counts in curation["counts"].items():
        rej = curation.get("rejections", {}).get(label, {})
        rej_str = f" | Rejections: {rej}" if rej else ""
        print(f"  Class '{label}': {counts['train']} Train | {counts['val']} Val{rej_str}")
    print(f"Total samples retrieved in this pass: {len(curation['samples'])}")
    print("==========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniCompiler Safe Image Retrieval Pipeline")
    parser.add_argument("-q", "--query", default=None, help="Target search query (e.g., 'cats and dogs')")
    parser.add_argument("-l", "--labels", default=None, help="Optional comma-separated class labels")
    parser.add_argument("--sources", default=",".join(SOURCES), help=f"Comma list of: {', '.join(SOURCES)}")
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN_COUNT, help="Number of training images per class")
    parser.add_argument("--val", type=int, default=DEFAULT_VAL_COUNT, help="Number of validation images per class")
    parser.add_argument("--verify", action="store_true", help="Run SHA-256 cryptographic integrity check")
    args = parser.parse_args()

    if args.verify:
        verify_dataset_integrity(MANIFEST_DIR)
        return

    query = (args.query or "").strip()
    if not query and not args.labels:
        query = input("Enter image search query for OmniCompiler (e.g., 'cats and dogs'): ").strip()
    if not query and not args.labels:
        raise SystemExit("Error: A search query or class labels are required.")

    chosen = [s.strip() for s in args.sources.split(",") if s.strip() in SOURCES]
    raw_labels = [args.labels] if args.labels else []
    clip = load_clip_gate()

    curation = run_curation(
        labels=raw_labels,
        dataset_query=query,
        target_train=args.train,
        target_val=args.val,
        sources=chosen,
        clip=clip,
    )
    print_curation_summary(curation)


if __name__ == "__main__":
    main()